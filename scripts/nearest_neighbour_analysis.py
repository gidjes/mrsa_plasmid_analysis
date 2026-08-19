import pandas as pd
import numpy as np
import subprocess
import shutil
import os
from pathlib import Path
from functools import partial
from multiprocessing import Pool, set_start_method
from helper_functions import lsf_hpcify_cmd
import config
from plotting_functions import plot_single_tangle

# ---------------------------------------------------------
# 5.0 Declare config variables
# ---------------------------------------------------------
ISOLATE_COL = config.ISOLATE_COL
ST_COL = config.ST_COL
SPECIES_COL = config.SPECIES_COL
ORIGIN_COL = config.ORIGIN_COL
MOBILITY_COL = config.MOBILITY_COL
CLUSTER_COL = config.CLUSTER_COLS


# ---------------------------------------------------------
# 5.1.1 Run mashtree for a single cluster
# ---------------------------------------------------------
def run_mashtree(cluster, df_in, cluster_col):
    df = df_in.copy()
    fastas = df["Plasmid"].loc[df[cluster_col] == cluster].to_list()

    print(f"Running plasmid mashtree for cluster {cluster}")
    path_pls = f"mashtree/{cluster}/pls"
    os.makedirs(path_pls, exist_ok=True)
    [shutil.copy2(f"fastas/{x}.fasta", f"{path_pls}") for x in fastas]
    mash_tree_cmd = f"conda run -n mash_master mashtree --mindepth 0 --numcpus 12 {path_pls}/*.fasta > mashtree/{cluster}_tree.dnd"
    mash_tree_cmd = lsf_hpcify_cmd(
        mash_tree_cmd, f"logs/mashtree/logs/{cluster}_mash.log", 12, 200, 3600
    )
    subprocess.call(
        f"{mash_tree_cmd}",
        shell=True,
    )

    print(f"Running chromosome mashtree for cluster {cluster}")
    fastas_chr = [f"{x.split('_')[0]}" for x in fastas]
    fastas_chr = list(set(fastas_chr))
    path_chr = f"mashtree/{cluster}/chr"
    os.makedirs(path_chr, exist_ok=True)
    [shutil.copy2(f"fastas_chr/{x}.fasta", f"{path_chr}") for x in fastas_chr]
    mash_tree_cmd_chr = f"conda run -n mash_master mashtree --mindepth 0 --numcpus 12 {path_chr}/*.fasta > mashtree/{cluster}_chr_tree.dnd"
    mash_tree_cmd_chr = lsf_hpcify_cmd(
        mash_tree_cmd_chr, f"logs/mashtree/logs/{cluster}_chr_mash.log", 12, 200, 3600
    )
    subprocess.call(
        f"{mash_tree_cmd_chr}",
        shell=True,
    )


# ---------------------------------------------------------
# 5.1.2 Run mashtree in parallel for each cluster
# ---------------------------------------------------------
def mashtree_builder(df_in: pd.DataFrame, n_jobs: int, cluster_col: str):
    df = df_in.copy()
    set_start_method("spawn")
    os.makedirs("logs", exist_ok=True)

    clusters = df[cluster_col].unique()

    with Pool(processes=n_jobs, maxtasksperchild=1) as pool:
        pool.map(partial(run_mashtree, df_in=df, plasmids=False), clusters)


# ---------------------------------------------------------
# 5.2.0 Helper functions
# ---------------------------------------------------------
def _load_mash(cluster: str, mode: str = "pls") -> Optional[pd.DataFrame]:
    path = Path(f"output/mash/dist/{cluster}_{mode}_mash_dist.tsv")
    if not path.exists():
        return None
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["subject", "target", "dist", "p_value", "hashes"],
    )
    df["subject"] = df["subject"].str.split("/").str[-1].str.split(".fas").str[0]
    df["target"] = df["target"].str.split("/").str[-1].str.split(".fas").str[0]
    df = df[df["subject"] != df["target"]].copy()
    return df


def _load_wgmlst(cluster: str, group: pd.DataFrame) -> Optional[pd.DataFrame]:
    # Always load chromosome Mash
    mash_df = _load_mash(cluster, mode="chr")
    if mash_df is None:
        return None

    mash_df = (
        mash_df[["subject", "target", "dist"]]
        .rename(
            columns={
                "subject": "isolate_a",
                "target": "isolate_b",
                "dist": "host_mash_dist",
            }
        )
        .pipe(lambda df: df[df["isolate_a"] < df["isolate_b"]])
        .reset_index(drop=True)
    )

    path = Path(f"output/wgmlst_2/{cluster}_distances.tab")
    if not path.exists():
        mash_df["host_wgmlst_dist"] = np.nan
        return mash_df

    wide = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    wide.index = wide.index.astype(str)
    wide.columns = wide.columns.astype(str)
    wide = wide.astype(float)

    plasmids = group["plasmid"].values
    isolates = [x.split("_")[0] for x in plasmids]

    if not set(isolates).issubset(wide.index):
        print("wgmlst incomplete")

    # Keep only isolates that are present
    wide = wide.loc[
        wide.index.intersection(isolates),
        wide.columns.intersection(isolates),
    ]

    wgmlst_df = (
        wide.stack()
        .reset_index()
        .rename(
            columns={
                wide.index.name: "isolate_a",
                "level_1": "isolate_b",
                0: "host_wgmlst_dist",
            }
        )
    )

    wgmlst_df = wgmlst_df[wgmlst_df["isolate_a"] < wgmlst_df["isolate_b"]]

    return mash_df.merge(
        wgmlst_df,
        on=["isolate_a", "isolate_b"],
        how="left",
    )


def _mash_cutoff(
    mash_df: pd.DataFrame, quantile: float = 0.25, use_cutoff: bool = True
) -> float:
    if use_cutoff:
        cutoff = float(np.quantile(mash_df["dist"], quantile))
        return cutoff
    else:
        return 0.0001


def _extract_isolate(plasmid_series: "pd.Series | str") -> "pd.Series | int":
    """Vectorised: '123_1' -> 123.  Also works on a plain string."""
    if isinstance(plasmid_series, str):
        return int(plasmid_series.split("_")[0])
    return plasmid_series.str.split("_").str[0].astype(int)


# ---------------------------------------------------------
# 5.2.1 Compare pairwise plasmid & host distance
# ---------------------------------------------------------
def _analyse_cluster_neighbours(
    outlier_plasmids: pd.Series,  # all outlier plasmids in this cluster
    evidence_tiers: pd.Series,  # aligned evidence tiers
    cluster_meta: pd.DataFrame,  # plasmid metadata for this cluster
    mash_df: pd.DataFrame,
    wgmlst_long: Optional[pd.DataFrame],
    mash_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fully vectorised replacement for the per-outlier loop.

    Returns (detail_df, summary_df) for all outliers in the cluster at once.
    """

    meta_cols = [
        ISOLATE_COL,
        ST_COL,
        SPECIES_COL,
        ORIGIN_COL,
    ]

    # Deduplicated isolate-level metadata
    isolate_meta = cluster_meta[meta_cols].drop_duplicates(subset="Parent")

    # ------------------------------------------------------------------ #
    # 1. Build (outlier_plasmid, neighbour_plasmid, mash_dist) pairs      #
    #    by filtering the full Mash table to only rows where the subject   #
    #    is one of our outliers and the distance is within cutoff.         #
    # ------------------------------------------------------------------ #
    outlier_set = set(outlier_plasmids)

    # Keep only edges where the subject is an outlier plasmid
    pairs = mash_df[
        mash_df["subject"].isin(outlier_set) & (mash_df["dist"] <= mash_cutoff)
    ][["subject", "target", "dist"]].copy()
    pairs.columns = ["outlier_plasmid", "neighbour_plasmid", "mash_plasmid_dist"]

    # Derive host isolate IDs (vectorised)
    pairs["outlier_isolate"] = _extract_isolate(pairs["outlier_plasmid"])
    pairs["neighbour_isolate"] = _extract_isolate(pairs["neighbour_plasmid"])

    # Drop self-isolate comparisons (same isolate, different replicon)
    pairs = pairs[
        pairs["outlier_isolate"] != pairs["neighbour_isolate"]
    ].drop_duplicates()

    # ------------------------------------------------------------------ #
    # 2. Attach outlier host metadata                                     #
    # ------------------------------------------------------------------ #
    pairs = pairs.merge(
        isolate_meta.rename(
            columns={
                ISOLATE_COL: "outlier_isolate",
                ST_COL: "outlier_st",
                SPECIES_COL: "outlier_species",
                ORIGIN_COL: "outlier_compartment",
            }
        ),
        on="outlier_isolate",
        how="left",
    )

    # ------------------------------------------------------------------ #
    # 3. Attach neighbour host metadata                                   #
    # ------------------------------------------------------------------ #
    pairs = pairs.merge(
        isolate_meta.rename(
            columns={
                ISOLATE_COL: "neighbour_isolate",
                ST_COL: "neighbour_st",
                SPECIES_COL: "neighbour_species",
                ORIGIN_COL: "neighbour_compartment",
            }
        ),
        on="neighbour_isolate",
        how="left",
    )

    # ------------------------------------------------------------------ #
    # 4. Vectorised flag columns                                          #
    # ------------------------------------------------------------------ #
    pairs["inter_species"] = pairs["neighbour_species"] != pairs["outlier_species"]
    pairs["same_st"] = pairs["neighbour_st"] == pairs["outlier_st"]
    pairs["inter_compartment"] = (
        pairs["neighbour_compartment"] != pairs["outlier_compartment"]
    )

    # ------------------------------------------------------------------ #
    # 5. Host distances (Mash + wgMLST)                                  #
    # ------------------------------------------------------------------ #
    if wgmlst_long is not None:
        iso_a = pairs[["outlier_isolate", "neighbour_isolate"]].min(axis=1).astype(str)
        iso_b = pairs[["outlier_isolate", "neighbour_isolate"]].max(axis=1).astype(str)
        lookup_keys = pd.DataFrame({"isolate_a": iso_a, "isolate_b": iso_b})

        host_dist = wgmlst_long.assign(
            isolate_a=wgmlst_long["isolate_a"].astype(str),
            isolate_b=wgmlst_long["isolate_b"].astype(str),
        )

        host_dist = lookup_keys.merge(
            host_dist,
            on=["isolate_a", "isolate_b"],
            how="left",
        )

        pairs["mash_host_dist"] = host_dist["host_mash_dist"].values
        pairs["wgmlst_host_dist"] = host_dist["host_wgmlst_dist"].values
    else:
        pairs["mash_host_dist"] = np.nan
        pairs["wgmlst_host_dist"] = np.nan

    pairs["wgmlst_available"] = pairs["wgmlst_host_dist"].notna()

    # ------------------------------------------------------------------ #
    # 6. Attach evidence tier (from the candidates DataFrame)             #
    # ------------------------------------------------------------------ #
    tier_map = dict(zip(outlier_plasmids, evidence_tiers))
    pairs["evidence_tier"] = pairs["outlier_plasmid"].map(tier_map)

    detail_df = pairs[
        [
            "outlier_plasmid",
            "outlier_isolate",
            "outlier_st",
            "outlier_species",
            "outlier_compartment",
            "neighbour_plasmid",
            "neighbour_isolate",
            "neighbour_st",
            "neighbour_species",
            "neighbour_compartment",
            "mash_plasmid_dist",
            "mash_host_dist",
            "wgmlst_host_dist",
            "wgmlst_available",
            "inter_species",
            "inter_compartment",
            "same_st",
            "evidence_tier",
        ]
    ].copy()

    return detail_df


# ---------------------------------------------------------
# 5.2.2 Build bin summaroes
# ---------------------------------------------------------
def _summarise_one(group: pd.DataFrame, cluster: str) -> dict:
    """Collapse one outlier's neighbour rows into a summary dict."""
    outlier_plasmid = group["outlier_plasmid"].iloc[0]
    evidence_tier = group["evidence_tier"].iloc[0]
    if group.empty:
        return {
            "cluster": cluster,
            "outlier_plasmid": outlier_plasmid,
            "evidence_tier": evidence_tier,
            "n_neighbours": 0,
            "n_distinct_hosts": 0,
            "n_same_st": 0,
            "n_diff_st": 0,
            "n_inter_species": 0,
            "n_inter_compartment": 0,
            # plasmid-level summary
            "median_mash_plasmid_dist": None,
            # host-level mash (always available)
            "median_mash_host_dist": None,
            # host-level wgMLST (optional)
            "median_wgmlst_host_dist": None,
            "max_wgmlst_host_dist": None,
            # coverage concept
            "wgmlst_coverage_pct": None,
            "dominant_neighbour_st": None,
            "inter_species_neighbour_species": None,
            "inter_compartment_neighbour_compartment": None,
        }

    wgmlst_avail = group[group["wgmlst_available"]]
    diff_st_mask = ~group["same_st"]

    dominant_st = None
    diff_st_series = group.loc[diff_st_mask, "neighbour_st"].dropna()
    if not diff_st_series.empty:
        dominant_st = diff_st_series.value_counts().idxmax()

    return {
        "cluster": cluster,
        "outlier_plasmid": outlier_plasmid,
        "evidence_tier": evidence_tier,
        "n_neighbours": len(group),
        "n_distinct_hosts": group["neighbour_isolate"].nunique(),
        "n_same_st": int(group["same_st"].sum()),
        "n_diff_st": int(diff_st_mask.sum()),
        "n_inter_species": int(group["inter_species"].sum()),
        "n_inter_compartment": int(group["inter_compartment"].sum()),
        # plasmid Mash
        "median_mash_plasmid_dist": float(group["mash_plasmid_dist"].median()),
        # host Mash (always present now)
        "median_mash_host_dist": float(group["mash_host_dist"].median()),
        "max_mash_host_dist": (float(group["mash_host_dist"].max())),
        # host wgMLST (optional)
        "median_wgmlst_host_dist": (
            float(wgmlst_avail["wgmlst_host_dist"].median())
            if not wgmlst_avail.empty
            else None
        ),
        "max_wgmlst_host_dist": (
            float(wgmlst_avail["wgmlst_host_dist"].max())
            if not wgmlst_avail.empty
            else None
        ),
        # TRUE coverage signal (important again)
        "wgmlst_coverage_pct": (
            round(100 * len(wgmlst_avail) / len(group), 1) if len(group) > 0 else None
        ),
        "dominant_neighbour_st": dominant_st,
        "inter_species_neighbour_species": (
            ", ".join(
                group[group["inter_species"]]["neighbour_species"]
                .dropna()
                .unique()
                .tolist()
            )
            or None
        ),
        "inter_compartment_neighbour_compartment": (
            ", ".join(
                group[group["inter_compartment"]]["neighbour_compartment"]
                .dropna()
                .unique()
                .tolist()
            )
            or None
        ),
    }


def _build_summaries(
    detail_df: pd.DataFrame,
    all_outliers: pd.DataFrame,  # columns: outlier_plasmid, evidence_tier, cluster
    cluster: str,
) -> pd.DataFrame:
    """
    Produce one summary row per outlier, including those with zero neighbours
    (which won't appear in detail_df after filtering).
    """
    rows = []
    if not detail_df.empty:
        seen = set()
        for name, grp in detail_df.groupby("outlier_plasmid", sort=False):
            rows.append(_summarise_one(grp, cluster))
            seen.add(name)
    else:
        seen = set()

    # Zero-neighbour outliers
    for _, row in all_outliers.iterrows():
        if row["outlier_plasmid"] not in seen:
            rows.append(
                {
                    "cluster": cluster,
                    "outlier_plasmid": row["outlier_plasmid"],
                    "evidence_tier": row["evidence_tier"],
                    "n_neighbours": 0,
                    "n_distinct_hosts": 0,
                    "n_same_st": 0,
                    "n_diff_st": 0,
                    "n_inter_species": 0,
                    "n_inter_compartment": 0,
                    "median_mash_plasmid_dist": None,
                    "n_inter_compartment": None,
                    "median_wgmlst_host_dist": None,
                    "wgmlst_coverage_pct": None,
                    "dominant_neighbour_st": None,
                    "inter_species_neighbour_species": None,
                    "inter_compartment_neighbour_compartment": None,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 5.2.3 Run the pairwise analysis pipeline
# ---------------------------------------------------------
def run_nearest_neighbour_analysis(
    hgt_candidates_df: pd.DataFrame,
    df_plasmids_clean: pd.DataFrame,
    mash_quantile: float = 0.25,
    use_cutoff: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorised nearest-neighbour host-distance analysis for all HGT candidates.

    Parameters
    ----------
    hgt_candidates_df : output of run_hgt_detection (columns: cluster, config.PLASMID_COL,
                        evidence_tier at minimum)
    df_plasmids_clean : full plasmid metadata (must contain ISOLATE_COL,
                        ST_COL, SPECIES_COL, ORIGIN_COL,
                        Standard_Cluster_mrsa)
    mash_quantile     : quantile of within-cluster Mash distances used as
                        neighbour cutoff (default 0.25 = lower quartile)

    Returns
    -------
    summary_df : one row per outlier plasmid
    detail_df  : one row per outlier-neighbour pair
    """
    if "Parent" not in df_plasmids_clean.columns:
        df_plasmids_clean = df_plasmids_clean.copy()
        df_plasmids_clean["Parent"] = _extract_isolate(df_plasmids_clean["Plasmid"])

    all_summaries: list[pd.DataFrame] = []
    all_details: list[pd.DataFrame] = []

    for cluster, group in hgt_candidates_df.groupby("cluster"):
        print(f"  Processing cluster {cluster} ({len(group)} outliers)...")

        mash_df = _load_mash(cluster, mode="pls")
        if mash_df is None:
            print(f"    [skip] no Mash file for {cluster}")
            continue
        wgmlst_long = _load_wgmlst(cluster, group)
        wgmlst_path = Path(f"output/wgmlst_2/{cluster}_distances.tab")
        chr_path = Path(f"output/mash/dist/{cluster}_chr_mash_dist.tsv")
        # wgmlst_source = (
        #     "wgmlst"
        #     if wgmlst_path.exists()
        #     else "mash_chr" if chr_path.exists() else "unavailable"
        # )

        if wgmlst_long is None:
            print(
                f"    [warn] no wgMLST file for {cluster}, host distances unavailable"
            )

        cutoff = _mash_cutoff(mash_df, quantile=mash_quantile, use_cutoff=use_cutoff)
        print(f"    Mash cutoff (q{int(mash_quantile * 100)}): {cutoff:.4f}")

        cluster_meta = df_plasmids_clean[
            df_plasmids_clean["Standard_Cluster_mrsa"] == str(cluster)
        ].copy()

        # ---- single vectorised call for the whole cluster ---- #
        detail = _analyse_cluster_neighbours(
            outlier_plasmids=group["plasmid"],
            evidence_tiers=group["evidence_tier"],
            cluster_meta=cluster_meta,
            mash_df=mash_df,
            wgmlst_long=wgmlst_long,
            mash_cutoff=cutoff,
        )

        # Summaries — one row per outlier (including zero-neighbour ones)
        outlier_index = group[["plasmid", "evidence_tier"]].rename(
            columns={"plasmid": "outlier_plasmid"}
        )
        summaries = _build_summaries(detail, outlier_index, cluster)

        all_summaries.append(summaries)
        if not detail.empty:
            all_details.append(detail)

    summary_df = (
        pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    )
    detail_df = (
        pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    )

    if not detail_df.empty:
        # ----------------------------------------------------------
        # Build graph → connected components = dissemination bins
        # ----------------------------------------------------------
        G = nx.Graph()
        for _, row in detail_df.iterrows():
            G.add_edge(row["outlier_plasmid"], row["neighbour_plasmid"])

        components = list(nx.connected_components(G))
        n_components = len(components)

        plasmid_to_bin = {}
        for i, component in enumerate(components):
            for plasmid in component:
                plasmid_to_bin[plasmid] = f"{cluster}_{i + 1}"

        detail_df["bin_id"] = detail_df["outlier_plasmid"].map(plasmid_to_bin)
        summary_df["bin_id"] = summary_df["outlier_plasmid"].map(plasmid_to_bin)

    if not summary_df.empty:
        tier_order = {"Strong": 0, "Moderate": 1, "Weak": 2}
        summary_df["_tier_rank"] = summary_df["evidence_tier"].map(tier_order)
        summary_df = (
            summary_df.sort_values(
                ["_tier_rank", "n_inter_species", "n_diff_st", "n_inter_compartment"],
                ascending=[True, False, False, False],
            )
            .drop(columns=["_tier_rank"])
            .reset_index(drop=True)
        )

    return summary_df, detail_df


# ---------------------------------------------------------
# 5.2.2 Build bin summaroes
# ---------------------------------------------------------
def run_nearest_neighbours_analysis_cluster(
    df_plasmids_clean, cluster_id, cluster_col, cutoff: bool = True
):

    os.makedirs(f"output/hgt_results/", exist_ok=True)
    os.makedirs(f"results/outlier_tangles/", exist_ok=True)

    df_cluster = df_plasmids_clean.loc[df_plasmids_clean[cluster_col] == cluster_id]

    hgt_candidates = df_cluster[["Plasmid", cluster_col]]
    hgt_candidates["evidence_tier"] = "None"
    hgt_candidates = hgt_candidates.rename(
        columns={"Plasmid": "plasmid", cluster_col: "cluster"}
    )

    nn_summary, nn_detail = run_nearest_neighbour_analysis(
        hgt_candidates_df=hgt_candidates,
        df_plasmids_clean=df_cluster,
        # cluster_col=cluster_col,
        mash_quantile=0.01,
        use_cutoff=cutoff,
    )

    nn_summary.to_csv(
        f"output/hgt_results/nn_summary_{cluster_id}.csv", index=False, sep=";"
    )
    nn_detail.to_csv(
        f"output/hgt_results/nn_detail_{cluster_id}.csv", index=False, sep=";"
    )

    nn_summary = pd.read_csv(f"output/hgt_results/nn_summary_{cluster_id}.csv", sep=";")

    try:
        nn_detail = pd.read_csv(
            f"output/hgt_results/nn_detail_{cluster_id}.csv", sep=";"
        )
    except pd.errors.EmptyDataError:
        nn_detail = pd.DataFrame()

    # fix/resolve parent column
    df_plasmids_clean_cluster = df_plasmids_clean.loc[
        df_plasmids_clean[cluster_col] == cluster_id
    ]
    nn_summary_cluster = nn_summary.loc[nn_summary["cluster"] == int(cluster_id)]

    def _classify_distance(row):
        """
        Priority:
            1. wgMLST if available
            2. fallback to Mash
        """

        # ----------------------------- #
        # WGMLST-based classification   #
        # ----------------------------- #
        if pd.notna(row.get("max_wgmlst_host_dist")):

            d = row["max_wgmlst_host_dist"]

            if d < 15:
                return "clonal"
            elif d < 500:
                return "close genogroup"
            elif d < 1000:
                return "wide genogroup"
            elif d < 2000:
                return "distinct lineage"
            else:
                return "maximal distant lineages"

        # ----------------------------- #
        # Mash fallback classification  #
        # ----------------------------- #
        if pd.notna(row.get("max_wgmlst_host_dist")):
            d = row["max_mash_host_dist"]

            if d < 0.00126:
                return "clonal"
            elif d < 0.00194:
                return "wide genogroup"
            elif d < 0.01:
                return "distinct lineage"
            else:
                return "maximal distant lineages"

    def classify_distance(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["group_width"] = df.apply(_classify_distance, axis=1)

        df["group_width"] = pd.Categorical(
            df["group_width"],
            categories=[
                "clonal",
                "close genogroup",
                "wide genogroup",
                "distinct lineage",
                "maximal distant lineages",
            ],
            ordered=True,
        )

        if "bin_id" not in df.columns:
            df["bin_id"] = "unbinned"

        return df

    nn_summary_cluster = classify_distance(nn_summary_cluster)

    tangle_df = df_plasmids_clean_cluster.merge(
        nn_summary_cluster,
        how="left",
        left_on="Plasmid",
        right_on="outlier_plasmid",
    )

    plot_single_tangle(
        tangle_df,
        cluster_id,
        "bin_id",
        ST_COL,
        "group_width",
        MOBILITY_COL,
        ORIGIN_COL,
    )


# ---------------------------------------------------------
# 5.2.3 Perfrom all the binning operations
# ---------------------------------------------------------
def run_nn_analysis(df_in: pd.DataFrame):
    cluster_col = CLUSTER_COLS
    df = df_in.copy()
    hgt_candidates = df[["Plasmid", cluster_col]]
    hgt_candidates["evidence_tier"] = "None"
    hgt_candidates = hgt_candidates.rename(
        columns={"Plasmid": "plasmid", cluster_col: "cluster"}
    )

    clusters = df[cluster_col].unique().tolist()

    for cluster in clusters:
        print(f"Running nearest neighbours for {cluster}\n")
        run_nearest_neighbours_analysis_cluster(df, cluster, cluster_col, False)


if __name__ == "__main__":
    print("Function related to the nearest_neighbour_analysis")
