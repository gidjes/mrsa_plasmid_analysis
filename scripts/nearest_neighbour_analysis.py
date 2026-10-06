import pandas as pd
import numpy as np
import subprocess
import shutil
import os
import networkx as nx
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from helper_functions import lsf_hpcify_cmd

import config
from plotting_functions import plot_single_tangle

# ---------------------------------------------------------
# 5.0 Configuration
# ---------------------------------------------------------
ISOLATE_COL = config.ISOLATE_COL
ST_COL = config.ST_COL
SPECIES_COL = config.SPECIES_COL
ORIGIN_COL = config.ORIGIN_COL
MOBILITY_COL = config.MOBILITY_COL
CLUSTER_COL = config.CLUSTER_COL

NJOBS = config.NJOBS


# ---------------------------------------------------------
# 5.1.1 Run mashtree for a single cluster
# ---------------------------------------------------------
def run_mashtree(cluster: str, df_in: pd.DataFrame):
    """
    Runs mashtree for the plasmids and isolates of a cluster

    Parameters
    ----------
    cluster : str
        cluster id for which to calculate mash distances
        and create the trees
    df_in : pd.DataFrame
        Dataframe containing plasmids and their associated cluster
    """
    df = df_in.copy()
    fastas = df["Plasmid"].loc[df[CLUSTER_COL] == cluster].to_list()
    outdir = f"output/mashtree/{cluster}"

    print(f"Running plasmid mashtree for cluster {cluster}")
    path_pls = f"{outdir}/{cluster}/pls"
    os.makedirs(path_pls, exist_ok=True)
    if not os.path.isfile(f"{outdir}/{cluster}_tree.dnd"):
        [shutil.copy2(f"fastas/{x}.fasta", f"{path_pls}") for x in fastas]
        mash_tree_cmd = f"'conda run -n mash_master mashtree --mindepth 0 --outmatrix {outdir}/{cluster}_pls_dist.tab --numcpus 12 {path_pls}/*.fasta > {outdir}/{cluster}_tree.dnd'"
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
    path_chr = f"{outdir}/{cluster}/chr"
    os.makedirs(path_chr, exist_ok=True)
    if not os.path.isfile(f"{outdir}/{cluster}_chr_tree.dnd"):
        [shutil.copy2(f"fastas_chr/{x}.fasta", f"{path_chr}") for x in fastas_chr]
        mash_tree_cmd_chr = f"'conda run -n mash_master mashtree --mindepth 0 --outmatrix {outdir}/{cluster}_chr_dist.tab --numcpus 12 {path_chr}/*.fasta > {outdir}/{cluster}_chr_tree.dnd'"
        mash_tree_cmd_chr = lsf_hpcify_cmd(
            mash_tree_cmd_chr,
            f"logs/mashtree/logs/{cluster}_chr_mash.log",
            12,
            200,
            3600,
        )
        subprocess.call(
            f"{mash_tree_cmd_chr}",
            shell=True,
        )


# ---------------------------------------------------------
# 5.1.2 Run mashtree in parallel for each cluster
# ---------------------------------------------------------
def mashtree_builder(df_in: pd.DataFrame):
    """
    Running function to set up run_mashtree for each cluster

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe with plasmids and clusters
    """
    df = df_in.copy()
    os.makedirs("logs", exist_ok=True)

    clusters = df[CLUSTER_COL].unique()

    with Pool(processes=NJOBS, maxtasksperchild=1) as pool:
        pool.map(partial(run_mashtree, df_in=df), clusters)


# ---------------------------------------------------------
# 5.1.3 convert wgMLST assignments to distance matrices
# ---------------------------------------------------------
def wgMLST_converter():
    """
    Convert the wgMLST output into distance matrices
    """
    os.makedirs("output/wgmlst/", exist_ok=True)

    cmd_sau = f"cgmlst-dists -j {NJOBS} -x 3000 data/wgMLST_sau.txt > output/wgmlst/wgMLST_sau_dist.tab"
    subprocess.call(
        cmd_sau,
        shell=True,
    )

    cmd_sar = f"cgmlst-dists -j {NJOBS} -x 3000 data/wgMLST_sar.txt > output/wgmlst/wgMLST_sar_dist.tab"
    subprocess.call(
        cmd_sar,
        shell=True,
    )


# ---------------------------------------------------------
# 5.1.4 Create wgMLST matrices for a cluster
# ---------------------------------------------------------
def wgMLST_per_cluster(
    df_in: pd.DataFrame, cluster_id: str, sau: pd.DataFrame, sar: pd.DataFrame
):
    """
    Create the wgMLST istance matrix for a specifc cluster

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe containing the plasmids associated isolate
        and their associated clusters
    cluster_id : str
        Cluster id to create the matrix for
    sau : pd.DataFrame
        S. aureus distance matrix
    sar : pd.DataFrame
        S. argenteus distance matrix
    """
    df = df_in.copy()

    # Get isolate IDs belonging to this cluster
    isolate_ids = (
        df.loc[df[CLUSTER_COL] == cluster_id, "Parent"].dropna().unique().tolist()
    )

    # 4. Select only isolates from this cluster
    sau_isolates = [x for x in isolate_ids if x in sau.index]
    sar_isolates = [x for x in isolate_ids if x in sar.index]

    sau_cluster = sau.loc[sau_isolates, sau_isolates]
    sar_cluster = sar.loc[sar_isolates, sar_isolates]

    # 5. Combine the two matrices
    # The union contains all isolates in the cluster from either species.
    all_isolates = sau_isolates + [x for x in sar_isolates if x not in sau_isolates]

    combined = pd.DataFrame(index=all_isolates, columns=all_isolates, dtype=float)

    # Put S. aureus distances into the combined matrix
    if not sau_cluster.empty:
        combined.loc[sau_isolates, sau_isolates] = sau_cluster

    # Put S. argenteus distances into the combined matrix
    if not sar_cluster.empty:
        combined.loc[sar_isolates, sar_isolates] = sar_cluster

    # 6. Save the cluster-specific matrix
    output_file = f"output/wgmlst/{cluster_id}_distances.tab"
    combined.to_csv(output_file, sep="\t", na_rep="")

    print(
        f"Cluster {cluster_id}: "
        f"{len(all_isolates)} isolates "
        f"({len(sau_isolates)} S. aureus, "
        f"{len(sar_isolates)} S. argenteus) -> {output_file}"
    )


# ---------------------------------------------------------
# 5.1.5 Runner to create wgMLST matrices by cluster
# ---------------------------------------------------------
def wgMLST_prepper(df_in: pd.DataFrame):
    """
    Runner function to convert wgMLST outputs into distance matrices
    for each cluster

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe to collect isolates associated with each plasmid
        cluster
    """
    df = df_in.copy()

    # Input files
    sau_file = "output/wgmlst/wgMLST_sau_dist.tab"
    sar_file = "output/wgmlst/wgMLST_sar_dist.tab"

    # 1. Get all unique cluster values
    cluster_ids = df[CLUSTER_COL].dropna().unique()

    # 2. Read the two distance matrices
    sau = pd.read_csv(sau_file, sep="\t", index_col=0)
    sar = pd.read_csv(sar_file, sep="\t", index_col=0)

    # Make sure isolate IDs are strings
    df["Parent"] = df["Parent"].astype(str)
    sau.index = sau.index.astype(str)
    sau.columns = sau.columns.astype(str)

    sar.index = sar.index.astype(str)
    sar.columns = sar.columns.astype(str)

    with Pool(processes=NJOBS, maxtasksperchild=1) as pool:
        pool.map(partial(wgMLST_per_cluster, df_in=df, sau=sau, sar=sar), cluster_ids)


# ---------------------------------------------------------
# 5.2.0 Helper functions
# ---------------------------------------------------------
def _load_mash(cluster: str, mode: str = "pls") -> pd.DataFrame:
    """
    Load the mash distance matrix

    Parameters
    ----------
    cluster : str
        cluster for which to open the file
    mode : str, optional
        open plasmid or chromosomal file, by default "pls"

    Returns
    -------
    pd.DataFrame
        Pairwise (long format) distance matrix
    """
    path = Path(f"output/mashtree/{cluster}/{cluster}_{mode}_dist.tab")
    if not path.exists():
        return None

    wide = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    wide.index = wide.index.astype(str)
    wide.columns = wide.columns.astype(str)
    wide = wide.astype(float)

    dataset = wide.index.values

    if mode == "pls":
        col_name = "mash_dist"
    elif mode == "chr":
        dataset = [x.split("_")[0] for x in dataset]
        col_name = "host_mash_dist"
    else:
        print("incorrect mash loading mode selected")
        return None

    if not set(dataset).issubset(wide.index):
        print("mash incomplete")

    # Keep only isolates that are present
    wide = wide.loc[
        wide.index.intersection(dataset),
        wide.columns.intersection(dataset),
    ]

    mash_df = (
        wide.stack()
        .reset_index()
        .rename(
            columns={
                wide.index.name: "isolate_a",
                "level_1": "isolate_b",
                0: col_name,
            }
        )
    )

    return mash_df


def _load_wgmlst(cluster: str, group: pd.DataFrame) -> pd.DataFrame:
    """
    Load the wgMLST distance matrix

    Parameters
    ----------
    cluster : str
        Cluster id for which to load the matrix
    group : pd.DataFrame
        Dataframe containing the analysis group

    Returns
    -------
    pd.DataFrame
        Pairwise (long format) distance matrix
    """

    # Always load chromosome Mash
    mash_df = _load_mash(cluster, mode="chr")
    if mash_df is None:
        return None

    path = Path(f"output/wgmlst/{cluster}_distances.tab")
    if not path.exists():
        mash_df["host_wgmlst_dist"] = np.nan
        return mash_df

    wide = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    wide.index = wide.index.astype(str)
    wide.columns = wide.columns.astype(str)
    wide = wide.astype(float)

    plasmids = group["Plasmid"].values
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
                "level_0": "isolate_a",
                "level_1": "isolate_b",
                0: "host_wgmlst_dist",
            }
        )
    )

    # wgmlst_df = wgmlst_df[wgmlst_df["isolate_a"] < wgmlst_df["isolate_b"]]

    return mash_df.merge(
        wgmlst_df,
        on=["isolate_a", "isolate_b"],
        how="left",
    )


def _extract_isolate(plasmid_series: "pd.Series | str") -> "pd.Series | int":
    """Convert plasmid ID to isolate ID"""
    if isinstance(plasmid_series, str):
        return int(plasmid_series.split("_")[0])
    return plasmid_series.str.split("_").str[0].astype(int)


# ---------------------------------------------------------
# 5.2.1 Compare pairwise plasmid & host distance
# ---------------------------------------------------------
def _analyse_cluster_neighbours(
    outlier_plasmids: pd.Series,  # all outlier plasmids in this cluster
    cluster_meta: pd.DataFrame,  # plasmid metadata for this cluster
    mash_df: pd.DataFrame,
    wgmlst_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the pairwise metadata comparisons
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
    # 1. Build (outlier_plasmid, neighbour_plasmid, mash_dist) pairs     #
    #    by filtering the full Mash table to only rows where the subject #
    #    is one of our outliers and the distance is within cutoff.       #
    # ------------------------------------------------------------------ #
    outlier_set = set(outlier_plasmids)

    # Keep only edges where the subject is an outlier plasmid
    pairs = mash_df[mash_df["isolate_a"].isin(outlier_set)][
        ["isolate_a", "isolate_b", "mash_dist"]
    ].copy()
    pairs.columns = ["outlier_plasmid", "neighbour_plasmid", "mash_plasmid_dist"]

    # Derive host isolate IDs (vectorised)
    pairs["outlier_isolate"] = _extract_isolate(pairs["outlier_plasmid"])
    pairs["neighbour_isolate"] = _extract_isolate(pairs["neighbour_plasmid"])

    # Drop self-isolate comparisons
    pairs = pairs[
        pairs["outlier_plasmid"] != pairs["neighbour_plasmid"]
    ].drop_duplicates()

    # ------------------------------------------------------------------ #
    # 2. Attach outlier host metadata                                    #
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
    pairs["is_bin"] = pairs["mash_plasmid_dist"] < 0.0001
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

    detail_df = (
        pairs[
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
                "is_bin",
                "mash_host_dist",
                "wgmlst_host_dist",
                "wgmlst_available",
                "inter_species",
                "inter_compartment",
                "same_st",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    return detail_df


# ---------------------------------------------------------
# 5.2.2 Build bin summaries
# ---------------------------------------------------------
def _summarise_one(group: pd.DataFrame, cluster: str) -> dict:
    """
    Summarise the pairwise analysis into a one-line summary for
    a plasmid
    """
    outlier_plasmid = group["outlier_plasmid"].values[0]
    binned_group = group.loc[group["is_bin"]]

    if binned_group.empty:
        return {
            CLUSTER_COL: cluster,
            "outlier_plasmid": outlier_plasmid,
            "n_neighbours": 0,
            "n_distinct_hosts": 0,
            "n_same_st": 0,
            "n_diff_st": 0,
            "n_inter_species": 0,
            "n_inter_compartment": 0,  # plasmid-level summary
            "median_mash_plasmid_dist": None,  # host-level mash (always available)
            "median_mash_host_dist": None,  # host-level wgMLST (optional)
            "median_wgmlst_host_dist": None,
            "max_wgmlst_host_dist": None,  # coverage concept
            "wgmlst_coverage_pct": None,
            "dominant_neighbour_st": None,
            "inter_species_neighbour_species": None,
            "inter_compartment_neighbour_compartment": None,
        }

    wgmlst_avail = binned_group[binned_group["wgmlst_available"]]
    diff_st_mask = ~binned_group["same_st"]
    dominant_st = None
    diff_st_series = binned_group.loc[diff_st_mask, "neighbour_st"].dropna()
    if not diff_st_series.empty:
        dominant_st = diff_st_series.value_counts().idxmax()

    return {
        CLUSTER_COL: cluster,
        "outlier_plasmid": outlier_plasmid,
        "n_neighbours": len(binned_group),
        "n_distinct_hosts": binned_group["neighbour_isolate"].nunique(),
        "n_same_st": int(binned_group["same_st"].sum()),
        "n_diff_st": int(diff_st_mask.sum()),
        "n_inter_species": int(binned_group["inter_species"].sum()),
        "n_inter_compartment": int(
            binned_group["inter_compartment"].sum()
        ),  # plasmid Mash
        "median_mash_plasmid_dist": float(
            binned_group["mash_plasmid_dist"].median()
        ),  # host Mash (always present now)
        "median_mash_host_dist": float(binned_group["mash_host_dist"].median()),
        "max_mash_host_dist": (
            float(binned_group["mash_host_dist"].max())
        ),  # host wgMLST (optional)
        "median_wgmlst_host_dist": (
            float(wgmlst_avail["wgmlst_host_dist"].median())
            if not wgmlst_avail.empty
            else None
        ),
        "max_wgmlst_host_dist": (
            float(wgmlst_avail["wgmlst_host_dist"].max())
            if not wgmlst_avail.empty
            else None
        ),  # TRUE coverage signal (important again)
        "wgmlst_coverage_pct": (
            round(100 * len(wgmlst_avail) / len(binned_group), 1)
            if len(binned_group) > 0
            else None
        ),
        "dominant_neighbour_st": dominant_st,
        "inter_species_neighbour_species": (
            ", ".join(
                binned_group[binned_group["inter_species"]]["neighbour_species"]
                .dropna()
                .unique()
                .tolist()
            )
            or None
        ),
        "inter_compartment_neighbour_compartment": (
            ", ".join(
                binned_group[binned_group["inter_compartment"]]["neighbour_compartment"]
                .dropna()
                .unique()
                .tolist()
            )
            or None
        ),
    }


def _build_summaries(
    detail_df: pd.DataFrame,
    all_outliers: pd.DataFrame,
    cluster: str,
) -> pd.DataFrame:
    """
    Produce one summary row per outlier, including those with zero neighbours
    """
    rows = []
    if not detail_df.empty:
        seen = set()
        for name, grp in detail_df.groupby("outlier_plasmid", sort=False):
            rows.append(_summarise_one(grp, cluster))
            seen.add(name)
    else:
        seen = set()  # Zero-neighbour outliers

    for _, row in all_outliers.iterrows():
        if row["outlier_plasmid"] not in seen:
            rows.append(
                {
                    CLUSTER_COL: cluster,
                    "outlier_plasmid": row["outlier_plasmid"],
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
    df_plasmids_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorised nearest-neighbour host-distance analysis for all HGT candidates.

    Parameters
    ----------
    df_plasmids_clean : full plasmid metadata

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
    cluster_seq_df = df_plasmids_clean[["Plasmid", CLUSTER_COL]]

    for cluster, group in cluster_seq_df.groupby(CLUSTER_COL):
        print(f"  Processing cluster {cluster} ({len(group)} plasmids)...")

        mash_df = _load_mash(cluster, mode="pls")
        if mash_df is None:
            print(f"    [skip] no Mash file for {cluster}")
            continue

        wgmlst_long = _load_wgmlst(cluster, group)
        if wgmlst_long is None:
            print(
                f"    [warn] no wgMLST file for {cluster}, host distances unavailable"
            )

        cluster_meta = df_plasmids_clean[
            df_plasmids_clean["Standard_Cluster_mrsa"] == str(cluster)
        ].copy()

        # single vectorised call for the whole cluster
        detail = _analyse_cluster_neighbours(
            outlier_plasmids=group["Plasmid"],
            cluster_meta=cluster_meta,
            mash_df=mash_df,
            wgmlst_long=wgmlst_long,
        )

        # Summaries — one row per outlier (including zero-neighbour ones)
        outlier_index = group[["Plasmid"]].rename(
            columns={"Plasmid": "outlier_plasmid"}
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

    # Build graph → connected components = dissemination bins
    if not detail_df.empty:
        G = nx.Graph()

        mask = detail_df["is_bin"]

        for _, row in detail_df.loc[mask].iterrows():
            G.add_edge(row["outlier_plasmid"], row["neighbour_plasmid"])

        components = list(nx.connected_components(G))

        plasmid_to_bin = {}
        for i, component in enumerate(components):
            for plasmid in component:
                plasmid_to_bin[plasmid] = f"{cluster}_{i + 1}"

        # Map bins back to all pairwise relationships
        outlier_bin = detail_df["outlier_plasmid"].map(plasmid_to_bin)
        neighbour_bin = detail_df["neighbour_plasmid"].map(plasmid_to_bin)

        detail_df["bin_id"] = outlier_bin.where(outlier_bin.eq(neighbour_bin))

        # Map bins to the one-row-per-plasmid summary
        summary_df["bin_id"] = summary_df["outlier_plasmid"].map(plasmid_to_bin)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["n_inter_species", "n_diff_st", "n_inter_compartment"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return summary_df, detail_df


# ---------------------------------------------------------
# 5.2.2 Build bin summaroes
# ---------------------------------------------------------
def run_nearest_neighbours_analysis_cluster(
    df_plasmids_clean: pd.DataFrame, cluster_id: str
):
    """
    Create the bins of near-identical plasmids and summarise
    bin metadata distribution

    Parameters
    ----------
    df_plasmids_clean : pd.DataFrame
        Plasmid metadata dataframe
    cluster_id : str
        Cluster for which to run the analysis
    """
    os.makedirs(f"output/hgt_results/", exist_ok=True)
    os.makedirs(f"results/outlier_tangles/", exist_ok=True)

    df_cluster = df_plasmids_clean.loc[df_plasmids_clean[CLUSTER_COL] == cluster_id]

    nn_summary, nn_detail = run_nearest_neighbour_analysis(
        df_plasmids_clean=df_cluster,
    )

    nn_summary.to_csv(
        f"output/hgt_results/nn_summary_{cluster_id}.csv", index=False, sep=";"
    )
    nn_detail.to_csv(
        f"output/hgt_results/nn_detail_{cluster_id}.csv", index=False, sep=";"
    )

    # fix/resolve parent column
    df_plasmids_clean_cluster = df_plasmids_clean.loc[
        df_plasmids_clean[CLUSTER_COL] == cluster_id
    ]
    nn_summary_cluster = nn_summary.loc[nn_summary[CLUSTER_COL] == cluster_id]

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
                return "Clonal"
            elif d < 1000:
                return "Same genogroup"
            else:
                return "Distant lineages"

        # ----------------------------- #
        # Mash fallback classification  #
        # ----------------------------- #
        if pd.notna(row.get("max_wgmlst_host_dist")):
            d = row["max_mash_host_dist"]

            if d < 0.00051:
                return "Clonal"
            elif d < 0.0021892:
                return "Same genogroup"
            else:
                return "Distant lineages"

    def classify_distance(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["group_width"] = df.apply(_classify_distance, axis=1)

        df["group_width"] = pd.Categorical(
            df["group_width"],
            categories=[
                "Clonal",
                "Same genogroup",
                "Distant lineages",
            ],
            ordered=True,
        )

        if "bin_id" not in df.columns:
            df["bin_id"] = "unbinned"
        else:
            df["bin_id"] = df["bin_id"].fillna(value="unbinned")

        return df

    nn_summary_cluster = classify_distance(nn_summary_cluster)

    tangle_df = df_plasmids_clean_cluster.merge(
        nn_summary_cluster,
        how="left",
        left_on=["Plasmid", "Standard_Cluster_mrsa"],
        right_on=["outlier_plasmid", "Standard_Cluster_mrsa"],
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
    """
    Runner function for near-identical bin analysis
    for each cluster

    Parameters
    ----------
    df_in : pd.DataFrame
        Plasmid metadata
    """
    df = df_in.copy()
    clusters = df[CLUSTER_COL].unique().tolist()

    with Pool(processes=NJOBS, maxtasksperchild=1) as pool:
        pool.map(
            partial(run_nearest_neighbours_analysis_cluster, df_plasmids_clean=df),
            clusters,
        )


if __name__ == "__main__":
    print("Function related to the nearest_neighbour_analysis")
