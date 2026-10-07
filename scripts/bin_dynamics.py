import glob
import os
import re
from itertools import combinations
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

import config

# ============================================================
# CONSTANTS
# ============================================================
WGMLST_CLONAL = 15
WGMLST_GENOGROUP = 1000
MIN_POSTINTRO_ISOLATES = 3

CAT_CLONAL = "clonal"
CAT_GENOGROUP = "genogroup"
CAT_DISTANT = "distant"
CAT_UNKNOWN = "unknown"
CAT_RANK = {CAT_CLONAL: 0, CAT_GENOGROUP: 1, CAT_DISTANT: 2, CAT_UNKNOWN: -1}

ISOLATE_ID_COL = "Parent"
CLUSTER_COL = config.CLUSTER_COL
ST_COL = config.ST_COL
DATE_COL = config.DATE_COL
ACCESSORY_GENES = [config.AMR_COL, config.VIR_COL, config.METAL_COL, config.BIOCIDE_COL]


# ============================================================
# DISTANCE HELPERS
# ============================================================
def categorise_wgmlst(dist):
    """Map a numeric wgMLST distance to a category string."""
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return CAT_UNKNOWN
    if d < WGMLST_CLONAL:
        return CAT_CLONAL
    elif d < WGMLST_GENOGROUP:
        return CAT_GENOGROUP
    else:
        return CAT_DISTANT


def categorise_mash(dist, mash_clonal, mash_genogroup):
    """Map a numeric mash distance to a category using calibrated thresholds."""
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return CAT_UNKNOWN
    if d < mash_clonal:
        return CAT_CLONAL
    elif d < mash_genogroup:
        return CAT_GENOGROUP
    else:
        return CAT_DISTANT


def categorise_edge(dist, source, mash_clonal, mash_genogroup):
    """Dispatch to the correct categoriser based on source metric."""
    src = str(source).strip().lower()
    if "wgmlst" in src:
        return categorise_wgmlst(dist)
    elif "mash" in src:
        return categorise_mash(dist, mash_clonal, mash_genogroup)
    return CAT_UNKNOWN


def worst_case_category(categories):
    """
    Return the single worst-case distance category from an iterable.
    distant > genogroup > clonal > unknown.
    """
    best_rank = -1
    best_cat = CAT_UNKNOWN
    for c in categories:
        r = CAT_RANK.get(c, -1)
        if r > best_rank:
            best_rank = r
            best_cat = c
    return best_cat


# ============================================================
# CROSS-CLUSTER MASH CALIBRATION
# ============================================================
def derive_mash_cutoffs(all_pairwise_df):
    """
    Calibrate mash host-distance cut-offs against wgMLST categories.

    Uses explicit dual-metric pairwise table:
        - mash_host_dist (always present)
        - wgmlst_host_dist (optional, ground truth when available)

    Calibration pairs are defined as rows where wgMLST exists.
    """

    pw = all_pairwise_df.copy()

    # ------------------------------------------------------------ #
    # 1. Define calibration pairs (wgMLST available = ground truth) #
    # ------------------------------------------------------------ #
    cal = pw[pw["wgmlst_host_dist"].notna()].copy()

    if cal.empty:
        print(
            "  [WARN] No wgMLST calibration pairs found. "
            "Using conservative defaults."
        )
        return 0.005, 0.05, 0

    cal["wgmlst_host_dist"] = pd.to_numeric(cal["wgmlst_host_dist"], errors="coerce")

    cal = cal.dropna(subset=["wgmlst_host_dist", "mash_host_dist"])

    calibration_n = len(cal)

    if calibration_n == 0:
        print(
            "  [WARN] No valid numeric wgMLST/mash calibration pairs. "
            "Using conservative defaults."
        )
        return 0.005, 0.05, 0

    # ------------------------------------------------------------ #
    # 2. Categorise wgMLST (ground truth)                          #
    # ------------------------------------------------------------ #
    cal["wgmlst_cat"] = cal["wgmlst_host_dist"].apply(categorise_wgmlst)

    MIN_CAL = 10
    mash_clonal = mash_genogroup = None

    # ------------------------------------------------------------ #
    # 3. Calibrate mash thresholds                                 #
    # ------------------------------------------------------------ #
    clonal_mash = cal.loc[
        cal["wgmlst_cat"] == CAT_CLONAL,
        "mash_host_dist",
    ]

    genogroup_mash = cal.loc[
        cal["wgmlst_cat"].isin([CAT_CLONAL, CAT_GENOGROUP]),
        "mash_host_dist",
    ]

    if len(clonal_mash) >= MIN_CAL:
        mash_clonal = float(clonal_mash.quantile(0.95))
    else:
        print(
            f"  [WARN] Only {len(clonal_mash)} clonal calibration pairs "
            f"(need {MIN_CAL}); using default."
        )

    if len(genogroup_mash) >= MIN_CAL:
        mash_genogroup = float(genogroup_mash.quantile(0.95))
    else:
        print(
            f"  [WARN] Only {len(genogroup_mash)} genogroup calibration pairs "
            f"(need {MIN_CAL}); using default."
        )

    mash_clonal = mash_clonal if mash_clonal is not None else 0.005
    mash_genogroup = mash_genogroup if mash_genogroup is not None else 0.05

    print(
        f"  Mash cut-offs calibrated from {calibration_n} wgMLST pairs:\n"
        f"    clonal < {mash_clonal:.5f}  |  genogroup < {mash_genogroup:.5f} | distant >= {mash_genogroup:.5}"
    )

    # ------------------------------------------------------------ #
    # 4. Save calibration set                                      #
    # ------------------------------------------------------------ #
    cal.to_csv(
        f"results/bin_dynamics/mash_calibration_pairs.csv",
        sep=";",
        index=False,
    )

    return mash_clonal, mash_genogroup, calibration_n


# ============================================================
# GENERAL HELPERS
# ============================================================
def empirical_pvalue(observed, null_dist, alternative="greater"):
    """
    Phipson & Smyth empirical p-value.
    null_dist must have exactly n_perm entries (zeros for permutations
    in which the event never occurred). [F1]
    """
    null_dist = np.asarray(null_dist)
    if alternative == "greater":
        return (np.sum(null_dist >= observed) + 1) / (len(null_dist) + 1)
    elif alternative == "less":
        return (np.sum(null_dist <= observed) + 1) / (len(null_dist) + 1)
    else:
        return (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (len(null_dist) + 1)


def find_cluster_files(detail_dir="output/hgt_results"):
    """Return sorted (cluster_id, detail_path, summary_path) tuples."""
    detail_paths = sorted(glob.glob(os.path.join(detail_dir, "nn_detail_*.csv")))
    pairs = []
    for dp in detail_paths:
        m = re.search(r"nn_detail_(\w+)\.csv$", dp)
        if not m:
            continue
        cluster_id = m.group(1)
        sp = os.path.join(detail_dir, f"nn_summary_{cluster_id}.csv")
        if not os.path.exists(sp):
            print(f"  [WARN] No summary file for cluster {cluster_id}, skipping.")
            continue
        pairs.append((cluster_id, dp, sp))
    return pairs


# ============================================================
# PER-CLUSTER PROCESSING
# ============================================================


def process_cluster(
    cluster_id, detail_path, summary_path, plasmid_df, mash_clonal, mash_genogroup
):
    """
    Process one cluster's detail file.

    Distance column : wgmlst_host_dist  (always this name regardless of metric)
    Metric source   : host_dist_source  ('wgmlst' or 'mash')

    Each edge is categorised independently as clonal / genogroup /
    distant (mutually exclusive: genogroup = 15 <= d < 1000 strictly,
    distant = d >= 1000).  Each bin is assigned a worst-case
    promiscuity level (distant > genogroup > clonal).

    Returns
    -------
    bin_summaries  : list[dict]
    plasmid_to_bin : dict  {plasmid_id: bin_id}
    pairwise_df    : DataFrame  (annotated)
    n_components   : int
    """
    try:
        pairwise_df = pd.read_csv(detail_path, sep=";")
    except pd.errors.EmptyDataError:
        return [], {}, pd.DataFrame(), 0

    if pairwise_df.empty:
        return [], {}, pairwise_df, 0

    # ── Distance categorisation ───────────────────────────────
    pairwise_df["wgmlst_host_dist"] = pd.to_numeric(
        pairwise_df["wgmlst_host_dist"], errors="coerce"
    )

    pairwise_df["host_dist_source"] = pairwise_df["wgmlst_available"].map(
        {
            False: "mash",
            True: "wgmlst",
        }
    )

    pairwise_df["is_mash"] = ~pairwise_df["wgmlst_available"]

    pairwise_df["dist_category"] = pairwise_df.apply(
        lambda r: categorise_edge(
            r["wgmlst_host_dist"], r["host_dist_source"], mash_clonal, mash_genogroup
        ),
        axis=1,
    )

    # ── Lookups ───────────────────────────────────────────────
    comp_lookup = plasmid_df.set_index("Plasmid")["origin"].to_dict()

    # ── Build graph → connected components [F8] ───────────────
    G = nx.Graph()

    mask = pairwise_df["is_bin"]

    for _, row in pairwise_df.loc[mask].iterrows():
        G.add_edge(row["outlier_plasmid"], row["neighbour_plasmid"])

    components = list(nx.connected_components(G))
    n_components = len(components)

    plasmid_to_bin = {
        plasmid: f"{cluster_id}_{i + 1}"
        for i, component in enumerate(components)
        for plasmid in component
    }
    outlier_bin = pairwise_df["outlier_plasmid"].map(plasmid_to_bin)
    neighbour_bin = pairwise_df["neighbour_plasmid"].map(plasmid_to_bin)

    pairwise_df["bin_id"] = outlier_bin.where(outlier_bin.eq(neighbour_bin))

    # ── Per-bin summaries ─────────────────────────────────────
    bin_summaries = []

    for i, members in enumerate(components):
        bin_id = f"{cluster_id}_{i + 1}"
        sub = pairwise_df[pairwise_df["bin_id"] == bin_id].copy()
        meta = plasmid_df[plasmid_df["Plasmid"].isin(members)]
        # Compartment set
        comp_set = {
            comp_lookup.get(pid)
            for pid in members
            if comp_lookup.get(pid) not in (None, "Unknown", "", "nan")
        }

        # Unique isolates: prefer explicit isolate columns in detail file
        isolate_set = set()
        if "outlier_isolate" in sub.columns and "neighbour_isolate" in sub.columns:
            isolate_set |= set(sub["outlier_isolate"].dropna().astype(str))
            isolate_set |= set(sub["neighbour_isolate"].dropna().astype(str))
        isolate_set -= {"nan", "None", "Unknown", ""}

        # Accessory gene counts
        gene_counts = {}
        for GENE_TYPE in ACCESSORY_GENES:
            col = f"{GENE_TYPE}_plasmid"
            n_gene = int(meta[col].sum()) if col in meta.columns else 0
            gene_counts[f"n_{GENE_TYPE}_plasmids"] = n_gene
            gene_counts[f"contains_{GENE_TYPE}"] = n_gene > 0

        # [F3] Deduplicate edges — canonical direction: outlier <= neighbour
        sub_dedup = sub[
            sub["outlier_plasmid"] <= sub["neighbour_plasmid"]
        ].drop_duplicates(subset=["outlier_plasmid", "neighbour_plasmid"])

        # Per-category edge counts (mutually exclusive categories)
        cat_counts = sub_dedup["dist_category"].value_counts().to_dict()
        n_clonal_edges = int(cat_counts.get(CAT_CLONAL, 0))
        n_genogroup_edges = int(cat_counts.get(CAT_GENOGROUP, 0))
        n_distant_edges = int(cat_counts.get(CAT_DISTANT, 0))
        n_unknown_edges = int(cat_counts.get(CAT_UNKNOWN, 0))
        n_mash_edges = int(sub_dedup["is_mash"].sum())

        # Worst-case promiscuity level
        promiscuity_level = worst_case_category(list(sub_dedup["dist_category"]))

        # Compartment pair crossings
        comp_pair_edges = set()
        for _, r in sub_dedup.iterrows():
            cu = comp_lookup.get(r["outlier_plasmid"], "Unknown") or "Unknown"
            cv = comp_lookup.get(r["neighbour_plasmid"], "Unknown") or "Unknown"
            if cu != cv and cu != "Unknown" and cv != "Unknown":
                comp_pair_edges.add(tuple(sorted([cu, cv])))

        # Graph metrics
        bin_G = nx.Graph()
        for _, r in sub.iterrows():
            bin_G.add_edge(r["outlier_plasmid"], r["neighbour_plasmid"])

        n_edges = bin_G.number_of_edges()
        density = nx.density(bin_G) if len(members) > 1 else np.nan

        mash_vals = (
            sub_dedup["mash_plasmid_dist"].dropna()
            if "mash_plasmid_dist" in sub_dedup.columns
            else pd.Series(dtype=float)
        )

        summary = {
            "cluster_id": str(cluster_id),
            "bin_id": bin_id,
            "bin_size": len(members),
            "n_unique_isolates": len(isolate_set),
            # Compartment
            "n_compartments": len(comp_set),
            "inter_compartment": len(comp_set) > 1,
            "compartment_list": sorted(comp_set),
            "n_compartment_pair_edges": len(comp_pair_edges),
            # Host distance (primary axis)
            "n_clonal_edges": n_clonal_edges,
            "n_genogroup_edges": n_genogroup_edges,
            "n_distant_edges": n_distant_edges,
            "n_unknown_dist_edges": n_unknown_edges,
            "n_mash_dist_edges": n_mash_edges,
            "has_genogroup_edge": n_genogroup_edges > 0,
            "has_distant_edge": n_distant_edges > 0,
            "promiscuity_level": promiscuity_level,
            "max_host_dist": sub_dedup["wgmlst_host_dist"].max(),
            "median_host_dist": sub_dedup["wgmlst_host_dist"].median(),
            # Graph
            "n_edges": n_edges,
            "graph_density": density,
            "median_mash_plasmid_dist": (
                mash_vals.median() if len(mash_vals) > 0 else np.nan
            ),
            # Member list (needed for enrichment)
            "member_plasmids": list(members),
        }
        summary.update(gene_counts)
        bin_summaries.append(summary)

    return bin_summaries, plasmid_to_bin, pairwise_df, n_components


# ============================================================
# CLUSTER-LEVEL REPORT
# ============================================================
def cluster_level_report(summary_df, cluster_n_components):
    """
    Per-cluster summary exported to cluster_level_report.csv.
    """

    def _cluster_stats(grp):
        n = len(grp)
        return pd.Series(
            {
                "n_bins": n,
                "n_bins_clonal_only": int(
                    (grp["promiscuity_level"] == CAT_CLONAL).sum()
                ),
                "n_bins_genogroup": int(grp["has_genogroup_edge"].sum()),
                "n_bins_distant": int(grp["has_distant_edge"].sum()),
                "n_bins_inter_compartment": int(grp["inter_compartment"].sum()),
                "n_bins_amr": int(grp["contains_amr"].sum()),
                "n_bins_virulence": int(grp["contains_virulence"].sum()),
                "n_bins_biocide": (
                    int(grp["contains_biocide"].sum())
                    if "contains_biocide" in grp
                    else 0
                ),
                "n_bins_distant_amr": int(
                    (grp["has_distant_edge"] & grp["contains_amr"]).sum()
                ),
                "n_bins_intercomp_amr": int(
                    (grp["inter_compartment"] & grp["contains_amr"]).sum()
                ),
                "median_bin_size": grp["bin_size"].median(),
                "max_bin_size": grp["bin_size"].max(),
                "max_host_dist_any_bin": grp["max_host_dist"].max(),
                "median_n_unique_isolates": grp["n_unique_isolates"].median(),
            }
        )

    if not summary_df.empty:
        cluster_stats = (
            summary_df.groupby("cluster_id", sort=False)
            .apply(_cluster_stats)
            .reset_index()
        )
    else:
        cluster_stats = pd.DataFrame(columns=["cluster_id"])

    report_rows = []
    for cid in sorted(cluster_n_components.keys()):
        n_comp = cluster_n_components[cid]
        row = {"cluster_id": cid, "has_bins": n_comp > 0, "n_components_raw": n_comp}

        if (
            n_comp > 0
            and not cluster_stats.empty
            and cid in cluster_stats["cluster_id"].values
        ):
            stats = cluster_stats.loc[cluster_stats["cluster_id"] == cid].iloc[0]
            row.update(stats.drop("cluster_id").to_dict())
        else:
            row.update(
                {
                    "n_bins": 0,
                    "n_bins_clonal_only": 0,
                    "n_bins_genogroup": 0,
                    "n_bins_distant": 0,
                    "n_bins_inter_compartment": 0,
                    "n_bins_amr": 0,
                    "n_bins_virulence": 0,
                    "n_bins_biocide": 0,
                    "n_bins_distant_amr": 0,
                    "n_bins_intercomp_amr": 0,
                    "median_bin_size": np.nan,
                    "max_bin_size": np.nan,
                    "max_host_dist_any_bin": np.nan,
                    "median_n_unique_isolates": np.nan,
                }
            )

        n_bins = row["n_bins"]
        for col in [
            "n_bins_genogroup",
            "n_bins_distant",
            "n_bins_inter_compartment",
            "n_bins_amr",
            "n_bins_virulence",
        ]:
            frac = col.replace("n_bins_", "pct_bins_")
            row[frac] = round(row[col] / n_bins * 100, 2) if n_bins > 0 else np.nan

        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    n_total = len(report_df)
    n_with = int(report_df["has_bins"].sum())

    text_file = "results/bin_dynamics/cluster_bin_counts.txt"
    with open(text_file, "w") as f:
        f.write(f"Total clusters processed   : {n_total}")
        f.write(f"  With bins                : {n_with}")
        f.write(f"  Without bins             : {n_total - n_with}")
        if n_with > 0:
            w = report_df[report_df["has_bins"]]
            f.write(f"\nAmong clusters WITH bins:")
            f.write(f"  Median bins/cluster      : {w['n_bins'].median():.1f}")
            f.write(f" (range {int(w['n_bins'].min())}–{int(w['n_bins'].max())})")
            f.write(
                f"  Clusters with ≥1 distant bin    : {(w['n_bins_distant'] > 0).sum()}"
            )
            f.write(
                f"  Clusters with ≥1 genogroup bin  : {(w['n_bins_genogroup'] > 0).sum()}"
            )
            f.write(
                f"  Clusters with ≥1 inter-comp bin : {(w['n_bins_inter_compartment'] > 0).sum()}"
            )

    return report_df


# ============================================================
# ACCESSORY GENE ENRICHMENT
# ============================================================
def accessory_gene_enrichment(summary_df, plasmid_df):
    bin_plasmid_ids = {
        pid for members in summary_df["member_plasmids"] for pid in members
    }

    pdf = plasmid_df.copy()
    pdf["in_bin"] = pdf["Plasmid"].isin(bin_plasmid_ids)

    # Attach bin-level worst-case category per plasmid
    plasmid_to_distant = {}
    for _, row in summary_df.iterrows():
        for pid in row["member_plasmids"]:
            plasmid_to_distant[pid] = row["has_distant_edge"]

    pdf["bin_has_distant"] = pdf["Plasmid"].map(plasmid_to_distant).fillna(False)

    n_total = len(pdf)
    n_in_bin = int(pdf["in_bin"].sum())

    rows = []
    for GENE_TYPE in ACCESSORY_GENES:
        col = f"{GENE_TYPE}_plasmid"
        if col not in pdf.columns:
            continue

        n_gene_total = int(pdf[col].sum())
        n_gene_in_bin = int(pdf.loc[pdf["in_bin"], col].sum())
        n_nogene_in_bin = int((pdf["in_bin"] & ~pdf[col]).sum())
        n_gene_not_bin = n_gene_total - n_gene_in_bin
        n_nogene_not_bin = (n_total - n_gene_total) - n_nogene_in_bin

        pct_gene_dataset = round(n_gene_total / n_total * 100, 2) if n_total else np.nan
        pct_gene_bins = round(n_gene_in_bin / n_in_bin * 100, 2) if n_in_bin else np.nan

        # Test A: in bin vs. not in bin
        if (
            n_gene_in_bin + n_gene_not_bin > 0
            and n_nogene_in_bin + n_nogene_not_bin > 0
        ):
            _, p_bin = fisher_exact(
                [[n_gene_in_bin, n_nogene_in_bin], [n_gene_not_bin, n_nogene_not_bin]],
                alternative="greater",
            )
        else:
            p_bin = np.nan

        # Test B: among binned plasmids, distant bin vs. non-distant
        binned = pdf[pdf["in_bin"]].copy()
        a2 = int((binned[col] & binned["bin_has_distant"]).sum())
        b2 = int((~binned[col] & binned["bin_has_distant"]).sum())
        c2 = int((binned[col] & ~binned["bin_has_distant"]).sum())
        d2 = int((~binned[col] & ~binned["bin_has_distant"]).sum())
        if a2 + c2 > 0 and b2 + d2 > 0:
            _, p_distant = fisher_exact([[a2, b2], [c2, d2]], alternative="greater")
        else:
            p_distant = np.nan

        rows.append(
            {
                "gene_type": GENE_TYPE,
                "n_total_dataset": n_total,
                "n_gene_in_dataset": n_gene_total,
                "pct_gene_in_dataset": pct_gene_dataset,
                "n_binned_plasmids": n_in_bin,
                "n_gene_in_bins": n_gene_in_bin,
                "pct_gene_in_bins": pct_gene_bins,
                "fisher_p_in_bin": p_bin,
                "fisher_p_distant_vs_other": p_distant,
            }
        )

    enrich_df = pd.DataFrame(rows)

    # Joint FDR across both columns of p-values
    all_p_vals = (
        enrich_df["fisher_p_in_bin"].tolist()
        + enrich_df["fisher_p_distant_vs_other"].tolist()
    )
    valid_mask = [not np.isnan(p) for p in all_p_vals]
    padj_all = np.full(len(all_p_vals), np.nan)
    valid_p = [p for p, v in zip(all_p_vals, valid_mask) if v]
    if valid_p:
        padj_valid = multipletests(valid_p, method="fdr_bh")[1]
        j = 0
        for idx, v in enumerate(valid_mask):
            if v:
                padj_all[idx] = padj_valid[j]
                j += 1
    n = len(enrich_df)
    enrich_df["padj_in_bin"] = padj_all[:n]
    enrich_df["padj_distant_vs_other"] = padj_all[n:]

    enrich_df.to_csv(
        f"results/bin_dynamics/accessory_group_bin_distance.csv",
        sep=";",
        index=False,
    )
    return enrich_df


# ============================================================
# COMPARTMENT ANALYSIS
# ============================================================
def compartment_analysis(summary_df, plasmid_df, n_perm=2000, seed=42):
    comp_lookup = plasmid_df.set_index("Plasmid")["origin"].to_dict()

    bin_members = {
        row["bin_id"]: row["member_plasmids"] for _, row in summary_df.iterrows()
    }

    # ── 1. Pairwise co-occurrence counts ──────────────────────
    pair_counts = Counter()
    for members in bin_members.values():
        comps = {
            comp_lookup.get(p, "Unknown")
            for p in members
            if comp_lookup.get(p, "Unknown") not in ("Unknown", "", None)
        }
        for pair in combinations(sorted(comps), 2):
            pair_counts[pair] += 1

    pair_df = (
        pd.DataFrame(
            [(p[0], p[1], n) for p, n in pair_counts.most_common()],
            columns=["compartment_A", "compartment_B", "n_bins_co_occurring"],
        )
        if pair_counts
        else pd.DataFrame(
            columns=["compartment_A", "compartment_B", "n_bins_co_occurring"]
        )
    )

    # ── Exports ───────────────────────────────────────────────
    pair_df.to_csv(
        f"results/bin_dynamics/compartment_cooccurrence_counts.csv",
        sep=";",
        index=False,
    )

    return pair_df


# ============================================================
# HOST DISTANCE ANALYSIS
# ============================================================
def host_distance_analysis(summary_df, all_pairwise_df):
    df = summary_df.copy()

    # ── Inter-compartment vs. intra-compartment ───────────────
    inter = df.loc[df["inter_compartment"], "max_host_dist"].dropna()
    intra = df.loc[~df["inter_compartment"], "max_host_dist"].dropna()
    stat, p = mannwhitneyu(inter, intra, alternative="two-sided")

    text_file = "results/bin_dynamics/inter_compartment_distance"
    with open(text_file, "w") as f:
        f.write(f"Max host dist — inter- vs. intra-compartment bins:\n")
        f.write(f"  Inter (n={len(inter)}): median={inter.median():.1f}\n")
        f.write(f"  Intra (n={len(intra)}): median={intra.median():.1f}\n")
        f.write(f"  Mann-Whitney U={stat:.0f}, p={p:.4f}\n")


# ============================================================
# AGGREGATION & REPORTING
# ============================================================
def aggregate_and_report(
    summary_df,
    all_plasmid_to_bin,
    plasmid_df,
    cluster_n_components,
    all_pairwise_df,
):
    cluster_level_report(summary_df, cluster_n_components)

    bin_plasmid_ids = set(all_plasmid_to_bin.keys())
    bin_meta = plasmid_df[plasmid_df["Plasmid"].isin(bin_plasmid_ids)].copy()

    n_bins = len(summary_df)
    n_plasmids = len(bin_plasmid_ids)
    n_isolates = int(len(bin_meta["Parent"].unique()))
    n_clusters = int(len(bin_meta[config.CLUSTER_COL].unique()))
    med_size = summary_df["bin_size"].median()
    q1 = summary_df["bin_size"].quantile(0.25)
    q3 = summary_df["bin_size"].quantile(0.75)
    n_intercomp = int(summary_df["inter_compartment"].sum())

    # ── Sub-analyses ──────────────────────────────────────────
    accessory_gene_enrichment(summary_df, plasmid_df)
    compartment_analysis(summary_df, plasmid_df)
    host_distance_analysis(summary_df, all_pairwise_df)

    # ── Master summary ─────────────────────────────────────────
    def _n(col):
        return int(summary_df[col].sum()) if col in summary_df.columns else np.nan

    bin_meta = bin_meta.merge(
        summary_df, how="left", left_on="bin_id", right_on="bin_id"
    )

    posthoc_summary = pd.DataFrame(
        {
            "metric": [
                "n_bins",
                "n_plasmids_in_bins",
                "n_unique_isolates_spanned",
                "median_bin_size",
                "bin_size_q1",
                "bin_size_q3",
                "n_bins_clonal_only",
                "n_bins_genogroup",
                "n_bins_distant",
                "n_bins_inter_compartment",
                "n_bins_inter_compartment_geno",
                "n_bins_inter_compartment_dist",
                "n_amr_bins",
                "n_virulence_bins",
                "n_biocide_bins",
                "n_metal_bins",
                "n_amr_bins_genogroup",
                "n_vir_bins_genogroup",
                "n_bio_bins_genogroup",
                "n_met_bins_genogroup",
                "n_amr_bins_distant",
                "n_virulence_bins_distant",
                "n_bio_bins_distant",
                "n_met_bins_distant",
                "n_amr_bins_inter_compartment",
                "n_vir_bins_inter_compartment",
                "n_bio_bins_inter_compartment",
                "n_met_bins_inter_compartment",
                "n_amr_bins_inter_compartment_genogroup",
                "n_vir_bins_inter_compartment_genogroup",
                "n_bio_bins_inter_compartment_genogroup",
                "n_met_bins_inter_compartment_genogroup",
                "n_amr_bins_inter_compartment_distant",
                "n_vir_bins_inter_compartment_distant",
                "n_bio_bins_inter_compartment_distant",
                "n_met_bins_inter_compartment_distant",
            ],
            "n_bins": [
                n_bins,
                n_plasmids,
                n_isolates,
                med_size,
                q1,
                q3,
                int((summary_df["promiscuity_level"] == CAT_CLONAL).sum()),
                int(
                    (
                        summary_df["has_genogroup_edge"]
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(summary_df["has_distant_edge"].sum()),
                n_intercomp,
                int(
                    (
                        summary_df["has_genogroup_edge"]
                        & ~summary_df["has_distant_edge"]
                        & summary_df["inter_compartment"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["has_distant_edge"] & summary_df["inter_compartment"]
                    ).sum()
                ),
                _n("contains_amr"),
                _n("contains_virulence"),
                _n("contains_biocide"),
                _n("contains_metal"),
                int(
                    (
                        summary_df["contains_amr"]
                        & (summary_df["has_genogroup_edge"])
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_virulence"]
                        & (summary_df["has_genogroup_edge"])
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_biocide"]
                        & (summary_df["has_genogroup_edge"])
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_metal"]
                        & (summary_df["has_genogroup_edge"])
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (summary_df["contains_amr"] & summary_df["has_distant_edge"]).sum()
                ),
                int(
                    (
                        summary_df["contains_virulence"]
                        & summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_biocide"] & summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_metal"] & summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (summary_df["contains_amr"] & summary_df["inter_compartment"]).sum()
                ),
                int(
                    (
                        summary_df["contains_virulence"]
                        & summary_df["inter_compartment"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_biocide"] & summary_df["inter_compartment"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_metal"] & summary_df["inter_compartment"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_amr"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_genogroup_edge"]
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_virulence"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_genogroup_edge"]
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_biocide"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_genogroup_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_metal"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_genogroup_edge"]
                        & ~summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_amr"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_virulence"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_biocide"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        summary_df["contains_metal"]
                        & summary_df["inter_compartment"]
                        & summary_df["has_distant_edge"]
                    ).sum()
                ),
            ],
            "n_plasmids": [
                n_bins,
                n_plasmids,
                n_isolates,
                med_size,
                q1,
                q3,
                int((bin_meta["promiscuity_level"] == CAT_CLONAL).sum()),
                int(
                    (
                        bin_meta["has_genogroup_edge"] & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(bin_meta["has_distant_edge"].sum()),
                int(bin_meta["inter_compartment"].sum()),
                int(
                    (
                        bin_meta["has_genogroup_edge"]
                        & ~bin_meta["has_distant_edge"]
                        & bin_meta["inter_compartment"]
                    ).sum()
                ),
                int(
                    (bin_meta["has_distant_edge"] & bin_meta["inter_compartment"]).sum()
                ),
                _n("contains_amr"),
                _n("contains_virulence"),
                _n("contains_biocide"),
                _n("contains_metal"),
                int(
                    (
                        bin_meta["contains_amr"]
                        & (bin_meta["has_genogroup_edge"])
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_virulence"]
                        & (bin_meta["has_genogroup_edge"])
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_biocide"]
                        & (bin_meta["has_genogroup_edge"])
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_metal"]
                        & (bin_meta["has_genogroup_edge"])
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int((bin_meta["contains_amr"] & bin_meta["has_distant_edge"]).sum()),
                int(
                    (
                        bin_meta["contains_virulence"] & bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (bin_meta["contains_biocide"] & bin_meta["has_distant_edge"]).sum()
                ),
                int((bin_meta["contains_metal"] & bin_meta["has_distant_edge"]).sum()),
                int((bin_meta["contains_amr"] & bin_meta["inter_compartment"]).sum()),
                int(
                    (
                        bin_meta["contains_virulence"] & bin_meta["inter_compartment"]
                    ).sum()
                ),
                int(
                    (bin_meta["contains_biocide"] & bin_meta["inter_compartment"]).sum()
                ),
                int((bin_meta["contains_metal"] & bin_meta["inter_compartment"]).sum()),
                int(
                    (
                        bin_meta["contains_amr"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_genogroup_edge"]
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_virulence"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_genogroup_edge"]
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_biocide"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_genogroup_edge"]
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_metal"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_genogroup_edge"]
                        & ~bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_amr"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_virulence"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_biocide"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_distant_edge"]
                    ).sum()
                ),
                int(
                    (
                        bin_meta["contains_metal"]
                        & bin_meta["inter_compartment"]
                        & bin_meta["has_distant_edge"]
                    ).sum()
                ),
            ],
        }
    )
    posthoc_summary["%_bins"] = posthoc_summary["n_bins"] / n_bins
    posthoc_summary["%_plasmids"] = posthoc_summary["n_plasmids"] / len(
        plasmid_df["Plasmid"].unique()
    )

    # ------------------------------------------------------------------
    # 2. Write the first six metrics to a text file
    # ------------------------------------------------------------------
    text_file = "results/bin_dynamics/bin_summary.txt"

    first_six = posthoc_summary.iloc[:6]

    with open(text_file, "w") as f:
        f.write(f"Number of bins: {int(first_six.loc[0, 'n_bins'])}\n")
        f.write(f"Number of plasmids in bins: " f"{int(first_six.loc[1, 'n_bins'])}\n")
        f.write(
            f"Number of unique isolates spanned: "
            f"{int(first_six.loc[2, 'n_bins'])}\n"
        )
        f.write(f"Median bin size: " f"{first_six.loc[3, 'n_bins']:.2f}\n")
        f.write(f"Bin size Q1: " f"{first_six.loc[4, 'n_bins']:.2f}\n")
        f.write(f"Bin size Q3: " f"{first_six.loc[5, 'n_bins']:.2f}\n")
    # ------------------------------------------------------------------
    # 3. Helper to extract values from posthoc_summary
    # ------------------------------------------------------------------

    def get_value(metric, column="n_bins"):
        """Return a value from posthoc_summary."""
        matches = posthoc_summary.loc[posthoc_summary["metric"] == metric, column]

        if matches.empty:
            raise KeyError(
                f"Metric {metric!r} not found in posthoc_summary. "
                f"Available metrics are: "
                f"{posthoc_summary['metric'].tolist()}"
            )

        return matches.iloc[0]

    # ------------------------------------------------------------------
    # 4. Construct the publication-style table
    # ------------------------------------------------------------------

    rows = []

    def add_row(
        label,
        scope,
        total_metric,
        genogroup_metric,
        distant_metric,
    ):
        """
        Add one row to the summary table.

        Clonal is calculated as:
            Total - Genogroup - Distant

        The metric names passed to this function are the exact names
        used in posthoc_summary.
        """

        total = int(get_value(total_metric))
        genogroup = int(get_value(genogroup_metric))
        distant = int(get_value(distant_metric))

        clonal = total - genogroup - distant

        if clonal < 0:
            raise ValueError(
                f"Negative clonal count for {label} ({scope}): "
                f"total={total}, genogroup={genogroup}, distant={distant}"
            )

        rows.append(
            {
                "Category": label,
                "Scope": scope,
                "Clonal_N": clonal,
                "Clonal_%": clonal / total if total else 0,
                "Genogroup_N": genogroup,
                "Genogroup_%": genogroup / total if total else 0,
                "Distant_N": distant,
                "Distant_%": distant / total if total else 0,
                "Total_N": total,
                "Total_%": 1.0 if total else 0,
            }
        )

    # ------------------------------------------------------------------
    # Total bins
    # ------------------------------------------------------------------

    add_row(
        "Total bins",
        "All",
        "n_bins",
        "n_bins_genogroup",
        "n_bins_distant",
    )

    add_row(
        "Total bins",
        "Inter-compartment",
        "n_bins_inter_compartment",
        "n_bins_inter_compartment_geno",
        "n_bins_inter_compartment_dist",
    )

    # ------------------------------------------------------------------
    # ARG-carrying
    # ------------------------------------------------------------------

    add_row(
        "ARG-carrying",
        "All",
        "n_amr_bins",
        "n_amr_bins_genogroup",
        "n_amr_bins_distant",
    )

    add_row(
        "ARG-carrying",
        "Inter-compartment",
        "n_amr_bins_inter_compartment",
        "n_amr_bins_inter_compartment_genogroup",
        "n_amr_bins_inter_compartment_distant",
    )

    # ------------------------------------------------------------------
    # VAG-carrying
    # ------------------------------------------------------------------

    add_row(
        "VAG-carrying",
        "All",
        "n_virulence_bins",
        "n_vir_bins_genogroup",
        "n_virulence_bins_distant",
    )

    add_row(
        "VAG-carrying",
        "Inter-compartment",
        "n_vir_bins_inter_compartment",
        "n_vir_bins_inter_compartment_genogroup",
        "n_vir_bins_inter_compartment_distant",
    )

    # ------------------------------------------------------------------
    # MRG-carrying
    # ------------------------------------------------------------------

    add_row(
        "MRG-carrying",
        "All",
        "n_metal_bins",
        "n_met_bins_genogroup",
        "n_met_bins_distant",
    )

    add_row(
        "MRG-carrying",
        "Inter-compartment",
        "n_met_bins_inter_compartment",
        "n_met_bins_inter_compartment_genogroup",
        "n_met_bins_inter_compartment_distant",
    )

    # ------------------------------------------------------------------
    # BRG-carrying
    # ------------------------------------------------------------------

    add_row(
        "BRG-carrying",
        "All",
        "n_biocide_bins",
        "n_bio_bins_genogroup",
        "n_bio_bins_distant",
    )

    add_row(
        "BRG-carrying",
        "Inter-compartment",
        "n_bio_bins_inter_compartment",
        "n_bio_bins_inter_compartment_genogroup",
        "n_bio_bins_inter_compartment_distant",
    )

    table = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Format N / % columns
    # ------------------------------------------------------------------

    formatted_table = table.copy()

    for col in [
        "Clonal_%",
        "Genogroup_%",
        "Distant_%",
        "Total_%",
    ]:
        formatted_table[col] = formatted_table[col].map(lambda x: f"{x:.1%}")

    for col in [
        "Clonal_N",
        "Genogroup_N",
        "Distant_N",
        "Total_N",
    ]:
        formatted_table[col] = formatted_table[col].astype(int)

    # ------------------------------------------------------------------
    # 7. Save the table
    # ------------------------------------------------------------------

    table_file = "results/tables/table1_bin_overview.csv"

    # Use the simple formatted version for TSV output
    table.to_csv(
        table_file,
        sep=";",
        index=False,
    )


# ============================================================
# HELPERS
# ============================================================
def _parse_date(series):
    """Parse dd-mm-yyyy date strings to datetime; NaT on failure."""
    return pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")


def _parse_genes(series):
    """
    Split comma-separated gene strings into individual gene names.
    Returns a list of non-empty stripped strings per cell.
    """
    return (
        series.fillna("")
        .astype(str)
        .apply(lambda x: [g.strip() for g in x.split(",") if g.strip()])
    )


def _canonical_pair(a, b):
    """Return a sorted tuple so (a,b) and (b,a) map to the same key."""
    return tuple(sorted([str(a), str(b)]))


def _build_distance_lookup(mash_df, wgmlst_df):
    """
    Build two dicts keyed by canonical isolate pair:
        mash_lookup   : {(a, b): mash_dist}
        wgmlst_lookup : {(a, b): wgmlst_dist}
    """
    mash_lookup = {
        _canonical_pair(r.outlier_isolate, r.neighbour_isolate): r.mash_host_dist
        for r in mash_df.itertuples()
        if pd.notna(r.mash_host_dist)
    }
    wgmlst_lookup = {
        _canonical_pair(r.outlier_isolate, r.neighbour_isolate): r.wgmlst_host_dist
        for r in wgmlst_df.itertuples()
        if pd.notna(r.wgmlst_host_dist)
    }
    return mash_lookup, wgmlst_lookup


# ============================================================
# 1. PER-GENE ENRICHMENT
# ============================================================
def per_gene_enrichment(summary_df, plasmid_df):
    # Plasmids in promiscuous bins (genogroup or distant edge)
    promiscuous_bin_ids = set(
        summary_df.loc[
            summary_df["has_genogroup_edge"] | summary_df["has_distant_edge"],
            "bin_id",
        ]
    )
    distant_bin_ids = set(summary_df.loc[summary_df["has_distant_edge"], "bin_id"])

    pdf = plasmid_df.copy()
    pdf["in_promiscuous_bin"] = pdf["bin_id"].isin(promiscuous_bin_ids)
    pdf["in_distant_bin"] = pdf["bin_id"].isin(distant_bin_ids)

    n_total = len(pdf)
    n_promiscuous = int(pdf["in_promiscuous_bin"].sum())

    rows = []

    for GENE_TYPE in ACCESSORY_GENES:
        if GENE_TYPE not in pdf.columns:
            continue

        pdf[f"_genes_{GENE_TYPE}"] = _parse_genes(pdf[GENE_TYPE])

        # All distinct gene names in this column
        all_genes = [g for genes in pdf[f"_genes_{GENE_TYPE}"] for g in genes]
        gene_counts_total = Counter(all_genes)

        for gene_name, n_gene_total in gene_counts_total.items():
            has_gene = pdf[f"_genes_{GENE_TYPE}"].apply(lambda gs: gene_name in gs)

            n_gene_promiscuous = int((has_gene & pdf["in_promiscuous_bin"]).sum())
            n_nogene_promiscuous = int((~has_gene & pdf["in_promiscuous_bin"]).sum())
            n_gene_not = n_gene_total - n_gene_promiscuous
            n_nogene_not = (n_total - n_gene_total) - n_nogene_promiscuous

            pct_dataset = round(n_gene_total / n_total * 100, 3) if n_total else np.nan
            pct_promiscuous = (
                round(n_gene_promiscuous / n_promiscuous * 100, 3)
                if n_promiscuous
                else np.nan
            )

            if (
                n_gene_promiscuous + n_gene_not > 0
                and n_nogene_promiscuous + n_nogene_not > 0
            ):
                _, p = fisher_exact(
                    [
                        [n_gene_promiscuous, n_nogene_promiscuous],
                        [n_gene_not, n_nogene_not],
                    ],
                    alternative="greater",
                )
            else:
                p = np.nan

            rows.append(
                {
                    "gene_type": GENE_TYPE,
                    "gene_name": gene_name,
                    "n_in_dataset": n_gene_total,
                    "pct_in_dataset": pct_dataset,
                    "n_in_promiscuous_bins": n_gene_promiscuous,
                    "pct_in_promiscuous_bins": pct_promiscuous,
                    "fisher_p": p,
                }
            )

    enrich_df = pd.DataFrame(rows)
    valid_mask = enrich_df["fisher_p"].notna()
    padj = np.full(len(enrich_df), np.nan)
    if valid_mask.sum() > 0:
        padj[valid_mask] = multipletests(
            enrich_df.loc[valid_mask, "fisher_p"], method="fdr_bh"
        )[1]
    enrich_df["padj"] = padj
    enrich_df = enrich_df.sort_values("padj")

    enrich_df.to_csv(
        f"results/tables/tableS7_bin_gene_enrichment.csv", sep=";", index=False
    )
    return enrich_df


# ============================================================
# 2. SPATIOTEMPORAL SPREAD INDEX
# ============================================================
def spatiotemporal_spread_index(summary_df, plasmid_df):
    pdf = plasmid_df.copy()
    pdf["_date"] = _parse_date(pdf["MATERIAL_SAMPLINGDATE"])

    spread_rows = []
    for _, bin_row in summary_df.iterrows():
        members = bin_row["member_plasmids"]
        meta = pdf[pdf["Plasmid"].isin(members)]

        dates = meta["_date"].dropna()
        munis = meta["municipality"].dropna().astype(str).str.strip()
        munis = munis[munis != ""]

        date_range = (dates.max() - dates.min()).days if len(dates) >= 2 else np.nan
        n_munis = munis.nunique()
        spread_score = (
            date_range * n_munis if (pd.notna(date_range) and n_munis > 0) else np.nan
        )

        row = {
            "bin_id": bin_row["bin_id"],
            "bin_size": bin_row["bin_size"],
            "promiscuity_level": bin_row["promiscuity_level"],
            "has_distant_edge": bin_row["has_distant_edge"],
            "has_genogroup_edge": bin_row["has_genogroup_edge"],
            "max_host_dist": bin_row["max_host_dist"],
            "n_dated_members": int(len(dates)),
            "earliest_date": dates.min() if len(dates) > 0 else pd.NaT,
            "latest_date": dates.max() if len(dates) > 0 else pd.NaT,
            "date_range_days": date_range,
            "n_municipalities": n_munis,
            "spread_score": spread_score,
        }

        spread_rows.append(row)

    spread_df = pd.DataFrame(spread_rows)

    # ── Spread vs. host distance ──────────────────────────────
    valid = spread_df[["spread_score", "max_host_dist"]].dropna()
    rho, p = spearmanr(valid["spread_score"], valid["max_host_dist"])

    text_file = "results/bin_dynamics/spearmans_rho.txt"
    with open(text_file, "w") as f:
        f.write(f"\nSpearman: spread_score (date_range * n_muni) ~ max_host_dist\n")
        f.write(f"rho={rho:.4f}, p={p:.4f}  (n={len(valid)} bins)")

    # ── Distant vs. non-distant bin spread ────────────────────────
    comparisons = [
        (
            "Distant",
            "Non-distant",
            spread_df["has_distant_edge"],
            ~spread_df["has_distant_edge"],
        ),
        (
            "Distant",
            "Genogroup",
            spread_df["has_distant_edge"],
            spread_df["has_genogroup_edge"] & ~spread_df["has_distant_edge"],
        ),
        (
            "Genogroup",
            "Clonal",
            spread_df["has_genogroup_edge"] & ~spread_df["has_distant_edge"],
            ~spread_df["has_genogroup_edge"] & ~spread_df["has_distant_edge"],
        ),
    ]

    results = []

    for metric in ["date_range_days", "n_municipalities", "spread_score"]:

        for group1_name, group2_name, group1_mask, group2_mask in comparisons:

            group1 = spread_df.loc[group1_mask, metric].dropna()
            group2 = spread_df.loc[group2_mask, metric].dropna()

            if len(group1) >= 2 and len(group2) >= 2:
                stat, p = mannwhitneyu(group1, group2, alternative="two-sided")

                results.append(
                    {
                        "Metric": metric,
                        "Comparison": f"{group1_name} vs. {group2_name}",
                        "Group 1": group1_name,
                        "n1": len(group1),
                        "Median 1": group1.median(),
                        "Group 2": group2_name,
                        "n2": len(group2),
                        "Median 2": group2.median(),
                        "U": stat,
                        "p": p,
                    }
                )

    # Convert to a single table
    results_df = pd.DataFrame(results)

    # ── Multiple-testing correction ────────────────────────────────
    # Benjamini-Hochberg false-discovery-rate correction
    results_df["p_adj"] = multipletests(results_df["p"], method="fdr_bh")[1]

    # Optional significance indicator
    results_df["significant"] = results_df["p_adj"] < 0.05
    results_df.to_csv(
        f"results/tables/tableS8_spatio_temporal_spread.csv", sep=";", index=False
    )


# ============================================================
# COORDINATOR
# ============================================================
def spatiotemporal_hgt_analysis(
    summary_df,
    plasmid_df,
    all_pairwise_df,
):
    # ── Run analyses ──────────────────────────────────────────
    per_gene_enrichment(summary_df, plasmid_df)
    spatiotemporal_spread_index(summary_df, plasmid_df)


def _parse_date(series):
    return pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")


def _parse_genes2(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return frozenset()
    return frozenset(g.strip() for g in str(cell).split(",") if g.strip())


def _all_genes(row):
    genes = frozenset()
    for column in ACCESSORY_GENES:
        if column in row.index:
            genes |= _parse_genes2(row[column])
    return genes


def build_cluster_st_timeline(plasmid_df):
    pdf = plasmid_df.copy()
    pdf["_date"] = _parse_date(pdf[DATE_COL])
    pdf["_st"] = pdf[ST_COL].fillna("Unknown").astype(str).str.strip()
    pdf["_cluster"] = pdf[CLUSTER_COL].astype(str).str.strip()
    pdf["_genes"] = pdf.apply(_all_genes, axis=1)

    rows = []
    for (cluster, st), grp in pdf.groupby(["_cluster", "_st"]):
        dated = grp.dropna(subset=["_date"])
        gene_rep = frozenset().union(*grp["_genes"])
        rows.append(
            {
                "cluster": cluster,
                "st": st,
                "n_isolates": len(grp),
                "n_dated": len(dated),
                "first_date": dated["_date"].min() if len(dated) else pd.NaT,
                "last_date": dated["_date"].max() if len(dated) else pd.NaT,
                "gene_repertoire": gene_rep,
                "n_genes": len(gene_rep),
            }
        )

    return pd.DataFrame(rows), pdf


def build_host_transitions(summary_df, plasmid_df_ann, timeline_df):
    pdf = plasmid_df_ann.copy()

    first_date_lookup = (
        timeline_df.dropna(subset=["first_date"])
        .set_index(["cluster", "st"])["first_date"]
        .to_dict()
    )

    cluster_st_plasmids = defaultdict(set)
    for _, row in pdf.iterrows():
        cluster_st_plasmids[(row["_cluster"], row["_st"])].add(row["Plasmid"])

    candidate_bins = summary_df[
        summary_df["has_distant_edge"] | summary_df["has_genogroup_edge"]
    ]

    rows = []

    for _, bin_row in candidate_bins.iterrows():
        members = bin_row["member_plasmids"]
        bin_meta = pdf[pdf["Plasmid"].isin(members)]

        cluster = bin_meta["_cluster"].iloc[0]
        bin_genes = frozenset().union(*bin_meta["_genes"])

        sts_in_bin = (
            bin_meta["_st"]
            .replace({"Unknown": np.nan, "": np.nan})
            .dropna()
            .unique()
            .tolist()
        )

        if len(sts_in_bin) < 2:
            continue

        for st_a, st_b in combinations(sorted(sts_in_bin), 2):
            fd_a = first_date_lookup.get((cluster, st_a))
            fd_b = first_date_lookup.get((cluster, st_b))

            if pd.isna(fd_a) and pd.isna(fd_b):
                donor_st, recipient_st = st_a, st_b
                time_gap = np.nan
            elif pd.isna(fd_a):
                donor_st, recipient_st = st_b, st_a
                time_gap = np.nan
            elif pd.isna(fd_b):
                donor_st, recipient_st = st_a, st_b
                time_gap = np.nan
            elif fd_a <= fd_b:
                donor_st, recipient_st = st_a, st_b
                time_gap = (fd_b - fd_a).days
            else:
                donor_st, recipient_st = st_b, st_a
                time_gap = (fd_a - fd_b).days

            recip_members = bin_meta[bin_meta["_st"] == recipient_st]
            recip_dates = recip_members["_date"].dropna()
            bin_intro_date = recip_dates.min() if len(recip_dates) else pd.NaT

            recip_prior = pdf[
                (pdf["_cluster"] == cluster)
                & (pdf["_st"] == recipient_st)
                & pdf["_date"].notna()
                & (pdf["_date"] < bin_intro_date if pd.notna(bin_intro_date) else False)
            ]
            naive_recipient = len(recip_prior) == 0

            recip_st_prior_any_cluster = pdf[
                (pdf["_st"] == recipient_st)
                & pdf["_date"].notna()
                & (pdf["_date"] < bin_intro_date if pd.notna(bin_intro_date) else False)
            ]
            n_recipient_st_prior_isolates = len(recip_st_prior_any_cluster)
            n_recipient_st_total_isolates = int((pdf["_st"] == recipient_st).sum())

            rows.append(
                {
                    "bin_id": bin_row["bin_id"],
                    "cluster": cluster,
                    "donor_st": donor_st,
                    "recipient_st": recipient_st,
                    "donor_first_date": first_date_lookup.get((cluster, donor_st)),
                    "recipient_first_date": first_date_lookup.get(
                        (cluster, recipient_st)
                    ),
                    "bin_intro_date": bin_intro_date,
                    "time_gap_days": time_gap,
                    "naive_recipient": naive_recipient,
                    "n_recipient_st_prior_isolates": n_recipient_st_prior_isolates,
                    "n_recipient_st_total_isolates": n_recipient_st_total_isolates,
                    "recipient_st_well_sampled": n_recipient_st_prior_isolates >= 10,
                    "recipient_st_poorly_sampled": (
                        n_recipient_st_prior_isolates >= 5
                        and n_recipient_st_prior_isolates < 10
                    ),
                    "recipient_st_rarely_sampled": (
                        n_recipient_st_prior_isolates >= 1
                        and n_recipient_st_prior_isolates < 5
                    ),
                    "recipient_st_un_sampled": n_recipient_st_prior_isolates == 0,
                    "n_donor_isolates": len(
                        cluster_st_plasmids.get((cluster, donor_st), [])
                    ),
                    "n_recipient_isolates": len(
                        cluster_st_plasmids.get((cluster, recipient_st), [])
                    ),
                    "bin_genes": bin_genes,
                    "bin_size": bin_row["bin_size"],
                    "promiscuity_level": bin_row["promiscuity_level"],
                    "edge_level": bin_row["promiscuity_level"],
                    "is_distant_level": bin_row["promiscuity_level"] == "distant",
                    "max_host_dist": bin_row["max_host_dist"],
                    **{
                        f"contains_{g}": bin_row.get(f"contains_{g}", False)
                        for g in ACCESSORY_GENES
                    },
                }
            )

    return pd.DataFrame(rows)


def identify_introductions(transitions_df):
    introductions_df = transitions_df[transitions_df["naive_recipient"]].copy()

    introductions_df["sampling_category"] = pd.cut(
        introductions_df["n_recipient_st_prior_isolates"],
        bins=[-1, 0, 9, np.inf],
        labels=["unsampled", "sparsely_sampled", "well_sampled"],
    )

    return introductions_df


def identify_novel_resistance_virulence_genes(introductions_df, plasmid_df_ann):
    pdf = plasmid_df_ann.copy()
    rows = []

    for _, intro in introductions_df.iterrows():
        if not intro["recipient_st_well_sampled"]:
            continue

        recipient_st = intro["recipient_st"]
        intro_date = intro["bin_intro_date"]
        incoming_genes = intro["bin_genes"]

        prior = pdf[
            (pdf["_st"] == recipient_st)
            & pdf["_date"].notna()
            & (pdf["_date"] < intro_date)
        ]

        prior_genes = frozenset().union(*prior["_genes"])
        novel_genes = sorted(incoming_genes - prior_genes)

        if novel_genes:
            rows.append(
                {
                    "bin_id": intro["bin_id"],
                    "cluster": intro["cluster"],
                    "donor_st": intro["donor_st"],
                    "recipient_st": recipient_st,
                    "bin_intro_date": intro_date,
                    "edge_level": intro["edge_level"],
                    "novel_genes": novel_genes,
                    "n_novel_genes": len(novel_genes),
                }
            )

    return pd.DataFrame(rows)


def assess_establishment(introductions_df, plasmid_df_ann):
    pdf = plasmid_df_ann.copy()
    rows = []

    for _, intro in introductions_df.iterrows():
        cluster = intro["cluster"]
        recipient_st = intro["recipient_st"]
        intro_date = intro["bin_intro_date"]

        if pd.isna(intro_date):
            continue

        postintro = pdf[
            (pdf["_cluster"] == cluster)
            & (pdf["_st"] == recipient_st)
            & pdf["_date"].notna()
            & (pdf["_date"] >= intro_date)
        ].sort_values("_date")

        n_post = len(postintro)

        rows.append(
            {
                "bin_id": intro["bin_id"],
                "cluster": cluster,
                "donor_st": intro["donor_st"],
                "recipient_st": recipient_st,
                "edge_level": intro["edge_level"],
                "bin_intro_date": intro_date,
                "n_postintro_isolates": n_post,
                "established": n_post >= MIN_POSTINTRO_ISOLATES,
            }
        )

    return pd.DataFrame(rows)


def run_introduction_analysis(summary_df, plasmid_df):
    timeline_df, plasmid_df_ann = build_cluster_st_timeline(plasmid_df)

    transitions_df = build_host_transitions(
        summary_df,
        plasmid_df_ann,
        timeline_df,
    )

    introductions_df = identify_introductions(transitions_df)

    novel_gene_df = identify_novel_resistance_virulence_genes(
        introductions_df,
        plasmid_df_ann,
    )

    establishment_df = assess_establishment(
        introductions_df,
        plasmid_df_ann,
    )

    transition_distant = int((transitions_df["edge_level"] == "distant").sum())
    transition_genogroup = int((transitions_df["edge_level"] == "genogroup").sum())

    introduction_distant = int((introductions_df["edge_level"] == "distant").sum())
    introduction_genogroup = int((introductions_df["edge_level"] == "genogroup").sum())

    well_sampled = int((introductions_df["sampling_category"] == "well_sampled").sum())
    sparsely_sampled = int(
        (introductions_df["sampling_category"] == "sparsely_sampled").sum()
    )
    unsampled = int((introductions_df["sampling_category"] == "unsampled").sum())

    established = int(establishment_df["established"].sum())
    established_distant = int(
        (
            establishment_df["established"]
            & (establishment_df["edge_level"] == "distant")
        ).sum()
    )
    established_genogroup = int(
        (
            establishment_df["established"]
            & (establishment_df["edge_level"] == "genogroup")
        ).sum()
    )

    novel_gene_events = len(novel_gene_df)
    novel_genes = sorted(
        {gene for genes in novel_gene_df["novel_genes"] for gene in genes}
    )
    gene_counts = Counter(
        gene for genes in novel_gene_df["novel_genes"] for gene in genes
    )

    gene_counts_df = (
        pd.Series(gene_counts, name="count").rename_axis("gene").reset_index()
    )

    gene_counts_df.to_csv(
        "results/tables/tableS9_introduced_gene_list.csv",
        sep=";",
        index=False,
    )

    text_file = "results/bin_dynamics/introduction_analysis.txt"
    with open(text_file, "w") as f:
        f.write("\nHost-ST transitions")
        f.write(f"  Total transitions : {len(transitions_df)}")
        f.write(f"  Distant           : {transition_distant}")
        f.write(f"  Genogroup         : {transition_genogroup}")

        f.write("\nIntroduction events")
        f.write(f"  Total introductions : {len(introductions_df)}")
        f.write(f"  Distant             : {introduction_distant}")
        f.write(f"  Genogroup           : {introduction_genogroup}")
        f.write(f"  Well sampled (≥10)  : {well_sampled}")
        f.write(f"  Sparsely sampled    : {sparsely_sampled}")
        f.write(f"  Unsampled           : {unsampled}")

        f.write("\nPost-introduction establishment")
        f.write(
            f"  Established (≥{MIN_POSTINTRO_ISOLATES} detections) : " f"{established}"
        )
        f.write(f"  Distant established   : {established_distant}")
        f.write(f"  Genogroup established : {established_genogroup}")

        f.write("\nNovel resistance/virulence gene introductions")
        f.write(f"  Introduction events : {novel_gene_events}")
        f.write(f"  Distinct genes      : {len(novel_genes)}")


# ---------------------------------------------------------
# 5.3.2 Detect HGT signals in bins
# ---------------------------------------------------------
def bin_post_hoc(df_in):
    plasmid_df = df_in.copy()

    cluster_pairs = find_cluster_files(f"output/hgt_results/")
    print(f"Found {len(cluster_pairs)} cluster(s) to process.")

    # ── Pass 1: load all pairwise data for cross-cluster mash calibration ──
    print("\nLoading all detail files for mash calibration ...")
    raw_pairwise = []
    for _, detail_path, _ in cluster_pairs:
        try:
            pw = pd.read_csv(detail_path, sep=";")
            if not pw.empty:
                raw_pairwise.append(pw)
        except pd.errors.EmptyDataError:
            pass

    all_raw = (
        pd.concat(raw_pairwise, ignore_index=True) if raw_pairwise else pd.DataFrame()
    )
    if not all_raw.empty and "host_dist_source" not in all_raw.columns:
        all_raw["host_dist_source"] = "wgmlst"

    mash_clonal, mash_genogroup, calibration_n = derive_mash_cutoffs(all_raw)

    # ── Pass 2: per-cluster processing ────────────────────────
    all_bin_summaries = []
    all_plasmid_to_bin = {}
    all_pairwise_dfs = []
    cluster_n_components = {}

    for cluster_id, detail_path, summary_path in cluster_pairs:
        print(f"  Processing cluster {cluster_id} ...", end=" ", flush=True)

        bin_summaries, plasmid_to_bin, pairwise_df, n_components = process_cluster(
            cluster_id,
            detail_path,
            summary_path,
            plasmid_df,
            mash_clonal,
            mash_genogroup,
        )

        cluster_n_components[cluster_id] = n_components
        all_bin_summaries.extend(bin_summaries)
        all_plasmid_to_bin.update(plasmid_to_bin)
        all_pairwise_dfs.append(pairwise_df)

        print(f"→ {n_components} component(s), {len(plasmid_to_bin)} plasmid(s).")

    # [F2] Merge bin_id onto plasmid_df BEFORE any enrichment call
    plasmid_df["bin_id"] = plasmid_df["Plasmid"].map(all_plasmid_to_bin)

    # [F10] Build summary_df once and pass it through
    summary_df = pd.DataFrame(all_bin_summaries)

    all_pairwise_df = (
        pd.concat(all_pairwise_dfs, ignore_index=True)
        if all_pairwise_dfs
        else pd.DataFrame()
    )

    aggregate_and_report(
        summary_df,
        all_plasmid_to_bin,
        plasmid_df,
        cluster_n_components,
        all_pairwise_df,
    )
    spatiotemporal_hgt_analysis(
        summary_df,
        plasmid_df,
        all_pairwise_df[
            [
                "outlier_isolate",
                "neighbour_isolate",
                "mash_host_dist",
                "wgmlst_host_dist",
                "is_bin",
            ]
        ],
        # "output/wgmlst/all_known_distance.tab",
    )

    run_introduction_analysis(summary_df, plasmid_df)


if __name__ == "__main__":
    bin_post_hoc()
