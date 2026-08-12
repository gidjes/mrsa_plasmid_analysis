# ============================================================
# POST-HOC ANALYSIS OF NEAR-IDENTICAL PLASMID BINS — MRSA VERSION
#
# Research context
# ----------------
# MRSA plasmids have historically been considered unimportant /
# non-promiscuous. This analysis challenges that assumption by
# examining whether nearly-identical plasmids (binned by sequence
# similarity) appear in genetically distant hosts — a signal of
# horizontal gene transfer (HGT). AMR, virulence, biocide and metal
# gene carriage add a clinical dimension: do mobile accessory-gene
# plasmids spread more broadly than their dataset prevalence would
# predict by chance?
#
# Primary axes of analysis
# ------------------------
# 1. HOST GENETIC DISTANCE (replaces species/genus/ST taxonomy)
#    wgMLST categories (surveillance-defined):
#      clonal    : wgmlst_host_dist < 15
#      genogroup : 15 <= wgmlst_host_dist < 1000
#      distant   : wgmlst_host_dist >= 1000
#    Mash-based categories are calibrated empirically by finding
#    isolate pairs that appear in BOTH a wgMLST-metric cluster and
#    a mash-metric cluster (cross-cluster calibration using shared
#    isolate IDs from 'outlier_isolate' / 'neighbour_isolate').
#    Each bin is summarised by its WORST-CASE category
#    (distant > genogroup > clonal) and has independent edge counts
#    for each category.
#
# 2. COMPARTMENT (origin) — retained: which compartments co-appear
#    most in bins, and are any over-represented in inter-compartment
#    bins vs. their baseline dataset prevalence?
#
# 3. ACCESSORY GENE CONTENT — AMR, virulence, biocide, metal:
#    are plasmids carrying each gene type over-represented among
#    promiscuous (binned) plasmids relative to dataset prevalence?
#
# Fixes carried forward from the enterobacteriaceae version
# ---------------------------------------------------------
# F1  Null permutation pre-initialised to zero arrays.
# F2  bin_id merged onto plasmid_df before enrichment.
# F3  Edge deduplication before distance accumulation.
# F4  FDR merge by stable _row_id.
# F5  cluster_propensity agg with explicit if/else.
# F6  Standard_Cluster_mrsa cast to str before filtering.
# F8  bin graph deduplication.
# F10 summary_df built once and passed everywhere.
# ============================================================

import glob
import os
import re
from itertools import combinations
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from helper_functions import clean_plasmid_df
from introduction_analysis import (
    run_introduction_analysis,
    run_postintroduction_dynamics,
)

# ============================================================
# CONSTANTS
# ============================================================

WGMLST_CLONAL = 15
WGMLST_GENOGROUP = 1000

CAT_CLONAL = "clonal"
CAT_GENOGROUP = "genogroup"
CAT_DISTANT = "distant"
CAT_UNKNOWN = "unknown"

# Worst-case ordering for bin-level promiscuity level
CAT_RANK = {CAT_CLONAL: 0, CAT_GENOGROUP: 1, CAT_DISTANT: 2, CAT_UNKNOWN: -1}

ACCESSORY_GENES = ["amr", "virulence", "biocide", "metal"]


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
    os.makedirs("output/hgt_summaries", exist_ok=True)
    cal.to_csv(
        f"output/hgt_summaries/mash_calibration_pairs.csv",
        sep=";",
        index=False,
    )

    return mash_clonal, mash_genogroup, calibration_n


# ============================================================
# GENERAL HELPERS
# ============================================================


def gene_annotation(df):
    """Add has_* boolean and n_*_genes count columns for each accessory gene type."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    for gene in ACCESSORY_GENES:
        if gene not in df.columns:
            df[gene] = np.nan
        df[f"has_{gene}"] = df[gene].fillna("").astype(str).str.strip() != ""
        df[f"n_{gene}_genes"] = (
            df[gene].fillna("").apply(lambda x: 0 if x == "" else len(x.split(",")))
        )
    # Backward-compatible alias used elsewhere
    df["AMR_plasmid"] = df["amr"].apply(lambda x: 0 if pd.isna(x) else 1)
    return df


def load_plasmid_df(plasmid_df):
    """Load and preprocess the plasmid metadata table once."""
    plasmid_df.fillna(
        {"ISOLATE_TL_MLST_ST": "Unknown", "origin": "Unknown"}, inplace=True
    )
    # [F6] Cast Standard_Cluster_mrsa to str before string comparisons
    plasmid_df["Standard_Cluster_mrsa"] = (
        plasmid_df["Standard_Cluster_mrsa"].astype(str).str.strip()
    )
    plasmid_df = plasmid_df.drop_duplicates()
    plasmid_df = gene_annotation(plasmid_df)
    plasmid_df.columns = plasmid_df.columns.str.strip()
    plasmid_df = plasmid_df.loc[:, ~plasmid_df.columns.duplicated()]
    return plasmid_df


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
    for _, row in pairwise_df.iterrows():
        G.add_edge(row["outlier_plasmid"], row["neighbour_plasmid"])

    components = list(nx.connected_components(G))
    n_components = len(components)

    plasmid_to_bin = {
        plasmid: f"{cluster_id}_{i + 1}"
        for i, component in enumerate(components)
        for plasmid in component
    }
    pairwise_df["bin_id"] = pairwise_df["outlier_plasmid"].map(plasmid_to_bin)

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
        for gene in ACCESSORY_GENES:
            col = f"has_{gene}"
            n_gene = int(meta[col].sum()) if col in meta.columns else 0
            gene_counts[f"n_{gene}_plasmids"] = n_gene
            gene_counts[f"contains_{gene}"] = n_gene > 0

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
    report_df.to_csv(
        f"output/hgt_summaries/cluster_level_report.csv", sep=";", index=False
    )

    n_total = len(report_df)
    n_with = int(report_df["has_bins"].sum())
    print("\n====================")
    print("CLUSTER-LEVEL OVERVIEW")
    print("====================")
    print(f"Total clusters processed   : {n_total}")
    print(f"  With bins                : {n_with}")
    print(f"  Without bins             : {n_total - n_with}")
    if n_with > 0:
        w = report_df[report_df["has_bins"]]
        print(f"\nAmong clusters WITH bins:")
        print(
            f"  Median bins/cluster      : {w['n_bins'].median():.1f}"
            f" (range {int(w['n_bins'].min())}–{int(w['n_bins'].max())})"
        )
        print(f"  Clusters with ≥1 distant bin    : {(w['n_bins_distant'] > 0).sum()}")
        print(
            f"  Clusters with ≥1 genogroup bin  : {(w['n_bins_genogroup'] > 0).sum()}"
        )
        print(
            f"  Clusters with ≥1 inter-comp bin : {(w['n_bins_inter_compartment'] > 0).sum()}"
        )
    print("\n→ cluster_level_report.csv")
    return report_df


# ============================================================
# ACCESSORY GENE ENRICHMENT
# ============================================================


def accessory_gene_enrichment(summary_df, plasmid_df):
    """
    Are plasmids carrying each accessory gene type (AMR, virulence,
    biocide, metal) over-represented among promiscuous (binned)
    plasmids relative to their prevalence in the full dataset?

    For each gene type:
      - Raw counts and % in full dataset vs. bins (descriptive)
      - Fisher's exact test A: 2x2
            rows = has gene / does not have gene
            cols = in a bin / not in a bin
        Tests whether gene-carrying plasmids are MORE likely to be
        promiscuous than non-carrying plasmids.
      - Fisher's exact test B: among binned plasmids only
            rows = has gene / does not have gene
            cols = bin has distant edge / bin does not
        Tests whether gene-carrying plasmids are disproportionately
        found in the most promiscuous (distant-host) bins.

    FDR correction is applied jointly across all tests.

    Exports
    -------
    output/hgt_summaries/accessory_gene_enrichment.csv
    """
    print("\n====================")
    print("ACCESSORY GENE ENRICHMENT AMONG PROMISCUOUS PLASMIDS")
    print("====================")

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
    for gene in ACCESSORY_GENES:
        col = f"has_{gene}"
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

        print(f"\n  {gene.upper()}")
        print(f"    Dataset : {n_gene_total}/{n_total} ({pct_gene_dataset:.1f}%)")
        print(f"    In bins : {n_gene_in_bin}/{n_in_bin} ({pct_gene_bins:.1f}%)")
        print(
            f"    Fisher (in bin vs. not) : p={p_bin:.4f}"
            if not np.isnan(p_bin)
            else "    Fisher (in bin): insufficient data"
        )
        print(
            f"    Fisher (distant bin vs. other): p={p_distant:.4f}"
            if not np.isnan(p_distant)
            else "    Fisher (distant): insufficient data"
        )

        rows.append(
            {
                "gene_type": gene,
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

    if not enrich_df.empty:
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
        f"output/hgt_summaries/accessory_gene_enrichment.csv",
        sep=";",
        index=False,
    )
    print("\n→ accessory_gene_enrichment.csv")
    return enrich_df


# ============================================================
# COMPARTMENT ANALYSIS
# ============================================================


def compartment_analysis(summary_df, plasmid_df, n_perm=2000, seed=42):
    """
    Two complementary compartment analyses:

    1. PAIRWISE CO-OCCURRENCE TABLE
       For each compartment pair (A, B): how many bins contain
       plasmids from both?  Shows which boundaries are crossed most.

    2. PERMUTATION TEST PER COMPARTMENT PAIR
       Shuffle compartment labels across plasmids (bin membership
       fixed) and recount co-occurrences per pair.  Tests whether
       each specific pair co-occurs more than expected by chance.
       [F1] Null arrays pre-initialised to zero.

    3. FISHER TEST PER COMPARTMENT LABEL
       Is compartment X over-represented among plasmids in inter-
       compartment bins vs. its baseline prevalence in the dataset?

    Exports
    -------
    output/hgt_summaries/compartment_pair_cooccurrence.csv
    output/hgt_summaries/compartment_pair_permutation.csv
    output/hgt_summaries/compartment_fisher.csv
    """
    print("\n====================")
    print("COMPARTMENT ANALYSIS")
    print("====================")

    rng = np.random.default_rng(seed)
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

    print("\nCompartment pair co-occurrences (observed bins):")
    print(pair_df.to_string(index=False))

    # ── 2. Permutation test per compartment pair ───────────────
    print("\n--- Permutation test per compartment pair ---")

    all_pairs = list(pair_counts.keys())
    plasmid_ids = np.array([pid for members in bin_members.values() for pid in members])
    comp_arr = np.array([comp_lookup.get(p, "Unknown") for p in plasmid_ids])

    # [F1] Pre-initialise null to zero arrays
    pair_null = {pair: np.zeros(n_perm, dtype=int) for pair in all_pairs}

    for perm_idx in range(n_perm):
        perm_comp = dict(zip(plasmid_ids, rng.permutation(comp_arr)))
        perm_pair_counts = Counter()
        for members in bin_members.values():
            comps = {
                perm_comp.get(p, "Unknown")
                for p in members
                if perm_comp.get(p, "Unknown") not in ("Unknown", "", None)
            }
            for pair in combinations(sorted(comps), 2):
                perm_pair_counts[pair] += 1
        for pair in all_pairs:
            pair_null[pair][perm_idx] = perm_pair_counts.get(pair, 0)

    perm_rows = []
    for pair in all_pairs:
        obs = pair_counts[pair]
        null = pair_null[pair]
        sd = null.std()
        perm_rows.append(
            {
                "compartment_A": pair[0],
                "compartment_B": pair[1],
                "n_bins_co_occurring": obs,
                "null_mean": round(null.mean(), 2),
                "null_sd": round(sd, 3),
                "zscore": round((obs - null.mean()) / sd, 3) if sd > 0 else np.nan,
                "empirical_p": empirical_pvalue(obs, null, alternative="greater"),
            }
        )

    perm_df = pd.DataFrame(perm_rows)
    if not perm_df.empty:
        perm_df["padj"] = multipletests(perm_df["empirical_p"], method="fdr_bh")[1]
        perm_df = perm_df.sort_values("padj")

    sig = perm_df[perm_df["padj"] < 0.05] if not perm_df.empty else pd.DataFrame()
    print(f"  Compartment pairs tested         : {len(perm_df)}")
    print(f"  Significant pairs (padj < 0.05) : {len(sig)}")
    if not sig.empty:
        print(sig.to_string(index=False))

    # ── 3. Fisher per compartment label ───────────────────────
    print("\n--- Fisher: compartment over-representation in inter-compartment bins ---")

    if "bin_id" not in plasmid_df.columns:
        raise ValueError("plasmid_df missing 'bin_id'. [F2] merge must happen first.")

    binned = plasmid_df.dropna(subset=["bin_id"]).copy()
    bin_flags = summary_df.set_index("bin_id")[["inter_compartment"]]
    binned = binned.join(bin_flags, on="bin_id", how="left")

    fisher_rows = []
    for label in binned["origin"].dropna().unique():
        lbl = str(label).strip()
        if lbl in ("Unknown", ""):
            continue
        has_label = binned["origin"] == label
        in_target = binned["inter_compartment"].fillna(False)

        a = int((has_label & in_target).sum())
        b = int((~has_label & in_target).sum())
        c = int((has_label & ~in_target).sum())
        d = int((~has_label & ~in_target).sum())

        if a + c == 0:
            continue
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        fisher_rows.append(
            {
                "compartment": label,
                "n_plasmids_in_dataset": a + c,
                "n_plasmids_inter_comp_bin": a,
                "pct_in_inter_comp_bin": (
                    round(a / (a + c) * 100, 2) if (a + c) > 0 else np.nan
                ),
                "fisher_p": p,
            }
        )

    fisher_df = pd.DataFrame(fisher_rows)
    if not fisher_df.empty:
        fisher_df["padj"] = multipletests(fisher_df["fisher_p"], method="fdr_bh")[1]
        fisher_df = fisher_df.sort_values("padj")

    print(f"  Fisher tests (compartments): {len(fisher_df)}")
    sig_f = (
        fisher_df[fisher_df["padj"] < 0.05] if not fisher_df.empty else pd.DataFrame()
    )
    if not sig_f.empty:
        print(sig_f.to_string(index=False))

    # ── Exports ───────────────────────────────────────────────
    pair_df.to_csv(
        f"output/hgt_summaries/compartment_pair_cooccurrence.csv",
        sep=";",
        index=False,
    )
    perm_df.to_csv(
        f"output/hgt_summaries/compartment_pair_permutation.csv",
        sep=";",
        index=False,
    )
    fisher_df.to_csv(
        f"output/hgt_summaries/compartment_fisher.csv", sep=";", index=False
    )
    print(
        "\n→ compartment_pair_cooccurrence.csv, "
        "compartment_pair_permutation.csv, compartment_fisher.csv"
    )
    return pair_df, perm_df, fisher_df


# ============================================================
# HOST DISTANCE ANALYSIS
# ============================================================


def host_distance_analysis(summary_df, all_pairwise_df):
    """
    Quantitative analysis of host genetic distance within bins.

    Reports
    -------
    - Edge-level distribution by distance category and metric source
    - Bin promiscuity level distribution
    - AMR / virulence / biocide vs. non-carrying bin host distances
      (Mann-Whitney)
    - Inter-compartment vs. intra-compartment host distances
    - Plasmid Mash similarity vs. host distance (Spearman; low/negative
      rho = HGT signal: similar plasmids in genetically distant hosts)
    - Stratified distributions for wgMLST vs. mash-calibrated edges

    Exports
    -------
    output/hgt_summaries/host_distance_analysis.csv
    output/hgt_summaries/edge_distance_summary.csv
    """
    print("\n====================")
    print("HOST DISTANCE ANALYSIS")
    print("====================")

    df = summary_df.copy()
    pw = all_pairwise_df.copy()

    # ── Edge-level distribution ────────────────────────────────
    print("\nEdge-level host distance category distribution:")
    n_total = len(pw)
    for cat in [CAT_CLONAL, CAT_GENOGROUP, CAT_DISTANT, CAT_UNKNOWN]:
        n = int((pw["dist_category"] == cat).sum())
        print(f"  {cat:12s}: {n:6d}  ({n / n_total * 100:.1f}%)")

    n_mash = int(pw["is_mash"].sum()) if "is_mash" in pw.columns else 0
    print(f"\n  wgMLST-metric edges : {n_total - n_mash}")
    print(f"  Mash-metric edges   : {n_mash}")

    # ── Bin promiscuity level distribution ────────────────────
    print("\nBin promiscuity level distribution:")
    for lvl in [CAT_CLONAL, CAT_GENOGROUP, CAT_DISTANT, CAT_UNKNOWN]:
        n = int((df["promiscuity_level"] == lvl).sum())
        print(f"  {lvl:12s}: {n} bins  ({n / len(df) * 100:.1f}%)")

    # ── Accessory gene bins vs. non-carrying ──────────────────
    for gene in ACCESSORY_GENES:
        col = f"contains_{gene}"
        if col not in df.columns:
            continue
        pos = df.loc[df[col], "max_host_dist"].dropna()
        neg = df.loc[~df[col], "max_host_dist"].dropna()
        if len(pos) == 0 and len(neg) == 0:
            continue
        print(f"\nMax host dist — {gene} vs. non-{gene} bins:")
        print(f"  {gene} bins     (n={len(pos)}): median={pos.median():.1f}")
        print(f"  non-{gene} bins (n={len(neg)}): median={neg.median():.1f}")
        if len(pos) > 0 and len(neg) > 0:
            stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
            print(f"  Mann-Whitney U={stat:.0f}, p={p:.4f}")

    # ── Inter-compartment vs. intra-compartment ───────────────
    inter = df.loc[df["inter_compartment"], "max_host_dist"].dropna()
    intra = df.loc[~df["inter_compartment"], "max_host_dist"].dropna()
    print(f"\nMax host dist — inter- vs. intra-compartment bins:")
    print(f"  Inter (n={len(inter)}): median={inter.median():.1f}")
    print(f"  Intra (n={len(intra)}): median={intra.median():.1f}")
    if len(inter) > 0 and len(intra) > 0:
        stat, p = mannwhitneyu(inter, intra, alternative="two-sided")
        print(f"  Mann-Whitney U={stat:.0f}, p={p:.4f}")

    # ── Plasmid Mash vs. host distance (HGT signal) ───────────
    if "mash_plasmid_dist" in pw.columns:
        valid = pw[["mash_plasmid_dist", "wgmlst_host_dist"]].dropna()
        if len(valid) > 2:
            r, p = spearmanr(valid["mash_plasmid_dist"], valid["wgmlst_host_dist"])
            print(f"\nSpearman: plasmid Mash similarity ~ host distance")
            print(f"  rho={r:.4f}, p={p:.4f}  (n={len(valid)} edges)")
            print(
                f"  [Low/negative rho = similar plasmids in distant hosts = HGT signal]"
            )

    # ── Stratified by metric ──────────────────────────────────
    if "is_mash" in pw.columns:
        for label, mask in [
            ("wgMLST edges", ~pw["is_mash"]),
            ("mash-calibrated edges", pw["is_mash"]),
        ]:
            sub = pw[mask]
            if sub.empty:
                continue
            sub_cats = sub["dist_category"].value_counts()
            print(f"\nCategory distribution ({label}, n={len(sub)}):")
            for cat in [CAT_CLONAL, CAT_GENOGROUP, CAT_DISTANT, CAT_UNKNOWN]:
                print(f"  {cat:12s}: {int(sub_cats.get(cat, 0))}")

    # ── Exports ───────────────────────────────────────────────
    dist_export_cols = [
        "bin_id",
        "bin_size",
        "n_unique_isolates",
        "inter_compartment",
        "n_clonal_edges",
        "n_genogroup_edges",
        "n_distant_edges",
        "n_unknown_dist_edges",
        "n_mash_dist_edges",
        "has_genogroup_edge",
        "has_distant_edge",
        "promiscuity_level",
        "max_host_dist",
        "median_host_dist",
        "median_mash_plasmid_dist",
        "graph_density",
    ] + [f"contains_{g}" for g in ACCESSORY_GENES if f"contains_{g}" in df.columns]

    df[[c for c in dist_export_cols if c in df.columns]].to_csv(
        f"output/hgt_summaries/host_distance_analysis.csv",
        sep=";",
        index=False,
    )

    if "wgmlst_host_dist" in pw.columns:
        edge_summary = (
            pw.groupby("bin_id")["wgmlst_host_dist"]
            .agg(["median", "mean", "max", "count"])
            .reset_index()
        )
        edge_summary.columns = [
            "bin_id",
            "median_host_dist",
            "mean_host_dist",
            "max_host_dist",
            "n_edges",
        ]
        edge_summary.to_csv(
            f"output/hgt_summaries/edge_distance_summary.csv",
            sep=";",
            index=False,
        )

    print("\n→ host_distance_analysis.csv, edge_distance_summary.csv")


# ============================================================
# GLOBAL PERMUTATION TEST
# ============================================================


def global_permutation_test(summary_df, plasmid_df, n_perm=2000, seed=42):
    """
    Dataset-level test of HGT signal: are the observed numbers of
    genogroup-edge, distant-edge, and inter-compartment bins higher
    than expected when distance category labels are shuffled across
    plasmids while preserving bin membership?

    Exports
    -------
    output/hgt_summaries/global_permutation_test.csv
    """
    print("\n====================")
    print("GLOBAL PERMUTATION TEST")
    print("====================")

    rng = np.random.default_rng(seed)

    bin_members = {
        row["bin_id"]: row["member_plasmids"] for _, row in summary_df.iterrows()
    }

    comp_lookup = plasmid_df.set_index("Plasmid")["origin"].to_dict()

    # Per-plasmid distance category from its bin's promiscuity level
    plasmid_to_level = {
        pid: row["promiscuity_level"]
        for _, row in summary_df.iterrows()
        for pid in row["member_plasmids"]
    }

    plasmid_ids = np.array([pid for members in bin_members.values() for pid in members])
    dist_cat_arr = np.array([plasmid_to_level.get(p, CAT_UNKNOWN) for p in plasmid_ids])
    comp_arr = np.array([comp_lookup.get(p, "Unknown") for p in plasmid_ids])

    obs_genogroup = int(summary_df["has_genogroup_edge"].sum())
    obs_distant = int(summary_df["has_distant_edge"].sum())
    obs_intercomp = int(summary_df["inter_compartment"].sum())

    # [F1] Pre-initialise null arrays
    null_genogroup = np.zeros(n_perm, dtype=int)
    null_distant = np.zeros(n_perm, dtype=int)
    null_intercomp = np.zeros(n_perm, dtype=int)

    for perm_idx in range(n_perm):
        perm_dist = dict(zip(plasmid_ids, rng.permutation(dist_cat_arr)))
        perm_comp = dict(zip(plasmid_ids, rng.permutation(comp_arr)))

        n_geo = n_dist = n_comp = 0
        for members in bin_members.values():
            cats = {perm_dist.get(p, CAT_UNKNOWN) for p in members}
            comps = {perm_comp.get(p, "Unknown") for p in members} - {
                None,
                "Unknown",
                "",
            }
            if CAT_GENOGROUP in cats:
                n_geo += 1
            if CAT_DISTANT in cats:
                n_dist += 1
            if len(comps) > 1:
                n_comp += 1

        null_genogroup[perm_idx] = n_geo
        null_distant[perm_idx] = n_dist
        null_intercomp[perm_idx] = n_comp

    p_geo = empirical_pvalue(obs_genogroup, null_genogroup, alternative="greater")
    p_dist = empirical_pvalue(obs_distant, null_distant, alternative="greater")
    p_comp = empirical_pvalue(obs_intercomp, null_intercomp, alternative="greater")

    for label, obs, null, p in [
        ("bins with genogroup edge", obs_genogroup, null_genogroup, p_geo),
        ("bins with distant edge", obs_distant, null_distant, p_dist),
        ("inter-compartment bins", obs_intercomp, null_intercomp, p_comp),
    ]:
        print(f"\n  Observed {label}: {obs}")
        print(f"  Null mean ± SD : {null.mean():.1f} ± {null.std():.1f}")
        print(f"  Empirical p    : {p:.4f}")

    result_df = pd.DataFrame(
        [
            {
                "outcome": "bins_with_genogroup_edge",
                "observed": obs_genogroup,
                "null_mean": null_genogroup.mean(),
                "null_sd": null_genogroup.std(),
                "empirical_p": p_geo,
            },
            {
                "outcome": "bins_with_distant_edge",
                "observed": obs_distant,
                "null_mean": null_distant.mean(),
                "null_sd": null_distant.std(),
                "empirical_p": p_dist,
            },
            {
                "outcome": "inter_compartment_bins",
                "observed": obs_intercomp,
                "null_mean": null_intercomp.mean(),
                "null_sd": null_intercomp.std(),
                "empirical_p": p_comp,
            },
        ]
    )
    result_df.to_csv(
        f"output/hgt_summaries/global_permutation_test.csv",
        sep=";",
        index=False,
    )
    print("\n→ global_permutation_test.csv")
    return result_df


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
    """
    Coordinate all sub-analyses and write master summary.

    [F2]  plasmid_df already has bin_id merged before this call.
    [F10] summary_df built once upstream and passed through.
    """
    cluster_level_report(summary_df, cluster_n_components)

    bin_plasmid_ids = set(all_plasmid_to_bin.keys())
    bin_meta = plasmid_df[plasmid_df["Plasmid"].isin(bin_plasmid_ids)].copy()

    n_bins = len(summary_df)
    n_plasmids = len(bin_plasmid_ids)
    n_isolates = int(len(bin_meta["Parent"].unique()))
    med_size = summary_df["bin_size"].median()
    q1 = summary_df["bin_size"].quantile(0.25)
    q3 = summary_df["bin_size"].quantile(0.75)

    print("\n====================")
    print("BASIC COUNTS")
    print("====================")
    print(f"Plasmids in bins             : {n_plasmids}")
    print(f"Total bins                   : {n_bins}")
    print(f"Unique isolates spanned      : {n_isolates}")
    print(f"Median bin size              : {med_size:.1f}")
    print(f"IQR bin size                 : {q3 - q1:.1f}  (Q1={q1:.1f}, Q3={q3:.1f})")
    print(
        f"Median unique isolates / bin : {summary_df['n_unique_isolates'].median():.1f}"
    )

    # ── Compartment distribution ──────────────────────────────
    total_comp = plasmid_df["origin"].value_counts().rename("n_total_dataset")
    bin_comp = bin_meta["origin"].value_counts().rename("n_bin_plasmids")
    comp_counts = pd.concat([bin_comp, total_comp], axis=1).fillna(0).reset_index()
    comp_counts.columns = ["compartment", "n_bin_plasmids", "n_total_dataset"]
    comp_counts["pct_of_compartment_in_bins"] = (
        comp_counts["n_bin_plasmids"] / comp_counts["n_total_dataset"] * 100
    ).round(2)

    print("\n====================")
    print("COMPARTMENT DISTRIBUTION")
    print("====================")
    print(comp_counts.to_string(index=False))

    # ── Accessory gene content (descriptive) ──────────────────
    print("\n====================")
    print("ACCESSORY GENE CONTENT")
    print("====================")
    n_total = len(plasmid_df)
    n_in_bin = len(bin_meta)
    for gene in ACCESSORY_GENES:
        col = f"has_{gene}"
        if col not in plasmid_df.columns:
            continue
        n_gene_total = int(plasmid_df[col].sum())
        n_gene_bin = int(bin_meta[col].sum())
        pct_dataset = round(n_gene_total / n_total * 100, 1) if n_total else np.nan
        pct_bins = round(n_gene_bin / n_in_bin * 100, 1) if n_in_bin else np.nan
        print(
            f"  {gene.upper():10s}: "
            f"dataset {n_gene_total}/{n_total} ({pct_dataset:.1f}%)  |  "
            f"in bins {n_gene_bin}/{n_in_bin} ({pct_bins:.1f}%)"
        )

    # ── Promiscuity overview ───────────────────────────────────
    print("\n====================")
    print("BIN PROMISCUITY OVERVIEW")
    print("====================")
    for lvl in [CAT_CLONAL, CAT_GENOGROUP, CAT_DISTANT]:
        n = int((summary_df["promiscuity_level"] == lvl).sum())
        print(f"  Bins at {lvl:12s} level : {n} ({n / n_bins * 100:.1f}%)")
    n_intercomp = int(summary_df["inter_compartment"].sum())
    print(
        f"  Inter-compartment bins         : {n_intercomp} ({n_intercomp / n_bins * 100:.1f}%)"
    )

    # ── Sub-analyses ──────────────────────────────────────────
    accessory_gene_enrichment(summary_df, plasmid_df)
    compartment_analysis(summary_df, plasmid_df)
    host_distance_analysis(summary_df, all_pairwise_df)
    global_permutation_test(summary_df, plasmid_df)

    # ── Master summary ─────────────────────────────────────────
    def _n(col):
        return int(summary_df[col].sum()) if col in summary_df.columns else np.nan

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
            "value": [
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
        }
    )

    print("\n====================")
    print("MASTER SUMMARY")
    print("====================")
    print(posthoc_summary.to_string(index=False))

    # ── Exports ───────────────────────────────────────────────
    summary_df.to_csv(
        f"output/hgt_summaries/plasmid_bin_summary.csv", sep=";", index=False
    )
    comp_counts.to_csv(
        f"output/hgt_summaries/compartment_distribution.csv",
        sep=";",
        index=False,
    )
    posthoc_summary.to_csv(
        "output/hgt_summaries/posthoc_summary.csv", sep=";", index=False
    )
    print("\nAll outputs written to output/hgt_summaries/")


# ============================================================
# SPATIOTEMPORAL HGT ANALYSIS — MRSA PLASMID POST-HOC
# ============================================================
#
# Adds four analysis layers to the existing post-hoc pipeline:
#
# 1. PER-GENE ENRICHMENT
#    Parse comma-separated gene columns into individual gene names.
#    For each gene: Fisher exact (distant-bin prevalence vs. dataset).
#    Gene-pair co-occurrence table for distant bins.
#
# 2. SPATIOTEMPORAL SPREAD INDEX
#    Per bin: date range (days) and municipality span (n distinct).
#    Mann-Whitney: do gene-carrying bins spread wider in space/time?
#
# 3. BACKGROUND-CORRECTED SPREAD
#    For each bin, identify background isolates co-circulating in
#    the same municipalities and time window (±BACKGROUND_WINDOW_DAYS)
#    but NOT in the bin. Compare host genetic diversity (median
#    pairwise distance) of bin members vs. background.
#    Signal: bin members are more genetically diverse than background
#    isolates sharing the same space-time window.
#
# 4. TEMPORAL DIRECTIONALITY
#    For bins with ≥ MIN_DATED_MEMBERS dated members across ≥ 2 host
#    genetic backgrounds: Spearman rho between sample date rank and
#    host genetic distance from the earliest member.
#    Positive rho = plasmid acquired by progressively more distant
#    hosts over time (HGT directionality signal).
#
# Integration
# -----------
# Call spatiotemporal_hgt_analysis() from aggregate_and_report()
# after the existing sub-analyses. It expects:
#   - summary_df        : bin-level summary (from process_cluster)
#   - plasmid_df        : metadata with bin_id already merged [F2]
#   - all_pairwise_df   : concatenated edge-level pairwise data
#   - mash_dist_path    : path to all-vs-all mash pairwise TSV
#   - wgmlst_dist_path  : path to all-vs-all wgMLST square matrix
#
# Expected plasmid_df columns used here:
#   Plasmid, origin, bin_id, MATERIAL_SAMPLINGDATE (dd-mm-yyyy),
#   MATERIAL_SUBMITTER_MUNICIPALITY, amr, virulence, biocide, metal
#   (isolate ID assumed to be accessible via outlier_isolate /
#    neighbour_isolate in all_pairwise_df, or via a column
#    ISOLATE_ID in plasmid_df — see ISOLATE_ID_COL constant below)
#
# ============================================================

import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

# ── Constants ────────────────────────────────────────────────
ACCESSORY_GENES = ["amr", "virulence", "biocide", "metal"]

# Column in plasmid_df that holds the isolate ID used as the key
# in the distance matrices.  Adjust if your column name differs.
ISOLATE_ID_COL = "Parent"

# Time window (days) on each side of a bin's sampling range used
# to define "co-circulating background" isolates.
BACKGROUND_WINDOW_DAYS = 90

# Minimum dated members with ≥2 distinct host backgrounds required
# to attempt temporal directionality analysis for a bin.
MIN_DATED_MEMBERS = 3

CAT_DISTANT = "distant"
CAT_GENOGROUP = "genogroup"
CAT_CLONAL = "clonal"


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


def _load_mash(path):
    """
    Load mash all-vs-all pairwise TSV.
    Expected columns: subject, target, distance  (tab-separated).
    Returns long DataFrame with columns: isolate_a, isolate_b, mash_dist.
    """
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2])
    df.columns = ["isolate_a", "isolate_b", "mash_dist"]
    df["mash_dist"] = pd.to_numeric(df["mash_dist"], errors="coerce")
    return df


def _load_wgmlst(path):
    """
    Load wgMLST square distance matrix (isolates as row and column index).
    Melts to long format: isolate_a, isolate_b, wgmlst_dist.
    Missing / non-aureus isolates will have NaN distances — handled downstream.
    """
    mat = pd.read_csv(path, sep="\t", index_col=0)
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    long = mat.stack().reset_index()
    long.columns = ["isolate_a", "isolate_b", "wgmlst_dist"]
    long["wgmlst_dist"] = pd.to_numeric(long["wgmlst_dist"], errors="coerce")
    # Drop self-pairs
    long = long[long["isolate_a"] != long["isolate_b"]]
    return long


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
        _canonical_pair(r.isolate_a, r.isolate_b): r.mash_dist
        for r in mash_df.itertuples()
        if pd.notna(r.mash_dist)
    }
    wgmlst_lookup = {
        _canonical_pair(r.isolate_a, r.isolate_b): r.wgmlst_dist
        for r in wgmlst_df.itertuples()
        if pd.notna(r.wgmlst_dist)
    }
    return mash_lookup, wgmlst_lookup


# ============================================================
# 1. PER-GENE ENRICHMENT
# ============================================================


def per_gene_enrichment(summary_df, plasmid_df):
    """
    For each individual gene name found in any accessory gene column:
      - Count prevalence in full dataset vs. in promiscuous bins
        (bins with ≥1 genogroup or distant edge).
      - Fisher exact test (one-sided: over-represented in promiscuous bins).
      - FDR correction across all genes and all gene types jointly.

    Also produces a gene-pair co-occurrence table restricted to
    distant-host bins: which gene combinations travel together.

    Exports
    -------
    per_gene_enrichment.csv
    gene_pair_cooccurrence_distant_bins.csv
    """
    print("\n====================")
    print("PER-GENE ENRICHMENT (individual gene resolution)")
    print("====================")

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

    for gene_type in ACCESSORY_GENES:
        if gene_type not in pdf.columns:
            continue

        pdf[f"_genes_{gene_type}"] = _parse_genes(pdf[gene_type])

        # All distinct gene names in this column
        all_genes = [g for genes in pdf[f"_genes_{gene_type}"] for g in genes]
        gene_counts_total = Counter(all_genes)

        for gene_name, n_gene_total in gene_counts_total.items():
            has_gene = pdf[f"_genes_{gene_type}"].apply(lambda gs: gene_name in gs)

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
                    "gene_type": gene_type,
                    "gene_name": gene_name,
                    "n_in_dataset": n_gene_total,
                    "pct_in_dataset": pct_dataset,
                    "n_in_promiscuous_bins": n_gene_promiscuous,
                    "pct_in_promiscuous_bins": pct_promiscuous,
                    "fisher_p": p,
                }
            )

    enrich_df = pd.DataFrame(rows)

    if not enrich_df.empty:
        valid_mask = enrich_df["fisher_p"].notna()
        padj = np.full(len(enrich_df), np.nan)
        if valid_mask.sum() > 0:
            padj[valid_mask] = multipletests(
                enrich_df.loc[valid_mask, "fisher_p"], method="fdr_bh"
            )[1]
        enrich_df["padj"] = padj
        enrich_df = enrich_df.sort_values("padj")

        sig = enrich_df[enrich_df["padj"] < 0.05]
        print(f"  Genes tested            : {len(enrich_df)}")
        print(f"  Significant (padj<0.05) : {len(sig)}")
        if not sig.empty:
            print(
                sig[
                    [
                        "gene_type",
                        "gene_name",
                        "n_in_dataset",
                        "pct_in_dataset",
                        "pct_in_promiscuous_bins",
                        "padj",
                    ]
                ]
                .head(20)
                .to_string(index=False)
            )

    # ── Gene-pair co-occurrence in distant bins ───────────────
    print("\n--- Gene-pair co-occurrence in distant-host bins ---")

    distant_meta = pdf[pdf["in_distant_bin"]].copy()

    pair_counter = Counter()
    for _, row in distant_meta.iterrows():
        gene_set = set()
        for gene_type in ACCESSORY_GENES:
            col = f"_genes_{gene_type}"
            if col in distant_meta.columns:
                gene_set.update(row[col])
        for pair in combinations(sorted(gene_set), 2):
            pair_counter[pair] += 1

    pair_df = (
        pd.DataFrame(
            [(a, b, n) for (a, b), n in pair_counter.most_common(50)],
            columns=["gene_a", "gene_b", "n_co_occurrences_distant_bins"],
        )
        if pair_counter
        else pd.DataFrame(columns=["gene_a", "gene_b", "n_co_occurrences_distant_bins"])
    )

    print(f"  Distinct gene pairs in distant bins: {len(pair_df)}")
    if not pair_df.empty:
        print(pair_df.head(15).to_string(index=False))

    # ── Exports ───────────────────────────────────────────────
    out_dir = f"output/hgt_summaries"
    os.makedirs(out_dir, exist_ok=True)
    enrich_df.to_csv(f"{out_dir}/per_gene_enrichment.csv", sep=";", index=False)
    pair_df.to_csv(
        f"{out_dir}/gene_pair_cooccurrence_distant_bins.csv", sep=";", index=False
    )
    print("\n→ per_gene_enrichment.csv, gene_pair_cooccurrence_distant_bins.csv")
    return enrich_df, pair_df


# ============================================================
# 2. SPATIOTEMPORAL SPREAD INDEX
# ============================================================


def spatiotemporal_spread_index(summary_df, plasmid_df):
    """
    Per bin, compute:
      - date_range_days  : max − min sampling date within bin members
      - n_municipalities : number of distinct MATERIAL_SUBMITTER_MUNICIPALITY
      - spread_score     : date_range_days × n_municipalities (composite)

    Then test: do bins carrying each accessory gene type show wider
    spatiotemporal spread? (Mann-Whitney, two-sided)

    Also computes Spearman correlation between spread_score and
    max_host_dist — to ask whether wider-spreading bins are also
    the ones reaching more genetically distant hosts.

    Exports
    -------
    bin_spatiotemporal_spread.csv
    spatiotemporal_gene_association.csv
    """
    print("\n====================")
    print("SPATIOTEMPORAL SPREAD INDEX")
    print("====================")

    pdf = plasmid_df.copy()
    pdf["_date"] = _parse_date(pdf["MATERIAL_SAMPLINGDATE"])

    spread_rows = []
    for _, bin_row in summary_df.iterrows():
        members = bin_row["member_plasmids"]
        meta = pdf[pdf["Plasmid"].isin(members)]

        dates = meta["_date"].dropna()
        munis = meta["MATERIAL_SUBMITTER_MUNICIPALITY"].dropna().astype(str).str.strip()
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
        for gene in ACCESSORY_GENES:
            col = f"contains_{gene}"
            row[col] = bin_row.get(col, False)

        spread_rows.append(row)

    spread_df = pd.DataFrame(spread_rows)

    # ── Gene association with spread ──────────────────────────
    assoc_rows = []
    for gene in ACCESSORY_GENES:
        col = f"contains_{gene}"
        if col not in spread_df.columns:
            continue
        for metric, metric_label in [
            ("date_range_days", "date range (days)"),
            ("n_municipalities", "n municipalities"),
            ("spread_score", "spread score"),
        ]:
            pos = spread_df.loc[spread_df[col], metric].dropna()
            neg = spread_df.loc[~spread_df[col], metric].dropna()
            if len(pos) < 2 or len(neg) < 2:
                continue
            stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
            assoc_rows.append(
                {
                    "gene_type": gene,
                    "metric": metric,
                    "median_gene_positive": round(pos.median(), 2),
                    "median_gene_negative": round(neg.median(), 2),
                    "n_positive": len(pos),
                    "n_negative": len(neg),
                    "mannwhitney_U": round(stat, 1),
                    "p_value": round(p, 5),
                }
            )

    assoc_df = pd.DataFrame(assoc_rows)
    if not assoc_df.empty:
        assoc_df["padj"] = multipletests(assoc_df["p_value"], method="fdr_bh")[1]
        assoc_df = assoc_df.sort_values("padj")
        print("\nGene-type × spatiotemporal spread associations:")
        print(assoc_df.to_string(index=False))

    # ── Spread vs. host distance ──────────────────────────────
    valid = spread_df[["spread_score", "max_host_dist"]].dropna()
    if len(valid) > 2:
        rho, p = spearmanr(valid["spread_score"], valid["max_host_dist"])
        print(
            f"\nSpearman: spread_score ~ max_host_dist  "
            f"rho={rho:.4f}, p={p:.4f}  (n={len(valid)} bins)"
        )
        print(
            "  [Positive rho: wider-spreading bins also reach more "
            "distant hosts — consistent with active HGT-driven dissemination]"
        )

    # ── Distant vs. non-distant bin spread ───────────────────
    for metric in ["date_range_days", "n_municipalities", "spread_score"]:
        d = spread_df.loc[spread_df["has_distant_edge"], metric].dropna()
        nd = spread_df.loc[~spread_df["has_distant_edge"], metric].dropna()
        if len(d) >= 2 and len(nd) >= 2:
            stat, p = mannwhitneyu(d, nd, alternative="two-sided")
            print(
                f"\n{metric}: distant bins (n={len(d)}, median={d.median():.1f}) "
                f"vs. non-distant (n={len(nd)}, median={nd.median():.1f})  "
                f"U={stat:.0f}, p={p:.4f}"
            )

        d = spread_df.loc[spread_df["has_distant_edge"], metric].dropna()
        nd = spread_df.loc[
            spread_df["has_genogroup_edge"] & ~spread_df["has_distant_edge"], metric
        ].dropna()
        if len(d) >= 2 and len(nd) >= 2:
            stat, p = mannwhitneyu(d, nd, alternative="two-sided")
            print(
                f"\n{metric}: distant bins (n={len(d)}, median={d.median():.1f}) "
                f"vs. genogroup (n={len(nd)}, median={nd.median():.1f})  "
                f"U={stat:.0f}, p={p:.4f}"
            )

        d = spread_df.loc[
            spread_df["has_genogroup_edge"] & ~spread_df["has_distant_edge"], metric
        ].dropna()
        nd = spread_df.loc[
            ~spread_df["has_genogroup_edge"] & ~spread_df["has_distant_edge"], metric
        ].dropna()
        if len(d) >= 2 and len(nd) >= 2:
            stat, p = mannwhitneyu(d, nd, alternative="two-sided")
            print(
                f"\n{metric}: genogroup bins (n={len(d)}, median={d.median():.1f}) "
                f"vs. clonal (n={len(nd)}, median={nd.median():.1f})  "
                f"U={stat:.0f}, p={p:.4f}"
            )

    # ── Exports ───────────────────────────────────────────────
    out_dir = f"output/hgt_summaries/"
    spread_df.to_csv(f"{out_dir}/bin_spatiotemporal_spread.csv", sep=";", index=False)
    assoc_df.to_csv(
        f"{out_dir}/spatiotemporal_gene_association.csv", sep=";", index=False
    )
    print("\n→ bin_spatiotemporal_spread.csv, spatiotemporal_gene_association.csv")
    return spread_df, assoc_df


# ============================================================
# 3. BACKGROUND-CORRECTED SPREAD
# ============================================================


def background_corrected_spread(summary_df, plasmid_df, mash_lookup, wgmlst_lookup):
    """
    For each bin with ≥2 dated members:

      1. Define the bin's space-time window:
           municipalities = union of member municipalities
           time window    = [earliest_date − BACKGROUND_WINDOW_DAYS,
                             latest_date   + BACKGROUND_WINDOW_DAYS]

      2. Identify background isolates: in the same municipalities and
         time window, NOT members of this bin.

      3. Compute:
           bin_diversity      : median pairwise host distance among
                                bin members (wgMLST preferred, mash fallback)
           background_diversity: same, among background isolates
                                (capped at 200 random pairs for speed)

      4. diversity_ratio = bin_diversity / background_diversity
         Ratio >> 1 → bin members are more genetically diverse than
         co-circulating background → HGT signal beyond clonal spread.

    Exports
    -------
    background_corrected_spread.csv
    """
    print("\n====================")
    print("BACKGROUND-CORRECTED SPREAD")
    print("====================")

    pdf = plasmid_df.copy()
    pdf["_date"] = _parse_date(pdf["MATERIAL_SAMPLINGDATE"])
    pdf["_muni"] = (
        pdf["MATERIAL_SUBMITTER_MUNICIPALITY"].fillna("").astype(str).str.strip()
    )

    # Ensure isolate ID column exists; fall back to Plasmid if not
    if ISOLATE_ID_COL not in pdf.columns:
        print(
            f"  [WARN] '{ISOLATE_ID_COL}' not found in plasmid_df; "
            "using 'Plasmid' as isolate key. Distance lookups may not match."
        )
        pdf["_isolate_id"] = pdf["Plasmid"]
    else:
        pdf["_isolate_id"] = pdf[ISOLATE_ID_COL].astype(str)

    def _median_pairwise_dist(isolate_ids, lookup, max_pairs=200):
        """Median pairwise distance for a set of isolate IDs from a lookup dict."""
        ids = list(isolate_ids)
        pairs = list(combinations(ids, 2))
        if len(pairs) > max_pairs:
            rng = np.random.default_rng(42)
            pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
        dists = [lookup.get(_canonical_pair(a, b)) for a, b in pairs]
        dists = [d for d in dists if d is not None]
        return np.median(dists) if dists else np.nan

    bc_rows = []
    for _, bin_row in summary_df.iterrows():
        members = set(bin_row["member_plasmids"])
        meta = pdf[pdf["Plasmid"].isin(members)]

        dates = meta["_date"].dropna()
        munis = set(meta.loc[meta["_muni"] != "", "_muni"])

        if len(dates) < 2 or not munis:
            continue

        t_min = dates.min() - pd.Timedelta(days=BACKGROUND_WINDOW_DAYS)
        t_max = dates.max() + pd.Timedelta(days=BACKGROUND_WINDOW_DAYS)

        # Background isolates: same municipalities, overlapping time, not in bin
        bg = pdf[
            pdf["_muni"].isin(munis)
            & pdf["_date"].between(t_min, t_max)
            & ~pdf["Plasmid"].isin(members)
            & pdf["_date"].notna()
        ]

        # Prefer wgMLST distances; fall back to mash
        member_isolates = set(meta["_isolate_id"].dropna().astype(str))
        bg_isolates = set(bg["_isolate_id"].dropna().astype(str))

        bin_div_wg = _median_pairwise_dist(member_isolates, wgmlst_lookup)
        bin_div_mash = _median_pairwise_dist(member_isolates, mash_lookup)
        bg_div_wg = _median_pairwise_dist(bg_isolates, wgmlst_lookup)
        bg_div_mash = _median_pairwise_dist(bg_isolates, mash_lookup)

        # Use wgMLST if available for both, else mash
        if pd.notna(bin_div_wg) and pd.notna(bg_div_wg):
            bin_div = bin_div_wg
            bg_div = bg_div_wg
            dist_source = "wgmlst"
        elif pd.notna(bin_div_mash) and pd.notna(bg_div_mash):
            bin_div = bin_div_mash
            bg_div = bg_div_mash
            dist_source = "mash"
        else:
            bin_div = bg_div = np.nan
            dist_source = "none"

        diversity_ratio = (
            bin_div / bg_div
            if (pd.notna(bin_div) and pd.notna(bg_div) and bg_div > 0)
            else np.nan
        )

        bc_rows.append(
            {
                "bin_id": bin_row["bin_id"],
                "bin_size": bin_row["bin_size"],
                "promiscuity_level": bin_row["promiscuity_level"],
                "has_distant_edge": bin_row["has_distant_edge"],
                "n_dated_members": len(dates),
                "n_municipalities": len(munis),
                "n_background_isolates": len(bg),
                "bin_diversity": round(bin_div, 4) if pd.notna(bin_div) else np.nan,
                "background_diversity": (
                    round(bg_div, 4) if pd.notna(bg_div) else np.nan
                ),
                "diversity_ratio": (
                    round(diversity_ratio, 4) if pd.notna(diversity_ratio) else np.nan
                ),
                "dist_source": dist_source,
                **{
                    f"contains_{g}": bin_row.get(f"contains_{g}", False)
                    for g in ACCESSORY_GENES
                },
            }
        )

    bc_df = pd.DataFrame(bc_rows)

    if not bc_df.empty:
        valid = bc_df["diversity_ratio"].dropna()
        print(f"\n  Bins with background comparison : {len(valid)}")
        print(f"  Median diversity ratio          : {valid.median():.3f}")
        print(
            f"  Bins with ratio > 1 (more diverse than background) : "
            f"{(valid > 1).sum()} ({(valid > 1).sum() / len(valid) * 100:.1f}%)"
        )

        # Distant vs. non-distant diversity ratio
        d = bc_df.loc[bc_df["has_distant_edge"], "diversity_ratio"].dropna()
        nd = bc_df.loc[~bc_df["has_distant_edge"], "diversity_ratio"].dropna()
        if len(d) >= 2 and len(nd) >= 2:
            stat, p = mannwhitneyu(d, nd, alternative="greater")
            print(
                f"\n  Diversity ratio — distant bins (n={len(d)}, "
                f"median={d.median():.3f}) vs. non-distant "
                f"(n={len(nd)}, median={nd.median():.3f})  "
                f"U={stat:.0f}, p={p:.4f}"
            )
            print(
                "  [p<0.05: distant-host bins are more genetically diverse "
                "than co-circulating background — HGT signal]"
            )

        # Gene-type association with diversity ratio
        for gene in ACCESSORY_GENES:
            col = f"contains_{gene}"
            if col not in bc_df.columns:
                continue
            pos = bc_df.loc[bc_df[col], "diversity_ratio"].dropna()
            neg = bc_df.loc[~bc_df[col], "diversity_ratio"].dropna()
            if len(pos) >= 2 and len(neg) >= 2:
                _, p = mannwhitneyu(pos, neg, alternative="two-sided")
                print(
                    f"  {gene.upper():10s}: ratio median "
                    f"{pos.median():.3f} (n={len(pos)}) vs. "
                    f"{neg.median():.3f} (n={len(neg)})  p={p:.4f}"
                )
    else:
        print("  No bins had sufficient dated + municipal data for comparison.")

    out_dir = f"output/hgt_summaries/"
    bc_df.to_csv(f"{out_dir}/background_corrected_spread.csv", sep=";", index=False)
    print("\n→ background_corrected_spread.csv")
    return bc_df


# ============================================================
# 4. TEMPORAL DIRECTIONALITY
# ============================================================


def temporal_directionality(summary_df, plasmid_df, mash_lookup, wgmlst_lookup):
    """
    For each bin with ≥ MIN_DATED_MEMBERS dated members spanning ≥2
    distinct host genetic backgrounds:

      - Sort members by sampling date.
      - Compute each member's host distance from the earliest member
        (wgMLST preferred, mash fallback using isolate IDs).
      - Spearman rho between date rank and host distance from index.
        Positive rho = progressively more distant hosts over time
        = temporal HGT acquisition signal.
      - Also record the gene content of the bin for stratification.

    Exports
    -------
    temporal_directionality.csv   (one row per bin tested)
    temporal_directionality_summary.csv  (gene-stratified rho distribution)
    """
    print("\n====================")
    print("TEMPORAL DIRECTIONALITY")
    print("====================")

    pdf = plasmid_df.copy()
    pdf["_date"] = _parse_date(pdf["MATERIAL_SAMPLINGDATE"])

    if ISOLATE_ID_COL not in pdf.columns:
        pdf["_isolate_id"] = pdf["Plasmid"]
    else:
        pdf["_isolate_id"] = pdf[ISOLATE_ID_COL].astype(str)

    dir_rows = []

    for _, bin_row in summary_df.iterrows():
        members = bin_row["member_plasmids"]
        meta = pdf[pdf["Plasmid"].isin(members)].copy()
        meta = meta.dropna(subset=["_date"]).sort_values("_date")

        if len(meta) < MIN_DATED_MEMBERS:
            continue

        isolate_ids = meta["_isolate_id"].tolist()
        index_iso = isolate_ids[0]

        # Distance from the earliest-dated isolate (index case) to each other
        dists_from_index = []
        for iso in isolate_ids[1:]:
            pair = _canonical_pair(index_iso, iso)
            # Prefer wgMLST
            d = wgmlst_lookup.get(pair)
            if d is None:
                d = mash_lookup.get(pair)
            dists_from_index.append(d)

        # Need ≥2 non-null distances across ≥2 distinct host backgrounds
        valid_pairs = [
            (rank + 1, d) for rank, d in enumerate(dists_from_index) if d is not None
        ]
        if len(valid_pairs) < MIN_DATED_MEMBERS - 1:
            continue

        ranks, dists = zip(*valid_pairs)

        # Check for ≥2 distinct host distance values (i.e. not all same background)
        if len(set(dists)) < 2:
            continue

        rho, p = spearmanr(ranks, dists)
        date_range = (meta["_date"].max() - meta["_date"].min()).days

        row = {
            "bin_id": bin_row["bin_id"],
            "n_dated_members": len(meta),
            "n_valid_pairs": len(valid_pairs),
            "index_date": meta["_date"].iloc[0].strftime("%Y-%m-%d"),
            "date_range_days": date_range,
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p, 5),
            "promiscuity_level": bin_row["promiscuity_level"],
            "has_distant_edge": bin_row["has_distant_edge"],
            "max_host_dist": bin_row["max_host_dist"],
            **{
                f"contains_{g}": bin_row.get(f"contains_{g}", False)
                for g in ACCESSORY_GENES
            },
        }
        dir_rows.append(row)

    dir_df = pd.DataFrame(dir_rows)

    if not dir_df.empty:
        dir_df["padj"] = multipletests(dir_df["spearman_p"], method="fdr_bh")[1]
        dir_df = dir_df.sort_values("spearman_rho", ascending=False)

        n_tested = len(dir_df)
        n_positive = int((dir_df["spearman_rho"] > 0).sum())
        n_sig = int((dir_df["padj"] < 0.05).sum())
        n_sig_pos = int(((dir_df["padj"] < 0.05) & (dir_df["spearman_rho"] > 0)).sum())

        print(f"\n  Bins tested for directionality : {n_tested}")
        print(
            f"  Bins with positive rho         : {n_positive} ({n_positive / n_tested * 100:.1f}%)"
        )
        print(f"  Significant (padj<0.05)        : {n_sig}")
        print(f"  Significant AND positive rho   : {n_sig_pos}  ← HGT temporal signal")

        # Gene-stratified rho summary
        strat_rows = []
        for gene in ACCESSORY_GENES:
            col = f"contains_{gene}"
            if col not in dir_df.columns:
                continue
            pos = dir_df.loc[dir_df[col], "spearman_rho"].dropna()
            neg = dir_df.loc[~dir_df[col], "spearman_rho"].dropna()
            if len(pos) >= 2 and len(neg) >= 2:
                stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
                strat_rows.append(
                    {
                        "gene_type": gene,
                        "n_gene_positive_bins": len(pos),
                        "median_rho_gene_positive": round(pos.median(), 4),
                        "n_gene_negative_bins": len(neg),
                        "median_rho_gene_negative": round(neg.median(), 4),
                        "mannwhitney_p": round(p, 5),
                    }
                )

        strat_df = pd.DataFrame(strat_rows)
        if not strat_df.empty:
            print("\nGene-stratified temporal directionality (rho):")
            print(strat_df.to_string(index=False))
    else:
        print(
            f"  No bins met the minimum of {MIN_DATED_MEMBERS} dated members "
            "across ≥2 host backgrounds."
        )
        strat_df = pd.DataFrame()

    out_dir = f"output/hgt_summaries/"
    dir_df.to_csv(f"{out_dir}/temporal_directionality.csv", sep=";", index=False)
    if not strat_df.empty:
        strat_df.to_csv(
            f"{out_dir}/temporal_directionality_summary.csv", sep=";", index=False
        )
    print("\n→ temporal_directionality.csv, temporal_directionality_summary.csv")
    return dir_df, strat_df


# ============================================================
# COORDINATOR
# ============================================================


def spatiotemporal_hgt_analysis(
    summary_df,
    plasmid_df,
    # all_pairwise_df,
    mash_dist_path,
    wgmlst_dist_path,
):
    """
    Run all four spatiotemporal / per-gene analyses.

    Call this from aggregate_and_report() after the existing
    sub-analyses (accessory_gene_enrichment, compartment_analysis,
    host_distance_analysis, global_permutation_test).

    Parameters
    ----------
    summary_df       : bin-level summary DataFrame (from process_cluster)
    plasmid_df       : metadata DataFrame with bin_id already merged
    all_pairwise_df  : concatenated pairwise edge DataFrame
    mash_dist_path   : str, path to all-vs-all mash TSV
                       (columns: subject, target, distance — tab-separated)
    wgmlst_dist_path : str, path to wgMLST square matrix TSV
                       (isolates as row and column index, tab-separated)
    """
    print("\n" + "=" * 60)
    print("SPATIOTEMPORAL HGT ANALYSIS")
    print("=" * 60)

    os.makedirs(f"output/hgt_summaries/", exist_ok=True)

    if summary_df.empty:
        print("  [WARN] summary_df is empty — skipping spatiotemporal analysis.")
        return

    # ── Load distance matrices once ───────────────────────────
    print("\nLoading distance matrices ...")
    mash_df = _load_mash(mash_dist_path)
    wgmlst_df = _load_wgmlst(wgmlst_dist_path)
    mash_lookup, wgmlst_lookup = _build_distance_lookup(mash_df, wgmlst_df)
    print(
        f"  Mash pairs loaded   : {len(mash_lookup):,}\n"
        f"  wgMLST pairs loaded : {len(wgmlst_lookup):,}"
    )

    # ── Run analyses ──────────────────────────────────────────
    per_gene_enrichment(summary_df, plasmid_df)
    spatiotemporal_spread_index(summary_df, plasmid_df)
    background_corrected_spread(summary_df, plasmid_df, mash_lookup, wgmlst_lookup)
    temporal_directionality(summary_df, plasmid_df, mash_lookup, wgmlst_lookup)

    print("\n" + "=" * 60)
    print(f"Spatiotemporal analysis complete → output/hgt_summaries/")
    print("=" * 60)


def attach_municipalities(df_in):
    df = df_in.copy()
    df.MATERIAL_SUBMITTER_CITY.fillna(df.PERSON_CITY, inplace=True)
    df.MATERIAL_SUBMITTER_PROVINCE.fillna(df.PERSON_PROVINCE, inplace=True)
    # Load geodata and clean
    NL_geo_key = pd.read_csv("data/WoonplaatsenCodes.csv", sep=";")
    NL_geo_key.drop(
        columns=[
            "Woonplaatscode",
            "Gemeente|Code ",
            "Provincie|Code",
            "Landsdeel|Naam",
            "Landsdeel|Code",
        ],
        inplace=True,
    )
    NL_geo_key.rename(
        columns={
            "Gemeente|Naam ": "MATERIAL_SUBMITTER_MUNICIPALITY",
            "Woonplaatsen": "MATERIAL_SUBMITTER_CITY",
            "Provincie|Naam": "MATERIAL_SUBMITTER_PROVINCE",
        },
        inplace=True,
    )
    NL_geo_key["MATERIAL_SUBMITTER_MUNICIPALITY"].replace(
        " ", "_", regex=True, inplace=True
    )

    # merge with incoming data
    df = pd.merge(
        df,
        NL_geo_key,
        on=["MATERIAL_SUBMITTER_CITY", "MATERIAL_SUBMITTER_PROVINCE"],
        how="left",
    )

    return df


# ---------------------------------------------------------
# 5.3.2 Detect HGT signals in bins
# ---------------------------------------------------------
def bin_post_hoc(df_in):
    os.makedirs(f"output/hgt_summaries/", exist_ok=True)
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
    plasmid_df.to_csv("plasmid_df_with_bins.csv", sep=";", index=False)

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
        "output/mash/mash_distances.tsv",
        "output/wgmlst/all_known_distance.tab",
    )

    intro_result = run_introduction_analysis(summary_df, plasmid_df)

    if intro_result is not None:
        run_postintroduction_dynamics(
            summary_df,
            intro_result["plasmid_df_ann"],
            intro_result["candidates_df"],
        )


if __name__ == "__main__":
    bin_post_hoc()
