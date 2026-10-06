# ============================================================
# PLASMID CLUSTER INTRODUCTION ANALYSIS — MRSA POST-HOC
# ============================================================
#
# Conceptual frame
# ----------------
# Plasmids are pre-assigned to clusters (column: Standard_Cluster_mrsa).
# Within a cluster, nearly-identical plasmids found in genetically
# distant host lineages (distant-edge bins) represent candidate
# recent HGT events — host adaptation after transfer would erode
# sequence identity, so near-identity implies recency.
#
# The question is NOT about within-bin dynamics. It is:
#   Has a plasmid cluster been introduced into a host lineage (ST)
#   where it was not previously circulating, and did it bring
#   resistance/virulence genes novel to that lineage?
#
# ST (ISOLATE_TL_MLST_ST) is used as the lineage proxy.
# Every distant-host bin is a candidate introduction event.
#
# Analysis layers
# ---------------
# 1. CLUSTER × ST TIMELINE
#    For every cluster × ST combination: first appearance date,
#    total isolates, gene repertoire over time.
#    Prerequisite for all downstream analyses.
#
# 2. CANDIDATE INTRODUCTION EVENTS
#    Every distant-edge bin yields one candidate per ST pair.
#    For each (cluster, ST_A, ST_B) candidate:
#      - donor ST   : earlier first-appearance of this cluster
#      - recipient ST: later first-appearance (or naive = no prior)
#      - naive_recipient flag
#      - time_gap_days: donor first-seen → recipient first-seen
#      - bin_id that triggered the candidate
#
# 3. GENE NOVELTY SCORING
#    For each candidate, relative to the recipient ST at introduction:
#      - novel_genes   : in bin, absent from recipient ST entirely
#      - rare_genes    : in bin, present in <RARE_THRESHOLD of recipient
#                        ST isolates prior to introduction date
#      - established   : already common in recipient ST
#    Computed against BOTH pre-introduction history and full history;
#    flagged separately so the reader can assess temporal bias.
#
# 4. SPATIOTEMPORAL PLAUSIBILITY
#    Do donor-ST and recipient-ST isolates spatially overlap in a
#    ±SPATIAL_WINDOW_DAYS window around the introduction date?
#    Overlap score = Jaccard of municipalities in that window.
#
# 5. PERMUTATION NULL (dataset-level)
#    Shuffle cluster assignments across isolates (preserving ST and
#    date distributions). Recount candidate introductions per shuffle.
#    Empirical p-value: are observed introductions excess of chance?
#
# 6. PER-GENE ENRICHMENT IN INTRODUCTION EVENTS
#    Are specific individual genes (parsed from comma-separated
#    columns) over-represented among introduction-associated plasmids
#    vs. the full cluster dataset? Fisher exact + FDR.
#
# 7. GEOGRAPHIC DISPERSION OF BINS
#    Recent HGT predicts tight spatial clustering (transmission is
#    local before it is national). For every bin, report how many
#    distinct municipalities its members span, stratified by
#    promiscuity level (clonal / genogroup / distant). A dispersed
#    distant bin is a weaker HGT-recency signal than a tight one.
#
# 8. POST-INTRODUCTION CLUSTER ESTABLISHMENT
#    For naive-recipient introduction candidates (from
#    hgt_introduction_analysis.py), track isolates of
#    (cluster, recipient_ST) over time after the introduction date.
#    Did the cluster establish and grow in the new lineage, or was
#    it a single dead-end detection?
#
# 9. GENE PREVALENCE TRAJECTORY POST-INTRODUCTION
#    Two framings, reported separately:
#      C1. Within (cluster, recipient_ST): does prevalence of the
#          introduced genes increase over time after introduction?
#      C2. Within recipient_ST as a whole (any cluster): did the
#          introduced genes escape onto other plasmid backbones in
#          that lineage — i.e. genuine onward dissemination beyond
#          the original transfer event?
#
# Integration
# -----------
# Call run_introduction_analysis() from aggregate_and_report()
# after existing sub-analyses. Requires:
#   plasmid_df with bin_id already merged [F2]
#   summary_df from process_cluster aggregation [F10]
# Then call run_postintroduction_dynamics()
# in aggregate_and_report(). Requires the dict returned by:
#   run_introduction_analysis() (timeline_df, candidates_df) plus the
#   annotated plasmid_df_ann (with _date, _st, _cluster, _genes, _muni)

# Key columns used from plasmid_df
# ----------------------------------
#   Plasmid                        unique plasmid ID
#   Standard_Cluster_mrsa          plasmid cluster assignment (one per plasmid)
#   ISOLATE_TL_MLST_ST             ST (lineage proxy)
#   MATERIAL_SAMPLINGDATE          dd-mm-yyyy
#   MATERIAL_SUBMITTER_MUNICIPALITY municipality
#   origin                         compartment
#   amr, virulence, biocide, metal comma-separated gene names
#   bin_id                         merged on before this call
# ============================================================

import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import config

# ── Tuneable constants ───────────────────────────────────────
RARE_THRESHOLD = 0.05  # gene present in <5 % of prior ST isolates → "rare"
SPATIAL_WINDOW_DAYS = 180  # ± days around introduction date for spatial overlap
ACCESSORY_GENES = ["amr", "virulence", "biocide", "metal"]
CLUSTER_COL = config.CLUSTER_COL
ST_COL = config.ST_COL
DATE_COL = config.DATE_COL
MUNI_COL = "municipality"


# ============================================================
# SHARED HELPERS
# ============================================================
def _parse_date(series):
    return pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")


def _parse_genes(cell):
    """Split a comma-separated gene string into a frozenset of names."""
    if pd.isna(cell) or str(cell).strip() == "":
        return frozenset()
    return frozenset(g.strip() for g in str(cell).split(",") if g.strip())


def _all_genes(row):
    """Union of genes across all accessory gene columns for one plasmid row."""
    genes = frozenset()
    for g in ACCESSORY_GENES:
        if g in row.index:
            genes |= _parse_genes(row[g])
    return genes


# ============================================================
# 1. CLUSTER × ST TIMELINE
# ============================================================
def build_cluster_st_timeline(plasmid_df):
    """
    For every (cluster, ST) combination present in the dataset,
    compute:
      first_date        : earliest sampling date
      last_date         : latest sampling date
      n_isolates        : number of plasmid-carrying isolates
      gene_repertoire   : union of all genes across all isolates (frozenset)

    Returns
    -------
    timeline_df : one row per (cluster, ST) — used as lookup table
    plasmid_df  : input df with _date, _genes, _st columns added
    """
    pdf = plasmid_df.copy()
    pdf["_date"] = _parse_date(pdf[DATE_COL])
    pdf["_st"] = pdf[ST_COL].fillna("Unknown").astype(str).str.strip()
    pdf["_cluster"] = pdf[CLUSTER_COL].astype(str).str.strip()
    pdf["_genes"] = pdf.apply(_all_genes, axis=1)
    pdf["_muni"] = pdf[MUNI_COL].fillna("").astype(str).str.strip()

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
                "first_date": dated["_date"].min() if len(dated) > 0 else pd.NaT,
                "last_date": dated["_date"].max() if len(dated) > 0 else pd.NaT,
                "gene_repertoire": gene_rep,
                "n_genes": len(gene_rep),
            }
        )

    timeline_df = pd.DataFrame(rows)
    print(f"  Cluster × ST combinations : {len(timeline_df)}")
    print(f"  Unique clusters           : {timeline_df['cluster'].nunique()}")
    print(f"  Unique STs                : {timeline_df['st'].nunique()}")

    return timeline_df, pdf


# ============================================================
# 2. CANDIDATE INTRODUCTION EVENTS
# ============================================================
def build_candidate_introductions(summary_df, plasmid_df_ann, timeline_df):
    """
    For every distant-edge OR genogroup-edge bin, enumerate
    (cluster, ST_A, ST_B) candidate introduction events.

    Every distant-edge or genogroup-edge bin yields one candidate per
    ST pair present among its members. Genogroup edges (15 ≤ wgMLST
    host distance < 1000) are a smaller host jump than distant edges
    (≥1000), but still cross ST boundaries and remain a candidate HGT
    signal — they are tagged via `edge_level` so distant and genogroup
    evidence can be reported separately at every downstream step
    without being conflated. Same-ST pairs are never generated
    (combinations() only pairs distinct STs), so within-ST
    diversification is naturally excluded.
    """
    pdf = plasmid_df_ann.copy()

    # Build fast lookup: cluster → {st → first_date}
    first_date_lookup = (
        timeline_df.dropna(subset=["first_date"])
        .set_index(["cluster", "st"])["first_date"]
        .to_dict()
    )

    # Build lookup: (cluster, st) → set of plasmid IDs
    cluster_st_plasmids = defaultdict(set)
    for _, row in pdf.iterrows():
        cluster_st_plasmids[(row["_cluster"], row["_st"])].add(row["Plasmid"])

    candidate_bins = summary_df[
        summary_df["has_distant_edge"] | summary_df["has_genogroup_edge"]
    ].copy()

    candidate_rows = []

    for _, bin_row in candidate_bins.iterrows():
        members = bin_row["member_plasmids"]
        bin_meta = pdf[pdf["Plasmid"].isin(members)]

        cluster = bin_meta["_cluster"].iloc[0]  # all members share one cluster
        bin_genes = frozenset().union(*bin_meta["_genes"])

        # STs represented in this bin
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

            # Determine donor / recipient by first-appearance date
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

            # Earliest date of recipient-ST members within this bin
            recip_members = bin_meta[bin_meta["_st"] == recipient_st]
            recip_dates = recip_members["_date"].dropna()
            bin_intro_date = recip_dates.min() if len(recip_dates) > 0 else pd.NaT

            # Naive recipient: did recipient_st carry this cluster BEFORE bin_intro_date?
            recip_prior = pdf[
                (pdf["_cluster"] == cluster)
                & (pdf["_st"] == recipient_st)
                & pdf["_date"].notna()
                & (pdf["_date"] < bin_intro_date if pd.notna(bin_intro_date) else False)
            ]
            naive_recipient = len(recip_prior) == 0

            # ── ST sampling context (addresses rare-ST confound) ──
            # How many isolates of recipient_st (ANY cluster) existed
            # before the introduction date? A "naive" recipient with
            # near-zero prior sampling is weak evidence of true absence
            # vs. just never having been looked at.
            recip_st_prior_any_cluster = pdf[
                (pdf["_st"] == recipient_st)
                & pdf["_date"].notna()
                & (pdf["_date"] < bin_intro_date if pd.notna(bin_intro_date) else False)
            ]
            n_recipient_st_prior_isolates = len(recip_st_prior_any_cluster)
            n_recipient_st_total_isolates = int((pdf["_st"] == recipient_st).sum())

            candidate_rows.append(
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
                    "recipient_st_poorly_sampled": n_recipient_st_prior_isolates
                    >= 5 & n_recipient_st_prior_isolates
                    < 10,
                    "recipient_st_rarely_sampled": n_recipient_st_prior_isolates
                    >= 1 & n_recipient_st_prior_isolates
                    < 5,
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
                    "edge_level": bin_row[
                        "promiscuity_level"
                    ],  # "distant" or "genogroup"
                    "is_distant_level": bin_row["promiscuity_level"] == "distant",
                    "max_host_dist": bin_row["max_host_dist"],
                    **{
                        f"contains_{g}": bin_row.get(f"contains_{g}", False)
                        for g in ACCESSORY_GENES
                    },
                }
            )

    candidates_df = pd.DataFrame(candidate_rows)
    return candidates_df


# ============================================================
# 3. GENE NOVELTY SCORING
# ============================================================


def score_gene_novelty(candidates_df, plasmid_df_ann, min_st_sample=10):
    """
    For each candidate introduction, score which genes the incoming
    plasmid cluster brings relative to the recipient ST's prior history.

    ST-rarity correction
    ---------------------
    A gene called "novel" purely because the recipient ST has zero or
    very few prior isolates is confounded with sampling depth, not
    biology. Genes are only classified as truly novel/rare/established
    when the recipient ST has at least `min_st_sample` isolates to
    compare against (any cluster). Below that threshold, genes are
    classified as "indeterminate_low_st_sample" instead — i.e. there
    isn't enough ST history to say whether the gene was actually absent.

    Adds columns:
      novel_genes_preintro          : genes absent from a WELL-SAMPLED
                                       recipient ST before bin_intro_date
      rare_genes_preintro           : genes rare (<RARE_THRESHOLD) in a
                                       well-sampled recipient ST before intro
      established_genes_preintro    : already common in recipient ST
      indeterminate_genes_preintro  : genes whose novelty call is unreliable
                                       because recipient ST sample size
                                       (any cluster, before intro) < min_st_sample
      n_st_sample_preintro          : recipient ST sample size used for the
                                       pre-intro novelty call (diagnostic)
      [same set repeated for *_fullhistory]
    """
    pdf = plasmid_df_ann.copy()

    def _gene_novelty(bin_genes, recipient_st, cutoff_date, use_preintro):
        """
        Return (novel, rare, established, indeterminate, n_st) for
        bin_genes relative to all recipient-ST isolates (optionally
        restricted to before cutoff_date).
        """
        mask = pdf["_st"] == recipient_st
        if use_preintro and pd.notna(cutoff_date):
            mask &= pdf["_date"] < cutoff_date

        st_isolates = pdf[mask]
        n_st = len(st_isolates)

        if n_st < min_st_sample:
            # Not enough ST history to reliably call absence vs. presence.
            # Every bin gene is flagged indeterminate rather than novel.
            return frozenset(), frozenset(), frozenset(), bin_genes, n_st

        gene_counts = defaultdict(int)
        for gs in st_isolates["_genes"]:
            for g in gs:
                gene_counts[g] += 1

        novel, rare, established = set(), set(), set()
        for gene in bin_genes:
            freq = gene_counts.get(gene, 0) / n_st
            if freq == 0:
                novel.add(gene)
            elif freq < RARE_THRESHOLD:
                rare.add(gene)
            else:
                established.add(gene)

        return (
            frozenset(novel),
            frozenset(rare),
            frozenset(established),
            frozenset(),
            n_st,
        )

    rows = []
    for _, cand in candidates_df.iterrows():
        bg = cand["bin_genes"]
        rst = cand["recipient_st"]
        intro_date = cand["bin_intro_date"]

        nov_pre, rare_pre, est_pre, indet_pre, n_st_pre = _gene_novelty(
            bg, rst, intro_date, use_preintro=True
        )
        nov_full, rare_full, est_full, indet_full, n_st_full = _gene_novelty(
            bg, rst, intro_date, use_preintro=False
        )

        rows.append(
            {
                **cand.to_dict(),
                "novel_genes_preintro": sorted(nov_pre),
                "rare_genes_preintro": sorted(rare_pre),
                "established_genes_preintro": sorted(est_pre),
                "indeterminate_genes_preintro": sorted(indet_pre),
                "n_st_sample_preintro": n_st_pre,
                "novel_genes_fullhistory": sorted(nov_full),
                "rare_genes_fullhistory": sorted(rare_full),
                "indeterminate_genes_fullhistory": sorted(indet_full),
                "n_st_sample_fullhistory": n_st_full,
                "n_novel_preintro": len(nov_pre),
                "n_rare_preintro": len(rare_pre),
                "n_indeterminate_preintro": len(indet_pre),
                "n_novel_fullhistory": len(nov_full),
                "n_rare_fullhistory": len(rare_full),
                "introduces_novel_gene": len(nov_pre) > 0,
                "introduces_rare_gene": len(rare_pre) > 0,
                "novelty_call_reliable": n_st_pre >= min_st_sample,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. SPATIOTEMPORAL PLAUSIBILITY
# ============================================================


def score_spatial_plausibility(candidates_df, plasmid_df_ann):
    """
    For each candidate introduction, compute the Jaccard overlap of
    municipalities between:
      - donor-ST isolates of this cluster within ±SPATIAL_WINDOW_DAYS
        of bin_intro_date
      - recipient-ST isolates of this cluster within the same window

    Jaccard → 0   : donor and recipient circulating in completely
                    different places → spatial transfer less plausible
    Jaccard → 1   : fully overlapping → plausible direct transmission

    Also records n_donor_munis, n_recipient_munis, n_shared_munis.
    """
    pdf = plasmid_df_ann.copy()

    def _munis_in_window(cluster, st, ref_date):
        if pd.isna(ref_date):
            mask = (pdf["_cluster"] == cluster) & (pdf["_st"] == st)
        else:
            t0 = ref_date - pd.Timedelta(days=SPATIAL_WINDOW_DAYS)
            t1 = ref_date + pd.Timedelta(days=SPATIAL_WINDOW_DAYS)
            mask = (
                (pdf["_cluster"] == cluster)
                & (pdf["_st"] == st)
                & pdf["_date"].between(t0, t1)
            )
        munis = pdf.loc[mask, "_muni"].replace("", np.nan).dropna()
        return set(munis)

    spatial_rows = []
    for _, cand in candidates_df.iterrows():
        ref = cand.get("bin_intro_date", pd.NaT)
        donor_munis = _munis_in_window(cand["cluster"], cand["donor_st"], ref)
        recip_munis = _munis_in_window(cand["cluster"], cand["recipient_st"], ref)

        shared = donor_munis & recip_munis
        union = donor_munis | recip_munis
        jaccard = len(shared) / len(union) if union else np.nan

        spatial_rows.append(
            {
                "bin_id": cand["bin_id"],
                "cluster": cand["cluster"],
                "donor_st": cand["donor_st"],
                "recipient_st": cand["recipient_st"],
                "n_donor_munis": len(donor_munis),
                "n_recipient_munis": len(recip_munis),
                "n_shared_munis": len(shared),
                "shared_municipalities": sorted(shared),
                "spatial_jaccard": round(jaccard, 4) if pd.notna(jaccard) else np.nan,
                "spatially_plausible": jaccard > 0 if pd.notna(jaccard) else False,
            }
        )

    return pd.DataFrame(spatial_rows)


# ============================================================
# 5. PER-GENE ENRICHMENT IN INTRODUCTION EVENTS
# ============================================================
def per_gene_enrichment_introductions(candidates_df, plasmid_df_ann):
    """
    Among introduction candidate plasmids, which individual genes are
    enriched relative to their prevalence in the full cluster dataset?

    For each gene:
      Fisher exact: in introduction candidate plasmid vs. not.
      Second test: in naive-recipient introductions specifically.

    Exports
    -------
    per_gene_enrichment_introductions.csv
    """
    pdf = plasmid_df_ann.copy()
    n_total = len(pdf)

    # Plasmids associated with any introduction candidate
    intro_plasmid_ids = set()
    naive_plasmid_ids = set()

    if not candidates_df.empty:
        for _, cand in candidates_df.iterrows():
            # bin members are not stored in candidates_df directly;
            # match back via bin_id → summary_df not available here,
            # so use cluster + recipient_st as proxy
            mask = (pdf["_cluster"] == cand["cluster"]) & (
                pdf["_st"] == cand["recipient_st"]
            )
            intro_plasmid_ids |= set(pdf.loc[mask, "Plasmid"])
            if cand.get("naive_recipient", False):
                naive_plasmid_ids |= set(pdf.loc[mask, "Plasmid"])

    pdf["in_intro"] = pdf["Plasmid"].isin(intro_plasmid_ids)
    pdf["in_naive_intro"] = pdf["Plasmid"].isin(naive_plasmid_ids)
    n_intro = int(pdf["in_intro"].sum())
    n_naive = int(pdf["in_naive_intro"].sum())

    rows = []
    for gene_type in ACCESSORY_GENES:
        if gene_type not in pdf.columns:
            continue

        pdf[f"_genes_{gene_type}"] = pdf[gene_type].apply(_parse_genes)
        all_genes_in_type = set(g for gs in pdf[f"_genes_{gene_type}"] for g in gs)

        for gene_name in all_genes_in_type:
            has_gene = pdf[f"_genes_{gene_type}"].apply(lambda gs: gene_name in gs)
            n_gene_total = int(has_gene.sum())

            # Test A: enriched among introduction candidates?
            n_gi = int((has_gene & pdf["in_intro"]).sum())
            n_ngi = int((~has_gene & pdf["in_intro"]).sum())
            n_gni = n_gene_total - n_gi
            n_ngni = (n_total - n_gene_total) - n_ngi
            _, p_intro = (
                fisher_exact([[n_gi, n_ngi], [n_gni, n_ngni]], alternative="greater")
                if (n_gi + n_gni > 0 and n_ngi + n_ngni > 0)
                else (np.nan, np.nan)
            )

            # Test B: enriched among naive-recipient introductions?
            n_gn = int((has_gene & pdf["in_naive_intro"]).sum())
            n_ngn = int((~has_gene & pdf["in_naive_intro"]).sum())
            n_gnn = n_gene_total - n_gn
            n_ngnn = (n_total - n_gene_total) - n_ngn
            _, p_naive = (
                fisher_exact([[n_gn, n_ngn], [n_gnn, n_ngnn]], alternative="greater")
                if (n_gn + n_gnn > 0 and n_ngn + n_ngnn > 0)
                else (np.nan, np.nan)
            )

            rows.append(
                {
                    "gene_type": gene_type,
                    "gene_name": gene_name,
                    "n_in_dataset": n_gene_total,
                    "pct_in_dataset": (
                        round(n_gene_total / n_total * 100, 3) if n_total else np.nan
                    ),
                    "n_in_intro_candidates": n_gi,
                    "pct_in_intro_candidates": (
                        round(n_gi / n_intro * 100, 3) if n_intro else np.nan
                    ),
                    "n_in_naive_intro": n_gn,
                    "pct_in_naive_intro": (
                        round(n_gn / n_naive * 100, 3) if n_naive else np.nan
                    ),
                    "fisher_p_intro": p_intro,
                    "fisher_p_naive": p_naive,
                }
            )

    enrich_df = pd.DataFrame(rows)

    if not enrich_df.empty:
        all_p = (
            enrich_df["fisher_p_intro"].tolist() + enrich_df["fisher_p_naive"].tolist()
        )
        valid_mask = [pd.notna(p) for p in all_p]
        padj_all = np.full(len(all_p), np.nan)
        valid_p = [p for p, v in zip(all_p, valid_mask) if v]
        if valid_p:
            padj_valid = multipletests(valid_p, method="fdr_bh")[1]
            j = 0
            for idx, v in enumerate(valid_mask):
                if v:
                    padj_all[idx] = padj_valid[j]
                    j += 1
        n = len(enrich_df)
        enrich_df["padj_intro"] = padj_all[:n]
        enrich_df["padj_naive"] = padj_all[n:]
        enrich_df = enrich_df.sort_values("padj_intro")

    out_dir = f"output/hgt_summaries/"
    os.makedirs(out_dir, exist_ok=True)
    enrich_df.to_csv(
        f"{out_dir}/per_gene_enrichment_introductions.csv", sep=";", index=False
    )
    print(f"\n→ per_gene_enrichment_introductions.csv  ({len(enrich_df)} genes tested)")
    return enrich_df


# ============================================================
# COORDINATOR
# ============================================================


def run_introduction_analysis(summary_df, plasmid_df):
    """
    Entry point. Call from aggregate_and_report() after existing
    sub-analyses.

    Parameters
    ----------
    summary_df  : bin-level summary DataFrame (with bin_id, member_plasmids,
                  has_distant_edge, contains_* columns)
    plasmid_df  : metadata DataFrame with bin_id already merged [F2]
    """
    print("\n" + "=" * 60)
    print("PLASMID CLUSTER INTRODUCTION ANALYSIS")
    print("=" * 60)

    out_dir = f"output/hgt_summaries/"
    os.makedirs(out_dir, exist_ok=True)

    if summary_df.empty:
        print("  [WARN] summary_df is empty — skipping introduction analysis.")
        return

    # ── 1. Build cluster × ST timeline ───────────────────────
    print("\n--- 1. Cluster × ST timeline ---")
    timeline_df, plasmid_df_ann = build_cluster_st_timeline(plasmid_df)
    timeline_export = timeline_df.copy()
    timeline_export["gene_repertoire"] = timeline_export["gene_repertoire"].apply(
        sorted
    )
    timeline_export.to_csv(f"{out_dir}/cluster_st_timeline.csv", sep=";", index=False)
    print("→ cluster_st_timeline.csv")

    # ── 2. Candidate introduction events ─────────────────────
    print("\n--- 2. Candidate introduction events ---")
    candidates_df = build_candidate_introductions(
        summary_df, plasmid_df_ann, timeline_df
    )
    n_cands = len(candidates_df)
    n_distant_level = (
        int((candidates_df["edge_level"] == "distant").sum()) if n_cands > 0 else 0
    )
    n_genogroup_level = (
        int((candidates_df["edge_level"] == "genogroup").sum()) if n_cands > 0 else 0
    )
    n_naive = int(candidates_df["naive_recipient"].sum()) if n_cands > 0 else 0
    n_naive_well_sampled = (
        int(
            (
                candidates_df["naive_recipient"]
                & candidates_df["recipient_st_well_sampled"]
            ).sum()
        )
        if n_cands > 0
        else 0
    )
    n_naive_poorly_sampled = (
        int(
            (
                candidates_df["naive_recipient"]
                & candidates_df["recipient_st_poorly_sampled"]
            ).sum()
        )
        if n_cands > 0
        else 0
    )
    n_naive_rarely_sampled = (
        int(
            (
                candidates_df["naive_recipient"]
                & candidates_df["recipient_st_rarely_sampled"]
            ).sum()
        )
        if n_cands > 0
        else 0
    )
    n_naive_un_sampled = (
        int(
            (
                candidates_df["naive_recipient"]
                & candidates_df["recipient_st_un_sampled"]
            ).sum()
        )
        if n_cands > 0
        else 0
    )
    # n_naive_poorly_sampled = n_naive - n_naive_well_sampled
    n_cands_poorly_sampled_recipient = (
        int((~candidates_df["recipient_st_well_sampled"]).sum()) if n_cands > 0 else 0
    )
    print(f"  Total candidate introductions : {n_cands}")
    print(f"    distant-edge bins   : {n_distant_level}")
    print(
        f"    genogroup-edge bins : {n_genogroup_level}  "
        "(smaller host jump, same caveats apply — still a different ST)"
    )
    print(f"  Naive-recipient candidates    : {n_naive}")
    print(
        f"    distant-level  : "
        f"{int((candidates_df['naive_recipient'] & (candidates_df['edge_level'] == 'distant')).sum())}"
    )
    print(
        f"    genogroup-level: "
        f"{int((candidates_df['naive_recipient'] & (candidates_df['edge_level'] == 'genogroup')).sum())}"
    )
    print(
        f"  Naive AND recipient ST well-sampled (≥10 prior isolates, "
        f"any cluster) : {n_naive_well_sampled}"
    )
    print(
        f"  Naive BUT recipient ST poorly sampled (≥5 prior isolates) "
        f": {n_naive_poorly_sampled}  ← weak evidence, treat with caution"
    )
    print(
        f"  Naive BUT recipient ST rarely sampled (≥1 prior isolates) "
        f": {n_naive_rarely_sampled}  ← weak evidence, treat with caution"
    )
    print(
        f"  Naive BUT recipient ST un sampled (0 prior isolates) "
        f": {n_naive_un_sampled}  ← weak evidence, treat with caution"
    )
    print(
        f"  All candidates with a poorly-sampled recipient ST "
        f"(naive or not) : {n_cands_poorly_sampled_recipient}/{n_cands} "
        f"({n_cands_poorly_sampled_recipient / n_cands * 100:.1f}%)"
    )
    print(
        "  [Naive recipients in poorly-sampled STs are weaker evidence — "
        "absence may reflect sparse sampling, not true absence of the cluster]"
    )

    if n_cands == 0:
        print(
            "  [WARN] No candidates — no distant- or genogroup-edge bins with ≥2 distinct STs."
        )
        return

    # ── 3. Gene novelty scoring ───────────────────────────────
    print("\n--- 3. Gene novelty scoring ---")
    candidates_df = score_gene_novelty(candidates_df, plasmid_df_ann)
    n_novel = int(candidates_df["introduces_novel_gene"].sum())
    n_rare = int(candidates_df["introduces_rare_gene"].sum())
    n_reliable = int(candidates_df["novelty_call_reliable"].sum())
    n_unreliable = n_cands - n_reliable
    print(f"  Candidates introducing ≥1 novel gene (pre-intro history) : {n_novel}")
    print(f"  Candidates introducing ≥1 rare gene  (pre-intro history) : {n_rare}")
    print(
        f"  Novelty calls based on well-sampled recipient ST "
        f"(≥10 prior isolates) : {n_reliable}/{n_cands}"
    )
    print(
        f"  Novelty calls INDETERMINATE due to sparse recipient ST "
        f"sampling : {n_unreliable}/{n_cands}  "
        "← genes here are NOT counted as novel, flagged separately"
    )
    n_novel_reliable = int(
        (
            candidates_df["introduces_novel_gene"]
            & candidates_df["novelty_call_reliable"]
        ).sum()
    )
    print(
        f"  Candidates introducing ≥1 novel gene AND reliably sampled "
        f": {n_novel_reliable}  ← strongest gene-novelty evidence"
    )
    print(
        f"    of which distant-level   : "
        f"{int((candidates_df['introduces_novel_gene'] & candidates_df['novelty_call_reliable'] & (candidates_df['edge_level'] == 'distant')).sum())}"
    )
    print(
        f"    of which genogroup-level : "
        f"{int((candidates_df['introduces_novel_gene'] & candidates_df['novelty_call_reliable'] & (candidates_df['edge_level'] == 'genogroup')).sum())}"
    )

    # Summarise top novel genes, split by edge level
    from collections import Counter

    for level in ["distant", "genogroup"]:
        sub = candidates_df[candidates_df["edge_level"] == level]
        all_novel = [g for genes in sub["novel_genes_preintro"] for g in genes]
        if all_novel:
            top_novel = Counter(all_novel).most_common(len(all_novel))
            print(f"\n  Most common novel genes introduced ({level}-level):")
            for gene, count in top_novel:
                print(f"    {gene}: {count} introduction events")

    # ── 4. Spatial plausibility ───────────────────────────────
    print("\n--- 4. Spatiotemporal plausibility ---")
    spatial_df = score_spatial_plausibility(candidates_df, plasmid_df_ann)
    candidates_df = candidates_df.merge(
        spatial_df.drop(columns=["cluster"]),
        on=["bin_id", "donor_st", "recipient_st"],
        how="left",
    )
    n_plaus = int(candidates_df["spatially_plausible"].sum())
    print(f"  Spatially plausible candidates (Jaccard > 0) : {n_plaus}/{n_cands}")
    if "spatial_jaccard" in candidates_df.columns:
        med_j = candidates_df["spatial_jaccard"].dropna().median()
        print(f"  Median spatial Jaccard                       : {med_j:.3f}")

    # Naive + spatially plausible = strongest signal
    strong = candidates_df[
        candidates_df["naive_recipient"] & candidates_df["spatially_plausible"]
    ]
    n_strong_distant = int((strong["edge_level"] == "distant").sum())
    n_strong_genogroup = int((strong["edge_level"] == "genogroup").sum())
    print(f"  Naive + spatially plausible (strongest signal): {len(strong)}")
    print(f"    distant-level   : {n_strong_distant}")
    print(f"    genogroup-level : {n_strong_genogroup}")

    # ── 5. Per-gene enrichment ────────────────────────────────
    print("\n--- 5. Per-gene enrichment in introduction events ---")
    per_gene_enrichment_introductions(candidates_df, plasmid_df_ann)

    # ── Export candidates ─────────────────────────────────────
    export_df = candidates_df.copy()
    for col in [
        "bin_genes",
        "novel_genes_preintro",
        "rare_genes_preintro",
        "established_genes_preintro",
        "indeterminate_genes_preintro",
        "novel_genes_fullhistory",
        "rare_genes_fullhistory",
        "indeterminate_genes_fullhistory",
        "shared_municipalities",
    ]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: (
                    ";".join(sorted(x)) if isinstance(x, (list, frozenset, set)) else x
                )
            )

    export_df.to_csv(f"{out_dir}/introduction_candidates.csv", sep=";", index=False)
    print(f"\n→ introduction_candidates.csv  ({n_cands} candidates)")

    # ── Console summary ───────────────────────────────────────
    print("\n====================")
    print("INTRODUCTION ANALYSIS SUMMARY")
    print("====================")
    n_bins_promiscuous = len(
        summary_df[summary_df["has_distant_edge"] | summary_df["has_genogroup_edge"]]
    )
    print(f"  Distant- or genogroup-edge bins analysed : {n_bins_promiscuous}")
    print(
        f"    distant-edge bins   : {len(summary_df[summary_df['has_distant_edge']])}"
    )
    print(
        f"    genogroup-edge bins : {len(summary_df[summary_df['has_genogroup_edge']])}"
    )
    print(
        f"  Candidate ST-pair introductions       : {n_cands}  "
        f"({n_distant_level} distant-level, {n_genogroup_level} genogroup-level)"
    )
    print(
        f"  Naive recipient (no prior carriage)   : {n_naive} ({n_naive/n_cands*100:.1f}%)"
    )
    print(f"    of which recipient ST well-sampled  : {n_naive_well_sampled}")
    print(
        f"    of which recipient ST poorly sampled: {n_naive_poorly_sampled}  (weak evidence)"
    )
    print(f"  Naive + spatially plausible           : {len(strong)}")
    print(
        f"  Introducing ≥1 novel gene (pre-intro) : {n_novel} ({n_novel/n_cands*100:.1f}%)"
    )
    print(f"    of which novelty call reliable      : {n_novel_reliable}")
    print(
        "\n  [No dataset-level significance test is reported. HGT does not "
        "need to occur more than a random null to be real or relevant — "
        "evidence is assessed per-candidate via the naive / well-sampled / "
        "spatially-plausible / reliably-novel-gene criteria above. See "
        "postintroduction_establishment.csv for whether candidate clusters "
        "persisted and diverged beyond the triggering bin in the recipient ST.]"
    )
    print(f"\nOutputs → output/hgt_summaries/")
    print("=" * 60)

    return {
        "timeline_df": timeline_df,
        "candidates_df": candidates_df,
        "spatial_df": spatial_df,
        "plasmid_df_ann": plasmid_df_ann,
    }


# ============================================================
# POST-INTRODUCTION DYNAMICS — MRSA PLASMID HGT POST-HOC
# ============================================================

# ============================================================

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from scipy.stats import ConstantInputWarning
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=ConstantInputWarning)

ACCESSORY_GENES = ["amr", "virulence", "biocide", "metal"]

# Minimum post-introduction isolates (any) required to attempt a
# growth-rate / trajectory estimate for a given introduction candidate.
MIN_POSTINTRO_ISOLATES = 3

# Time bin width (days) used when building prevalence-over-time curves.
PREVALENCE_BIN_DAYS = 365


# ============================================================
# A. GEOGRAPHIC DISPERSION OF BINS
# ============================================================


def geographic_dispersion(summary_df, plasmid_df_ann):
    """
    For every bin, compute geographic dispersion metrics:
      n_municipalities     : distinct municipalities among members
      shannon_evenness     : Shannon entropy of municipality distribution,
                              normalised to [0,1] (0 = all in one place,
                              1 = perfectly spread across all observed munis)
      is_tight              : n_municipalities == 1
      is_dispersed          : n_municipalities >= 3 (arbitrary descriptive cut)

    Stratifies by promiscuity_level and reports whether distant bins
    (the strongest recency signal) skew tighter than genogroup/clonal
    bins, as recent-HGT theory predicts.

    Exports
    -------
    bin_geographic_dispersion.csv
    """
    print("\n====================")
    print("GEOGRAPHIC DISPERSION OF BINS")
    print("====================")

    pdf = plasmid_df_ann.copy()

    rows = []
    for _, bin_row in summary_df.iterrows():
        members = bin_row["member_plasmids"]
        meta = pdf[pdf["Plasmid"].isin(members)]
        munis = meta["_muni"].replace("", np.nan).dropna()

        n_munis = munis.nunique()
        if len(munis) > 0:
            counts = munis.value_counts(normalize=True)
            shannon = -(counts * np.log(counts)).sum()
            max_shannon = np.log(len(counts)) if len(counts) > 1 else np.nan
            evenness = (
                shannon / max_shannon if (max_shannon and max_shannon > 0) else 0.0
            )
        else:
            evenness = np.nan

        rows.append(
            {
                "bin_id": bin_row["bin_id"],
                "bin_size": bin_row["bin_size"],
                "promiscuity_level": bin_row["promiscuity_level"],
                "has_distant_edge": bin_row["has_distant_edge"],
                "has_genogroup_edge": bin_row["has_genogroup_edge"],
                "n_members_with_muni": len(munis),
                "n_municipalities": n_munis,
                "shannon_evenness": (
                    round(evenness, 4) if pd.notna(evenness) else np.nan
                ),
                "is_tight": n_munis == 1,
                "is_dispersed": n_munis >= 3,
            }
        )

    disp_df = pd.DataFrame(rows)

    print("\nDispersion by promiscuity level:")
    for lvl in ["clonal", "genogroup", "distant"]:
        sub = disp_df[disp_df["promiscuity_level"] == lvl]
        if sub.empty:
            continue
        pct_tight = (sub["is_tight"].sum() / len(sub)) * 100
        pct_disp = (sub["is_dispersed"].sum() / len(sub)) * 100
        med_munis = sub["n_municipalities"].median()
        print(
            f"  {lvl:10s} (n={len(sub):4d}): "
            f"tight={pct_tight:5.1f}%  dispersed={pct_disp:5.1f}%  "
            f"median n_munis={med_munis:.1f}"
        )

    # Statistical comparison: distant vs. non-distant n_municipalities
    d = disp_df.loc[disp_df["has_distant_edge"], "n_municipalities"].dropna()
    nd = disp_df.loc[~disp_df["has_distant_edge"], "n_municipalities"].dropna()
    if len(d) >= 2 and len(nd) >= 2:
        stat, p = mannwhitneyu(d, nd, alternative="two-sided")
        print(
            f"\nMann-Whitney n_municipalities — distant (n={len(d)}, "
            f"median={d.median():.1f}) vs. non-distant (n={len(nd)}, "
            f"median={nd.median():.1f}): U={stat:.0f}, p={p:.4f}"
        )
        if d.median() < nd.median():
            print(
                "  → Distant bins are spatially TIGHTER than non-distant bins, "
                "consistent with recent HGT (transfer precedes wide spatial spread)."
            )
        else:
            print(
                "  → Distant bins are NOT tighter than non-distant bins — "
                "weakens the 'recent transfer' interpretation; could reflect "
                "older circulating plasmid backbones."
            )

    out_dir = f"output/hgt_summaries"
    os.makedirs(out_dir, exist_ok=True)
    disp_df.to_csv(f"{out_dir}/bin_geographic_dispersion.csv", sep=";", index=False)
    print("\n→ bin_geographic_dispersion.csv")
    return disp_df


# ============================================================
# B. POST-INTRODUCTION CLUSTER ESTABLISHMENT
# ============================================================


def postintroduction_establishment(candidates_df, plasmid_df_ann):
    """
    For each naive-recipient introduction candidate, track isolates
    of (cluster, recipient_ST) over time AFTER bin_intro_date.

    Computes:
      n_postintro_isolates : count of (cluster, recipient_ST) isolates
                              sampled after bin_intro_date (excluding
                              the introducing bin members themselves
                              is not attempted — they are part of the
                              establishment count by construction)
      followup_days        : bin_intro_date → last (cluster,recipient_ST)
                              isolate date observed in the dataset
      establishment_rate    : n_postintro_isolates / (followup_days/365.25)
                              isolates-per-year after introduction
      established          : n_postintro_isolates >= MIN_POSTINTRO_ISOLATES
                              i.e. more than a one-off detection
      n_postintro_still_in_triggering_bin : follow-up isolates still
                              classified into the SAME bin as the
                              original near-identical snapshot (no
                              detectable divergence yet)
      n_postintro_diverged_from_bin       : follow-up isolates of the
                              same cluster that are NOT in the
                              triggering bin — i.e. the plasmid has
                              moved on (host adaptation) but the
                              backbone (cluster) persists. This is
                              the strongest evidence for the model:
                              bin = recent transfer snapshot,
                              cluster = what actually disseminates
                              and establishes afterward.
      established_via_divergence : established AND most/all of that
                              establishment is via diverged isolates,
                              not isolates still sitting in the
                              original bin
      growth_trend_rho      : Spearman rho between isolate rank and date
                              for postintro isolates (proxy for accelerating
                              vs. decaying detection — weak with small n)

    Exports
    -------
    postintroduction_establishment.csv
    """
    print("\n====================")
    print("POST-INTRODUCTION CLUSTER ESTABLISHMENT")
    print("====================")

    pdf = plasmid_df_ann.copy()
    naive = candidates_df[candidates_df["naive_recipient"]].copy()

    if naive.empty:
        print("  No naive-recipient candidates — skipping.")
        empty = pd.DataFrame()
        out_dir = f"output/hgt_summaries"
        os.makedirs(out_dir, exist_ok=True)
        empty.to_csv(
            f"{out_dir}/postintroduction_establishment.csv", sep=";", index=False
        )
        return empty

    rows = []
    for _, cand in naive.iterrows():
        cluster = cand["cluster"]
        recipient_st = cand["recipient_st"]
        intro_date = cand["bin_intro_date"]

        if pd.isna(intro_date):
            continue

        postintro = pdf[
            (pdf["_cluster"] == cluster)
            & (pdf["_st"] == recipient_st)
            & pdf["_date"].notna()
            & (pdf["_date"] >= intro_date)
        ].sort_values("_date")

        n_post = len(postintro)

        # ── Divergence signal ─────────────────────────────────
        # Host-adaptation model: the triggering bin (near-identical
        # snapshot) is expected to disappear as the plasmid diverges,
        # while the CLUSTER (backbone) persists. So the strongest
        # evidence for your model is follow-up isolates of the same
        # cluster in the recipient ST that are NOT in the triggering
        # bin — i.e. the lineage carries on the backbone independent
        # of the original near-identical snapshot.
        triggering_bin_id = cand["bin_id"]
        if "bin_id" in postintro.columns:
            n_post_still_in_bin = int((postintro["bin_id"] == triggering_bin_id).sum())
            n_post_diverged = n_post - n_post_still_in_bin
        else:
            n_post_still_in_bin = np.nan
            n_post_diverged = np.nan

        if n_post == 0:
            followup_days = 0
            rate = 0.0
            rho = np.nan
        else:
            followup_days = (postintro["_date"].max() - intro_date).days
            rate = n_post / (followup_days / 365.25) if followup_days > 0 else np.nan
            if n_post >= MIN_POSTINTRO_ISOLATES:
                ranks = np.arange(n_post)
                date_ordinals = postintro["_date"].map(pd.Timestamp.toordinal).values
                if len(set(date_ordinals)) > 1:
                    rho, _ = spearmanr(ranks, date_ordinals)
                else:
                    rho = np.nan
            else:
                rho = np.nan

        rows.append(
            {
                "bin_id": cand["bin_id"],
                "cluster": cluster,
                "donor_st": cand["donor_st"],
                "recipient_st": recipient_st,
                "edge_level": cand.get(
                    "edge_level", cand.get("promiscuity_level", np.nan)
                ),
                "bin_intro_date": intro_date,
                "n_postintro_isolates": n_post,
                "n_postintro_still_in_triggering_bin": n_post_still_in_bin,
                "n_postintro_diverged_from_bin": n_post_diverged,
                "followup_days": followup_days,
                "establishment_rate_per_year": (
                    round(rate, 3) if pd.notna(rate) else np.nan
                ),
                "established": n_post >= MIN_POSTINTRO_ISOLATES,
                "established_via_divergence": (
                    pd.notna(n_post_diverged)
                    and n_post_diverged >= MIN_POSTINTRO_ISOLATES
                ),
                "growth_trend_rho": round(rho, 4) if pd.notna(rho) else np.nan,
            }
        )

    estab_df = pd.DataFrame(rows)

    if not estab_df.empty:
        n_est = int(estab_df["established"].sum())
        n_est_diverged = int(estab_df["established_via_divergence"].sum())
        n_singleton = int((estab_df["n_postintro_isolates"] == 0).sum())
        print(
            f"  Naive introductions evaluated      : {len(estab_df)}  "
            f"({int((estab_df['edge_level']=='distant').sum())} distant-level, "
            f"{int((estab_df['edge_level']=='genogroup').sum())} genogroup-level)"
        )
        print(
            f"  Established (≥{MIN_POSTINTRO_ISOLATES} follow-up isolates) : {n_est} "
            f"({n_est/len(estab_df)*100:.1f}%)"
        )
        print(
            f"  Established via DIVERGED isolates (cluster persists, "
            f"plasmid sequence has moved on from the triggering bin) "
            f": {n_est_diverged}  ← strongest evidence for host adaptation "
            f"+ stable backbone dissemination"
        )
        print(
            f"  Dead-end (0 follow-up isolates)    : {n_singleton} "
            f"({n_singleton/len(estab_df)*100:.1f}%)"
        )
        if n_est > 0:
            med_rate = estab_df.loc[
                estab_df["established"], "establishment_rate_per_year"
            ].median()
            print(
                f"  Median establishment rate (established only): {med_rate:.2f} isolates/year"
            )

    out_dir = f"output/hgt_summaries"
    os.makedirs(out_dir, exist_ok=True)
    estab_df.to_csv(
        f"{out_dir}/postintroduction_establishment.csv", sep=";", index=False
    )
    print("\n→ postintroduction_establishment.csv")
    return estab_df


# ============================================================
# C. GENE PREVALENCE TRAJECTORY POST-INTRODUCTION
# ============================================================


def _prevalence_timeseries(
    pdf, mask_cohort, genes, intro_date, bin_days=PREVALENCE_BIN_DAYS
):
    """
    Build a time-binned prevalence series for a set of genes within
    a cohort (defined by mask_cohort), starting at intro_date.

    Returns a DataFrame: time_bin_start, n_isolates, n_with_any_gene,
    prevalence, plus a Spearman rho of prevalence ~ time_bin_index.
    """
    cohort = pdf[
        mask_cohort & pdf["_date"].notna() & (pdf["_date"] >= intro_date)
    ].copy()
    if cohort.empty:
        return pd.DataFrame(), np.nan, np.nan

    cohort["_days_since_intro"] = (cohort["_date"] - intro_date).dt.days
    cohort["_bin_idx"] = (cohort["_days_since_intro"] // bin_days).astype(int)

    def _has_any_gene(gs):
        return len(gs & genes) > 0

    cohort["_has_gene"] = cohort["_genes"].apply(_has_any_gene)

    ts = (
        cohort.groupby("_bin_idx")
        .agg(n_isolates=("Plasmid", "count"), n_with_gene=("_has_gene", "sum"))
        .reset_index()
    )
    ts["prevalence"] = ts["n_with_gene"] / ts["n_isolates"]

    if len(ts) >= 3 and ts["prevalence"].nunique() > 1 and ts["_bin_idx"].nunique() > 1:
        rho, p = spearmanr(ts["_bin_idx"], ts["prevalence"])
    else:
        rho, p = np.nan, np.nan

    return ts, rho, p


def gene_prevalence_trajectory(candidates_df, plasmid_df_ann):
    """
    For each naive-recipient introduction candidate with ≥1 novel or
    rare gene (pre-introduction), track prevalence of those genes
    over time after bin_intro_date in two cohorts:

      C1. cluster-specific cohort : (cluster, recipient_ST) isolates only
      C2. lineage-wide cohort     : recipient_ST isolates regardless of
                                     cluster — tests whether the gene
                                     escaped the original plasmid backbone

    A positive rho in C2 but weak/absent in C1 is the strongest signal:
    the gene spread through the lineage independently of the original
    plasmid cluster (onward horizontal transfer beyond the index event).

    Exports
    -------
    gene_prevalence_trajectory.csv         (one row per candidate)
    gene_prevalence_timeseries_long.csv    (time-binned curves, long format)
    """
    print("\n====================")
    print("GENE PREVALENCE TRAJECTORY POST-INTRODUCTION")
    print("====================")

    pdf = plasmid_df_ann.copy()
    naive = candidates_df[candidates_df["naive_recipient"]].copy()

    if naive.empty:
        print("  No naive-recipient candidates — skipping.")
        out_dir = f"output/hgt_summaries"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame().to_csv(
            f"{out_dir}/gene_prevalence_trajectory.csv", sep=";", index=False
        )
        return pd.DataFrame(), pd.DataFrame()

    traj_rows = []
    long_rows = []

    for _, cand in naive.iterrows():
        cluster = cand["cluster"]
        recipient_st = cand["recipient_st"]
        intro_date = cand["bin_intro_date"]

        genes_to_track = set(cand.get("novel_genes_preintro", []) or []) | set(
            cand.get("rare_genes_preintro", []) or []
        )
        if not genes_to_track or pd.isna(intro_date):
            continue

        # C1: cluster-specific cohort
        mask_c1 = (pdf["_cluster"] == cluster) & (pdf["_st"] == recipient_st)
        ts_c1, rho_c1, p_c1 = _prevalence_timeseries(
            pdf, mask_c1, genes_to_track, intro_date
        )

        # C2: lineage-wide cohort (any cluster)
        mask_c2 = pdf["_st"] == recipient_st
        ts_c2, rho_c2, p_c2 = _prevalence_timeseries(
            pdf, mask_c2, genes_to_track, intro_date
        )

        traj_rows.append(
            {
                "bin_id": cand["bin_id"],
                "cluster": cluster,
                "donor_st": cand["donor_st"],
                "recipient_st": recipient_st,
                "edge_level": cand.get(
                    "edge_level", cand.get("promiscuity_level", np.nan)
                ),
                "bin_intro_date": intro_date,
                "genes_tracked": ";".join(sorted(genes_to_track)),
                "n_genes_tracked": len(genes_to_track),
                "n_timepoints_cluster_specific": len(ts_c1),
                "rho_cluster_specific": (
                    round(rho_c1, 4) if pd.notna(rho_c1) else np.nan
                ),
                "p_cluster_specific": round(p_c1, 5) if pd.notna(p_c1) else np.nan,
                "n_timepoints_lineage_wide": len(ts_c2),
                "rho_lineage_wide": round(rho_c2, 4) if pd.notna(rho_c2) else np.nan,
                "p_lineage_wide": round(p_c2, 5) if pd.notna(p_c2) else np.nan,
                "escaped_backbone_signal": (
                    pd.notna(rho_c2)
                    and rho_c2 > 0
                    and (pd.isna(rho_c1) or rho_c1 <= rho_c2)
                ),
            }
        )

        for _, r in ts_c1.iterrows():
            long_rows.append(
                {
                    "bin_id": cand["bin_id"],
                    "cohort": "cluster_specific",
                    "bin_idx": r["_bin_idx"],
                    "n_isolates": r["n_isolates"],
                    "n_with_gene": r["n_with_gene"],
                    "prevalence": r["prevalence"],
                }
            )
        for _, r in ts_c2.iterrows():
            long_rows.append(
                {
                    "bin_id": cand["bin_id"],
                    "cohort": "lineage_wide",
                    "bin_idx": r["_bin_idx"],
                    "n_isolates": r["n_isolates"],
                    "n_with_gene": r["n_with_gene"],
                    "prevalence": r["prevalence"],
                }
            )

    traj_df = pd.DataFrame(traj_rows)
    long_df = pd.DataFrame(long_rows)

    if not traj_df.empty:
        # FDR across both rho columns jointly
        all_p = (
            traj_df["p_cluster_specific"].tolist() + traj_df["p_lineage_wide"].tolist()
        )
        valid_mask = [pd.notna(p) for p in all_p]
        padj_all = np.full(len(all_p), np.nan)
        valid_p = [p for p, v in zip(all_p, valid_mask) if v]
        if valid_p:
            padj_valid = multipletests(valid_p, method="fdr_bh")[1]
            j = 0
            for idx, v in enumerate(valid_mask):
                if v:
                    padj_all[idx] = padj_valid[j]
                    j += 1
        n = len(traj_df)
        traj_df["padj_cluster_specific"] = padj_all[:n]
        traj_df["padj_lineage_wide"] = padj_all[n:]

        n_escaped = int(traj_df["escaped_backbone_signal"].sum())
        print(
            f"  Candidates with trackable novel/rare genes : {len(traj_df)}  "
            f"({int((traj_df['edge_level']=='distant').sum())} distant-level, "
            f"{int((traj_df['edge_level']=='genogroup').sum())} genogroup-level)"
        )
        print(
            f"  Possible backbone-escape signal (gene spreading"
            f" lineage-wide beyond original cluster) : {n_escaped}"
        )

        sig_lineage = traj_df[
            (traj_df["padj_lineage_wide"] < 0.05) & (traj_df["rho_lineage_wide"] > 0)
        ]
        print(
            f"  Significant positive lineage-wide trend (padj<0.05): {len(sig_lineage)}"
        )
        if not sig_lineage.empty:
            print(
                sig_lineage[
                    [
                        "bin_id",
                        "cluster",
                        "recipient_st",
                        "genes_tracked",
                        "rho_lineage_wide",
                        "padj_lineage_wide",
                    ]
                ].to_string(index=False)
            )

    out_dir = f"output/hgt_summaries"
    os.makedirs(out_dir, exist_ok=True)
    traj_df.to_csv(f"{out_dir}/gene_prevalence_trajectory.csv", sep=";", index=False)
    long_df.to_csv(
        f"{out_dir}/gene_prevalence_timeseries_long.csv", sep=";", index=False
    )
    print("\n→ gene_prevalence_trajectory.csv, gene_prevalence_timeseries_long.csv")
    return traj_df, long_df


# ============================================================
# COORDINATOR
# ============================================================
def run_postintroduction_dynamics(summary_df, plasmid_df_ann, candidates_df):
    """
    Entry point. Call after run_introduction_analysis() in
    aggregate_and_report(), passing its returned candidates_df and
    the annotated plasmid_df (plasmid_df_ann, with _date/_st/_cluster/
    _genes/_muni columns already added by build_cluster_st_timeline).

    Parameters
    ----------
    summary_df      : bin-level summary DataFrame
    plasmid_df_ann  : annotated plasmid_df (output of
                       build_cluster_st_timeline in
                       hgt_introduction_analysis.py)
    candidates_df   : output of run_introduction_analysis()["candidates_df"]
                       (must include novel_genes_preintro / rare_genes_preintro
                       from score_gene_novelty, and bin_intro_date)
    """
    print("\n" + "=" * 60)
    print("POST-INTRODUCTION DYNAMICS")
    print("=" * 60)

    disp_df = geographic_dispersion(summary_df, plasmid_df_ann)

    if candidates_df is None or candidates_df.empty:
        print(
            "\n  [WARN] No introduction candidates available — "
            "skipping establishment and gene trajectory analyses."
        )
        return {"dispersion_df": disp_df}

    estab_df = postintroduction_establishment(candidates_df, plasmid_df_ann)
    traj_df, long_df = gene_prevalence_trajectory(candidates_df, plasmid_df_ann)

    print("\n" + "=" * 60)
    print(f"Post-introduction dynamics complete → output/hgt_summaries/")
    print("=" * 60)

    return {
        "dispersion_df": disp_df,
        "establishment_df": estab_df,
        "trajectory_df": traj_df,
        "trajectory_timeseries_long": long_df,
    }
