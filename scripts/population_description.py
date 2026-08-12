import pandas as pd
import numpy as np
import os
import geopandas as gp
from itertools import combinations

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, fisher_exact, chi2
from scipy.spatial.distance import pdist, squareform

from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

import config
from helper_functions import clean_plasmid_df
import plotting_functions as plot


# ---------------------------------------------------------
# 3.1.0 Helper Functions
# ---------------------------------------------------------
def plot_population(df_in: pd.DataFrame):
    parent_df = df_in[
        [config.PARENT_COL, "city", "municipality", "province"]
    ].drop_duplicates()

    map_counts = parent_df.value_counts(["municipality", "province"]).reset_index()

    # # Map loading
    mapNL = gp.read_file("data/NLenBESenCAS_2024.json")
    map_boxes = mapNL.loc[mapNL["regio_soort"] == "rand"]
    municiple_map = mapNL.loc[mapNL["regio_soort"] == "GM"].copy()
    municiple_map["regio_naam"] = municiple_map["regio_naam"].str.replace(
        " ", "_", regex=True
    )
    municiple_map = municiple_map.rename(
        columns={
            "regio_naam": "municipality",
            "pv_naam": "province",
        }
    )
    municiple_map = municiple_map.merge(
        map_counts, on=["municipality", "province"], how="left"
    )
    plot.plot_isolate_data(df_in, municiple_map, map_boxes)


def save_glm_tables(
    model,
    dispersion,
    name,
    reference_categories,
    term_labels=None,
):
    """
    Save publication-ready GLM results as two CSV files.

    Table A:
        Model-level information.

    Table B:
        Exponentiated coefficients (IRRs) with 95% CIs.
        Intercept is excluded.
        Reference categories are explicitly included.

    Parameters
    ----------
    model : statsmodels GLMResults
        Fitted GLM.

    dispersion : float
        Pearson dispersion statistic.

    name : str
        Prefix used for output files.

    reference_categories : dict
        Reference categories, e.g.
        {"origin": "CA-MRSA"}

    term_labels : dict, optional
        Dictionary to make coefficient names more readable.

    """

    # =========================================================
    # TABLE A: MODEL INFORMATION
    # =========================================================
    model_info = pd.DataFrame(
        {
            "Statistic": [
                "Dependent variable",
                "No. observations",
                "Df residuals",
                "Df model",
                "Model family",
                "Link function",
                "Log-likelihood",
                "AIC",
                "Deviance",
                "Pearson chi-square",
                "Pseudo R-squared (CS)",
                "Covariance type",
                "Pearson dispersion",
            ],
            "Value": [
                model.model.endog_names,
                int(model.nobs),
                int(model.df_resid),
                int(model.df_model),
                model.family.__class__.__name__,
                model.family.link.__class__.__name__,
                model.llf,
                model.aic,
                model.deviance,
                model.pearson_chi2,
                model.pseudo_rsquared(kind="cs"),
                model.cov_type,
                dispersion,
            ],
        }
    )

    # Round numerical values for publication
    numerical_rows = [
        "Log-likelihood",
        "AIC",
        "Deviance",
        "Pearson chi-square",
        "Pseudo R-squared (CS)",
        "Pearson dispersion",
    ]

    model_info.loc[
        model_info["Statistic"].isin(numerical_rows),
        "Value",
    ] = (
        model_info.loc[
            model_info["Statistic"].isin(numerical_rows),
            "Value",
        ]
        .astype(float)
        .round(3)
    )

    model_info.to_csv(
        f"results/tables/{name}_A.csv",
        sep=";",
        index=False,
    )

    # =========================================================
    # TABLE B: COEFFICIENTS
    # =========================================================
    conf_int = model.conf_int()

    results = pd.DataFrame(
        {
            "Term": model.params.index,
            "IRR": np.exp(model.params.values),
            "SE": model.bse.values,
            "z": model.tvalues.values,
            "P_value": model.pvalues.values,
            "CI_lower": np.exp(conf_int[0].values),
            "CI_upper": np.exp(conf_int[1].values),
        }
    )

    # Remove intercept
    results = results[results["Term"] != "Intercept"].copy()

    # ---------------------------------------------------------
    # Make term names readable
    # ---------------------------------------------------------

    def clean_term(term):

        term = term.replace("C(origin)[T.", "")
        term = term.replace("C(ST_collapsed)[T.", "")
        term = term.replace("]", "")

        return term

    results["Term"] = results["Term"].apply(clean_term)

    if term_labels is not None:
        results["Term"] = results["Term"].replace(term_labels)

    # ---------------------------------------------------------
    # Add reference categories
    # ---------------------------------------------------------

    reference_rows = []

    for variable, category in reference_categories.items():

        variable_label = {
            "origin": "Origin",
            "ST_collapsed": "ST",
        }.get(variable, variable)

        reference_rows.append(
            {
                "Term": f"{variable_label}: {category} (reference)",
                "IRR": 1.0,
                "SE": np.nan,
                "z": np.nan,
                "P_value": np.nan,
                "CI_lower": np.nan,
                "CI_upper": np.nan,
            }
        )

    reference_df = pd.DataFrame(reference_rows)

    results = pd.concat(
        [reference_df, results],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Format values
    # ---------------------------------------------------------

    results["IRR"] = results["IRR"].round(3)
    results["SE"] = results["SE"].round(3)
    results["z"] = results["z"].round(3)
    results["CI_lower"] = results["CI_lower"].round(3)
    results["CI_upper"] = results["CI_upper"].round(3)

    results["P_value"] = results["P_value"].apply(
        lambda x: ("" if pd.isna(x) else "<0.001" if x < 0.001 else f"{x:.3f}")
    )

    # More explicit column names for supplementary material
    results = results.rename(
        columns={
            "CI_lower": "IRR_95CI_lower",
            "CI_upper": "IRR_95CI_upper",
            "P_value": "P_value",
        }
    )

    results.to_csv(
        f"results/tables/{name}_B.csv",
        sep=";",
        index=False,
    )

    return model_info, results


def plasmid_carriage_counting(df_in):
    df = df_in.copy()

    # ---------------------------------------------------------
    # 1. Create isolate-level data
    # ---------------------------------------------------------
    isolate_df = df[
        [
            "Parent",
            "ISOLATE_TL_MLST_ST",
            "origin",
        ]
    ].drop_duplicates()

    # ---------------------------------------------------------
    # 2. Determine common STs based on NUMBER OF ISOLATES
    # ---------------------------------------------------------
    isolate_st_counts = isolate_df["ISOLATE_TL_MLST_ST"].value_counts()
    common_STs = isolate_st_counts[isolate_st_counts >= 10].index
    isolate_df["ST_collapsed"] = isolate_df["ISOLATE_TL_MLST_ST"].where(
        isolate_df["ISOLATE_TL_MLST_ST"].isin(common_STs),
        "rare_ST",
    )

    # ---------------------------------------------------------
    # 3. Count plasmids per isolate
    # ---------------------------------------------------------
    plasmid_counts = df.groupby("Parent").size().reset_index(name="plasmid_count")

    # ---------------------------------------------------------
    # 4. Combine plasmid count with isolate metadata
    # ---------------------------------------------------------
    analysis_df = isolate_df.merge(
        plasmid_counts,
        on="Parent",
        how="left",
    )
    print(analysis_df)
    print(analysis_df["plasmid_count"].describe())
    print(analysis_df.groupby("origin")["plasmid_count"].describe())

    # ---------------------------------------------------------
    # 5. Save counts
    # ---------------------------------------------------------
    analysis_df.to_csv(
        "plasmid_counts_per_isolate.csv",
        sep=";",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Model: plasmid count ~ origin
    # ---------------------------------------------------------
    model_origin = smf.glm(
        "plasmid_count ~ C(origin)",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit()

    pearson_dispersion = sum(model_origin.resid_pearson**2) / model_origin.df_resid
    print(pearson_dispersion)
    print(model_origin.summary())
    save_glm_tables(
        model_origin,
        pearson_dispersion,
        "tableS1_plasmids_by_origin",
        reference_categories={
            "origin": "CA-MRSA",
        },
    )

    # ---------------------------------------------------------
    # 7. Model: plasmid count ~ ST
    # ---------------------------------------------------------
    model_ST = smf.glm(
        "plasmid_count ~ C(ST_collapsed)",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit()

    pearson_dispersion = sum(model_ST.resid_pearson**2) / model_ST.df_resid
    print(pearson_dispersion)
    print(model_ST.summary())
    save_glm_tables(
        model_ST,
        pearson_dispersion,
        "tableS2_plasmids_by_ST",
        reference_categories={
            "ST_collapsed": "1",
        },
    )

    # ---------------------------------------------------------
    # 8. Model: plasmid count ~ ST + origin
    # ---------------------------------------------------------
    model_both = smf.glm(
        "plasmid_count ~ C(ST_collapsed) + C(origin)",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit()

    pearson_dispersion = sum(model_both.resid_pearson**2) / model_both.df_resid
    print(pearson_dispersion)
    print(model_both.summary())
    save_glm_tables(
        model_both,
        pearson_dispersion,
        "tableS3_plasmids_by_both",
        reference_categories={
            "origin": "CA-MRSA",
            "ST_collapsed": "1",
        },
    )
    plot.glm_forest(
        models=[
            model_origin,
            model_ST,
            model_both,
        ],
        panel_titles=[
            "A. Epidemiological origin",
            "B. ST",
            "C. ST adjusted for origin",
        ],
        reference_categories=[
            {"origin": "CA-MRSA"},
            {"ST_collapsed": "1"},
            {
                "ST_collapsed": "1",
                "origin": "CA-MRSA",
            },
        ],
        output_name="figure2_forest_plot",
    )


def count_column_composition(df_in: pd.DataFrame, col: str, split: bool = False):
    df = df_in.copy()

    if not split:
        counts = df[col].value_counts()
        percentages = (
            df[col].value_counts(normalize=True).mul(100).round(1).astype(str) + "%"
        )
    else:
        s = df[col].dropna().astype(str).str.split(",").explode().str.strip()
        counts = s.value_counts()
        percentages = s.value_counts(normalize=True).mul(100).round(1).astype(str) + "%"

    col_values = pd.concat([counts, percentages], axis=1)
    col_values.columns = ["count", "percentage"]
    return col_values


def test_gene_origin_association(
    gene,
    plasmid_gene_binary,
    origin_col="origin",
    min_positive=5,
    n_permutations=2000,
    random_state=0,
):
    """
    Test whether a gene's presence/absence is associated with plasmid
    origin/compartment (designed for a 4-way comparison, e.g.
    LA-MRSA / HA-MRSA / CA-MRSA / other).

    Statistics reported
    --------------------
    - Omnibus chi-square test (Pearson) across all compartments, with
      Cramer's V as effect size (correct generalization for r x c tables,
      not just 2x2).
    - A Monte-Carlo permutation p-value (`p_chi2_perm`) for the same
      chi-square statistic. This is the practical stand-in for the
      Fisher-Freeman-Halton exact test (the proper r x c generalization
      of Fisher's exact test), which isn't available in scipy/statsmodels
      without an R dependency. Use this instead of `p_chi2` whenever
      `low_expected_counts` is True, since the asymptotic chi-square
      p-value is unreliable when >20% of expected cell counts are < 5
      (common here, since many genes will be rare in 1-2 compartments).
    - Post-hoc pairwise tests: for each compartment vs. the other three
      pooled, a 2x2 Fisher exact test (odds ratio + p-value), BH-corrected
      across the 4 compartments. Use this to see *which* compartment(s)
      drive a significant omnibus result.

    Parameters
    ----------
    gene : str
        Gene name (column in plasmid_gene_binary)
    plasmid_gene_binary : pd.DataFrame
        Plasmid-level binary gene matrix (0/1 columns), must also contain
        `origin_col`
    origin_col : str
        Column with the 4 compartment/origin labels
    min_positive : int
        Minimum number of gene-positive plasmids required to test
    n_permutations : int
        Number of label-shuffling permutations for the Monte-Carlo p-value
    random_state : int
        Seed for reproducibility

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(random_state)

    gene_status = plasmid_gene_binary[gene]
    origin = plasmid_gene_binary[origin_col]

    n_pos = int((gene_status == 1).sum())
    n_neg = int((gene_status == 0).sum())

    if n_pos < min_positive:
        return {
            "gene": gene,
            "n_gene_positive": n_pos,
            "tested": False,
            "reason": "Too few gene-positive plasmids",
        }

    contingency = pd.crosstab(gene_status, origin)
    contingency.index = ["gene_negative", "gene_positive"]

    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    low_expected_counts = bool((expected < 5).mean() > 0.2)

    # Monte-Carlo permutation p-value (label shuffling of origin)
    origin_values = origin.to_numpy()
    gene_values = gene_status.to_numpy()
    perm_chi2 = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(origin_values)
        perm_table = pd.crosstab(gene_values, shuffled)
        perm_chi2[i], _, _, _ = chi2_contingency(perm_table)
    p_chi2_perm = (np.sum(perm_chi2 >= chi2) + 1) / (n_permutations + 1)

    n = contingency.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    residuals = (contingency - expected) / np.sqrt(expected)

    # Post-hoc: each compartment vs. the rest, Fisher exact (2x2), BH-corrected
    posthoc_rows = []
    for compartment in contingency.columns:
        two_by_two = pd.DataFrame(
            {
                compartment: contingency[compartment],
                "Other": contingency.drop(columns=compartment).sum(axis=1),
            }
        )
        odds_ratio, p_fisher = fisher_exact(two_by_two)
        posthoc_rows.append(
            {"compartment": compartment, "odds_ratio": odds_ratio, "p_fisher": p_fisher}
        )

    posthoc_df = pd.DataFrame(posthoc_rows)
    posthoc_df["p_fisher_adj"] = multipletests(posthoc_df["p_fisher"], method="fdr_bh")[
        1
    ]

    return {
        "gene": gene,
        "tested": True,
        "n_gene_positive": n_pos,
        "n_gene_negative": n_neg,
        "chi2": chi2,
        "p_chi2": p_chi2,
        "p_chi2_perm": p_chi2_perm,
        "low_expected_counts": low_expected_counts,
        "cramers_v": cramers_v,
        "contingency": contingency,
        "residuals": residuals,
        "posthoc": posthoc_df,
    }


def gene_origin_enrichment(df_in, min_positive=10, n_permutations=2000, random_state=0):
    """
    Gene presence/absence vs. plasmid origin/compartment (4 categories),
    tested directly at the plasmid level -- no cluster-level aggregation.

    Parameters
    ----------
    df_in : pd.DataFrame
        Must contain 'Plasmid', 'origin', and the gene-list columns
        ["amr", "virulence", "metal", "biocide", "heat", "acid"]
    min_positive : int
        Minimum gene-positive plasmids required to test a gene
    n_permutations : int
        Permutations for the Monte-Carlo chi-square p-value
    random_state : int
        Seed

    Returns
    -------
    pd.DataFrame
        One row per tested gene, sorted by p_chi2, with BH-adjusted
        p-values for both the asymptotic and permutation chi-square tests.
    """
    df = df_in.copy()

    gene_columns = [
        config.AMR_COL,
        config.VIR_COL,
        config.METAL_COL,
        config.BIOCIDE_COL,
    ]

    long_genes = (
        df.set_index("Plasmid")[gene_columns]
        .stack()
        .reset_index(level=1, drop=True)
        .reset_index(name="gene_list")
    )
    long_genes = (
        long_genes.dropna(subset=["gene_list"])
        .assign(gene=lambda x: x["gene_list"].str.split(","))
        .explode("gene")
    )
    long_genes["gene"] = long_genes["gene"].str.strip()

    gene_names = sorted(long_genes["gene"].unique())
    print(f"{len(gene_names)} unique genes")

    ## Plasmid x gene binary matrix
    plasmid_gene_matrix = long_genes.assign(present=1).pivot_table(
        index="Plasmid", columns="gene", values="present", aggfunc="max", fill_value=0
    )
    all_plasmids = df["Plasmid"].unique()
    plasmid_gene_matrix = plasmid_gene_matrix.reindex(all_plasmids, fill_value=0)

    metadata = df[["Plasmid", "origin"]].drop_duplicates().set_index("Plasmid")
    plasmid_gene_binary = plasmid_gene_matrix.join(metadata)

    ## Test each gene against origin (4 compartments)
    results = [
        test_gene_origin_association(
            gene,
            plasmid_gene_binary,
            origin_col="origin",
            min_positive=min_positive,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        for gene in gene_names
    ]

    results_df = pd.DataFrame(results).query("tested == True").sort_values("p_chi2")
    results_df["p_chi2_adj"] = multipletests(results_df["p_chi2"], method="fdr_bh")[1]
    results_df["p_chi2_perm_adj"] = multipletests(
        results_df["p_chi2_perm"], method="fdr_bh"
    )[1]

    print(
        results_df[
            [
                "gene",
                "n_gene_positive",
                "p_chi2",
                "p_chi2_adj",
                "p_chi2_perm",
                "p_chi2_perm_adj",
                "cramers_v",
                "low_expected_counts",
            ]
        ]
    )

    return results_df


# ---------------------------------------------------------
# 3.2.0 Helper Functions
# ---------------------------------------------------------
def count_column_composition_by_cluster(
    df_in: pd.DataFrame,
    column_col: str,
    row_col: str = config.CLUSTER_COL,
    split: bool = False,
):
    df = df_in.copy()

    if not split:
        counts = pd.crosstab(
            df[row_col],
            df[column_col],
        )

        percentages = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)

    else:
        # Split into categories and count each category only once per row
        split_df = (
            df[[row_col, column_col]]
            .dropna(subset=[column_col])
            .assign(**{column_col: lambda x: x[column_col].astype(str).str.split(",")})
            .explode(column_col)
        )

        split_df[column_col] = split_df[column_col].str.strip()

        # Remove duplicate category combinations within a row
        split_df = split_df.drop_duplicates()

        counts = pd.crosstab(
            split_df[row_col],
            split_df[column_col],
        )

        # Denominator = total number of original rows in each row category
        row_totals = df.groupby(row_col).size()

        percentages = counts.div(row_totals, axis=0).mul(100).round(1)

    return counts.astype(str) + " (" + percentages.astype(str) + "%)"


def build_plasmidome_distance(
    df, cluster_col: str, isolate_col: str = "Parent", origin_col="origin"
):
    """
    Build a Jaccard distance matrix between isolates based on which
    plasmid backbone clusters they carry (presence/absence "plasmidome").
    """
    df_unique = df[[isolate_col, cluster_col]].drop_duplicates()

    isolate_cluster_matrix = (
        pd.crosstab(df_unique[isolate_col], df_unique[cluster_col])
        .astype(bool)
        .astype(int)
    )

    metadata = (
        df[[isolate_col, origin_col]]
        .drop_duplicates()
        .set_index(isolate_col)
        .loc[isolate_cluster_matrix.index]
    )

    jaccard_distances = pdist(isolate_cluster_matrix.values, metric="jaccard")
    dm = DistanceMatrix(
        squareform(jaccard_distances), ids=isolate_cluster_matrix.index.tolist()
    )

    return dm, metadata


def pairwise_permanova(dm, metadata, group_col="origin", n_perm=999):
    """
    Post-hoc pairwise PERMANOVA between each pair of compartments,
    BH-corrected. A significant global (4-group) PERMANOVA only tells
    you *some* compartments differ, not which ones -- this fills that gap.
    """
    groups = metadata[group_col].unique()
    rows = []
    for g1, g2 in combinations(groups, 2):
        ids = metadata.index[metadata[group_col].isin([g1, g2])].tolist()
        dm_sub = dm.filter(ids)
        meta_sub = metadata.loc[ids]
        res = permanova(
            distance_matrix=dm_sub, grouping=meta_sub[group_col], permutations=n_perm
        )
        rows.append(
            {
                "group1": g1,
                "group2": g2,
                "F": res["test statistic"],
                "p": res["p-value"],
                "n1": int((meta_sub[group_col] == g1).sum()),
                "n2": int((meta_sub[group_col] == g2).sum()),
            }
        )
    pairwise_df = pd.DataFrame(rows)
    pairwise_df["p_adj"] = multipletests(pairwise_df["p"], method="fdr_bh")[1]
    return pairwise_df.sort_values("p")


def distance_to_centroid(coords, metadata, group_col):
    """
    Multivariate dispersion per isolate: distance from the isolate's PCoA
    position to its group's centroid. Same quantity tested by
    PERMDISP/betadisper -- a required companion to PERMANOVA, since
    PERMANOVA can be significant either because groups differ in
    *location* (true composition) or in *dispersion* (within-group
    variability), and the two need to be told apart before interpreting
    the PERMANOVA result.
    """
    id_name = metadata.index.name or "id"
    merged = coords.join(metadata[group_col])
    rows = []
    for group, subdf in merged.groupby(group_col):
        if len(subdf) < 2:
            continue
        centroid = subdf.drop(columns=group_col).mean(axis=0)
        dists = np.linalg.norm(
            subdf.drop(columns=group_col).values - centroid.values, axis=1
        )
        rows.extend(zip(subdf.index, [group] * len(dists), dists))
    return pd.DataFrame(
        rows, columns=[id_name, group_col, "distance_to_centroid"]
    ).set_index(id_name)


def compartment_plasmidome_dispersion(
    df_in: pd.DataFrame, cluster_col: str, n_perm=999
):
    """
    Test whether plasmidome composition (plasmid backbone cluster
    presence/absence) differs significantly between the 4 origin
    compartments.

    Two complementary tests are run, both as a global (4-group) omnibus
    test followed by BH-corrected pairwise post-hoc tests:

      1. PERMANOVA -- do compartments differ in composition (centroid
         location in Jaccard-distance space)?
      2. Dispersion (PERMDISP-style: distance-to-centroid + Kruskal-Wallis,
         then pairwise Mann-Whitney) -- do compartments differ in
         within-group variability? A significant PERMANOVA alongside a
         significant dispersion difference means the PERMANOVA result may
         partly/wholly reflect dispersion rather than a true compositional
         shift, so this needs to be checked before interpreting #1.

    Parameters
    ----------
    df_in : pd.DataFrame
        Must contain 'Parent' (isolate id), cluster_col
        (plasmid backbone cluster), and 'origin' (compartment, 4 levels)
    cluster_col : str
        Column name containing the cluster IDs
    n_perm : int
        Number of permutations for PERMANOVA

    Returns
    -------
    dict
    """
    df = df_in.copy()

    dm, metadata = build_plasmidome_distance(df, cluster_col)

    ## Global PERMANOVA (omnibus, all 4 compartments)
    global_res = permanova(
        distance_matrix=dm, grouping=metadata["origin"], permutations=n_perm
    )
    print("Global PERMANOVA (origin):")
    print(global_res)

    ## Pairwise PERMANOVA post-hoc
    pairwise_res = pairwise_permanova(dm, metadata, group_col="origin", n_perm=n_perm)
    print("\nPairwise PERMANOVA (BH-corrected):")
    print(pairwise_res)

    ## Dispersion (PERMDISP-style) test
    pcoa_res = pcoa(dm)
    coords_weighted = pcoa_res.samples.copy()
    for i, eig in enumerate(pcoa_res.eigvals):
        coords_weighted.iloc[:, i] *= np.sqrt(eig) if eig > 0 else 0.0

    disp_df = distance_to_centroid(coords_weighted, metadata, "origin")

    disp_summary = (
        disp_df.groupby("origin")["distance_to_centroid"]
        .median()
        .rename("median")
        .to_frame()
    )
    disp_summary["Q1"] = disp_df.groupby("origin")["distance_to_centroid"].quantile(
        0.25
    )
    disp_summary["Q3"] = disp_df.groupby("origin")["distance_to_centroid"].quantile(
        0.75
    )
    disp_summary = disp_summary.sort_values("median", ascending=False)
    print("\nDispersion by compartment:")
    print(disp_summary)

    groups_comp = [
        disp_df.loc[disp_df["origin"] == g, "distance_to_centroid"].values
        for g in disp_df["origin"].unique()
    ]
    stat, pval = kruskal(*groups_comp)
    print(f"\nOmnibus dispersion test (Kruskal-Wallis): H={stat:.2f}, p={pval:.4e}")

    ## Pairwise dispersion post-hoc (Mann-Whitney, BH-corrected)
    disp_pairs = []
    for g1, g2 in combinations(disp_df["origin"].unique(), 2):
        x = disp_df.loc[disp_df["origin"] == g1, "distance_to_centroid"]
        y = disp_df.loc[disp_df["origin"] == g2, "distance_to_centroid"]
        stat_mw, p_mw = mannwhitneyu(x, y)
        disp_pairs.append({"group1": g1, "group2": g2, "U": stat_mw, "p": p_mw})
    disp_pairs_df = pd.DataFrame(disp_pairs)
    disp_pairs_df["p_adj"] = multipletests(disp_pairs_df["p"], method="fdr_bh")[1]
    print("\nPairwise dispersion (Mann-Whitney, BH-corrected):")
    print(disp_pairs_df.sort_values("p"))

    return {
        "distance_matrix": dm,
        "metadata": metadata,
        "global_permanova": global_res,
        "pairwise_permanova": pairwise_res,
        "dispersion_summary": disp_summary,
        "dispersion_kruskal": (stat, pval),
        "dispersion_pairwise": disp_pairs_df,
    }


def gene_spillover_analysis(
    df_in: pd.DataFrame, cluster_col: str, group_col="origin", gene_columns=None
):
    """
    Quantify gene spillover between compartments within plasmid clusters.
    """

    if gene_columns is None:
        gene_columns = ["amr", "virulence", "metal", "biocide", "heat", "acid"]

    # Step 1: long-format gene table
    long_genes = (
        df_in.set_index("Plasmid")[gene_columns]
        .stack()
        .reset_index(level=1, drop=True)
        .reset_index(name="gene_list")
    )
    long_genes = (
        long_genes.dropna(subset=["gene_list"])
        .assign(gene=lambda x: x["gene_list"].str.split(","))
        .explode("gene")
    )
    long_genes["gene"] = long_genes["gene"].str.strip()

    target_genes = long_genes["gene"].unique()

    # Step 2: plasmid × gene presence matrix
    plasmid_metadata = df_in[["Plasmid", cluster_col, group_col]].drop_duplicates()

    plasmid_gene_matrix = long_genes.assign(present=1).pivot_table(
        index="Plasmid",
        columns="gene",
        values="present",
        aggfunc="max",
        fill_value=0,
    )

    plasmid_gene_matrix = plasmid_gene_matrix.join(
        plasmid_metadata.set_index("Plasmid")
    )

    # Precompute cluster-level compartment sets
    cluster_all_compartments = (
        plasmid_metadata.groupby(cluster_col)[group_col].apply(set).to_dict()
    )

    # Step 3: cluster × compartment presence for each gene
    cluster_compartment_dict = {}
    cluster_gene_counts = {}

    for gene in target_genes:
        gene_df = plasmid_gene_matrix.loc[plasmid_gene_matrix[gene] == 1]

        cluster_compartments = gene_df.groupby(cluster_col)[group_col].apply(set)

        gene_counts = gene_df.groupby(cluster_col).size()

        cluster_compartment_dict[gene] = cluster_compartments
        cluster_gene_counts[gene] = gene_counts

    # Step 4: compute spillover summaries
    gene_spillover_summary = []
    cluster_spillover_records = []

    for gene, cluster_dict in cluster_compartment_dict.items():
        n_clusters = len(cluster_dict)
        n_multi_compartment = sum(len(comps) > 1 for comps in cluster_dict)
        multi_origin_clusters_with_gene = [
            cluster
            for cluster in cluster_dict.keys()
            if len(cluster_all_compartments.get(cluster, set())) > 1
        ]
        n_multi_origin_clusters_with_gene = len(multi_origin_clusters_with_gene)

        # number of plasmids with this gene that belong to multi-origin clusters
        n_plasmids_in_multi_origin_clusters = sum(
            cluster_gene_counts[gene].get(cluster, 0)
            for cluster in multi_origin_clusters_with_gene
        )

        total_gene_count = plasmid_gene_matrix[gene].sum()

        gene_spillover_summary.append(
            {
                "gene": gene,
                "n_clusters": n_clusters,
                "n_clusters_with_gene_spillover": n_multi_compartment,
                "n_multi_origin_clusters_with_gene": n_multi_origin_clusters_with_gene,
                "spillover_fraction": (
                    n_multi_compartment / n_clusters if n_clusters > 0 else 0.0
                ),
                "total_plasmids_with_gene": int(total_gene_count),
                "n_plasmids_with_gene_in_multi_origin_clusters": int(
                    n_plasmids_in_multi_origin_clusters
                ),
            }
        )

        for cluster, gene_comps in cluster_dict.items():
            cluster_spillover_records.append(
                {
                    "gene": gene,
                    "cluster": cluster,
                    "gene_compartments": gene_comps,
                    "cluster_all_compartments": cluster_all_compartments.get(
                        cluster, set()
                    ),
                    "gene_count_in_cluster": int(
                        cluster_gene_counts[gene].get(cluster, 0)
                    ),
                    "spillover": len(gene_comps) > 1,
                }
            )

    gene_spillover_summary = pd.DataFrame(gene_spillover_summary).sort_values(
        "spillover_fraction", ascending=False
    )

    plot.spillover_summary(gene_spillover_summary)
    cluster_spillover = pd.DataFrame(cluster_spillover_records)

    return gene_spillover_summary, cluster_spillover


# ---------------------------------------------------------
# 3.1 Dataset Overview
# ---------------------------------------------------------
def dataset_overview(df_in: pd.DataFrame):
    df = df_in.copy()
    out_path = "results/dataset_overview"
    os.makedirs(out_path, exist_ok=True)
    gene_columns = [
        config.AMR_COL,
        config.VIR_COL,
        config.METAL_COL,
        config.BIOCIDE_COL,
    ]

    # ---------------------------------------------------------
    # 3.1.1 Simple counts
    # ---------------------------------------------------------
    # Iso count
    n_isolates = df[config.PARENT_COL].nunique()
    n_people = df[config.ID_COL].nunique()
    n_plasmids = df[config.PLASMID_COL].nunique()

    # Temporal
    start_date = df[config.DATE_COL].min()
    end_date = df[config.DATE_COL].max()

    # Geo
    n_cities = df["city"].nunique()
    n_municipalities = df["municipality"].nunique()
    n_provinces = df["province"].nunique()
    with open(f"{out_path}/output.txt", "w") as f:
        f.write(f"n_isolates = {n_isolates}\n")
        f.write(f"n_patients = {n_people}\n")
        f.write(f"n_plasmids = {n_plasmids}\n")
        f.write(f"\nIsolated between:\n")
        f.write(f"start_date = {start_date}\n")
        f.write(f"end_date = {end_date}\n")
        f.write(f"\nIsolated from:\n")
        f.write(f"n_cities = {n_cities}\n")
        f.write(f"n_municipalities = {n_municipalities}\n")
        f.write(f"n_provinces = {n_provinces}\n")

    # ---------------------------------------------------------
    # 3.1.2 Data distribution
    # ---------------------------------------------------------
    # Epi origin
    origin = count_column_composition(df, "origin", False)
    origin.to_csv(f"{out_path}/origin_distribution.csv", sep=";")

    # Species origin
    species = count_column_composition(df, config.SPECIES_COL, False)
    species.to_csv(f"{out_path}/species_distribution.csv", sep=";")

    # ---------------------------------------------------------
    # 3.1.3 Plasmid carriage statistics
    # ---------------------------------------------------------
    plasmid_carriage_counting(df)
    plot_population(df)

    # ---------------------------------------------------------
    # 3.1.4 Functional gene analysis
    # ---------------------------------------------------------
    # Mobility
    mobility = count_column_composition(df, "mobility", False)
    mobility.to_csv(f"{out_path}/mobility_distribution.csv", sep=";")

    # Replicon
    replicon_full = count_column_composition(df, "replicon", False)
    replicon_full.to_csv(f"{out_path}/replicon_full_distribution.csv", sep=";")
    replicon = count_column_composition(df, "replicon", True)
    replicon.to_csv(f"{out_path}/replicon_distribution.csv", sep=";")

    # Functional genes
    for col in gene_columns:
        col_presence = count_column_composition(df, f"{col}_plasmid", False)
        col_presence.to_csv(f"{out_path}/{col}_presence_distribution.csv", sep=";")
        colgenes = count_column_composition(df, f"{col}", True)
        colgenes.to_csv(f"{out_path}/{col}_gene_distribution.csv", sep=";")

    gene_statics = gene_origin_enrichment(df)
    gene_statics.to_csv(f"results/tables/tableS3_gene_distribution.csv", sep=";")


# ---------------------------------------------------------
# 3.2 Plasmid Cluster Composition
# ---------------------------------------------------------
def cluster_overview(df_in: pd.DataFrame):
    df = df_in.copy()
    out_path = "results/cluster_composition"
    os.makedirs(out_path, exist_ok=True)
    cluster_col = str(config.CLUSTER_COL)
    gene_columns = [
        config.AMR_COL,
        config.VIR_COL,
        config.METAL_COL,
        config.BIOCIDE_COL,
    ]

    # ---------------------------------------------------------
    # 3.2.1 Clustering stats
    # ---------------------------------------------------------
    df["clustered"] = (
        df[cluster_col].map({"-1": "Unclustered", "-": "Untyped"}).fillna("Clustered")
    )
    clustered = count_column_composition(df, "clustered", False)
    clustered.to_csv(f"{out_path}/clustered_rate.csv", sep=";")
    clustered = count_column_composition(df, "clustered", False)
    clustered.to_csv(f"{out_path}/clustered_rate.csv", sep=";")

    df_clustered = df.loc[df["clustered"] == "Clustered"]
    sizes = df_clustered.groupby("group").size()

    q1 = sizes.quantile(0.25)
    q3 = sizes.quantile(0.75)
    with open(f"{out_path}/cluster_stats.txt", "w") as f:
        f.write(f"Number of groups: {sizes.size}\n")
        f.write(f"Mean group size:  {sizes.mean():.2f}\n")
        f.write(f"Q1:               {q1:.2f}\n")
        f.write(f"Q3:               {q3:.2f}\n")
        f.write(f"IQR:              {q3 - q1:.2f}\n")
        f.write(f"Std:              {sizes.std():.2f}\n")
        f.write(f"Min:              {sizes.min()}\n")
        f.write(f"Max:              {sizes.max()}\n")
        f.write("\n")

    plot.tsne_by_cluster(df)

    # ---------------------------------------------------------
    # 3.2.2 Clustering compositions
    # ---------------------------------------------------------
    for col in ["origin", "replicon", "mobility", "AMR_plasmid"]:
        table = count_column_composition_by_cluster(
            df_in=df,
            column_col="country",
            split=True,
        )
        table.to_csv(f"{out_path}/{col}_cluster_counts.csv", sep=";")

    plot.composition_by_cluster(df)

    # ---------------------------------------------------------
    # 3.2.3 Plasmidome differences
    # ---------------------------------------------------------
    composition_dict = compartment_plasmidome_dispersion(df_clustered)
    with open(f"{out_path}/cluster_stats.txt", "a") as f:
        f.write(f"Cluster composition output:\n")
        f.write(f"global_permanova:    {composition_dict["global_permanova"]}\n")
        f.write(f"pairwise_permanova:  {composition_dict["pairwise_permanova"]}\n")
        f.write(f"dispersion_summary:  {composition_dict["dispersion_summary"]}\n")
        f.write(f"dispersion_pairwise: {composition_dict["dispersion_pairwise"]}\n")
        f.write("\n")

    # ---------------------------------------------------------
    # 3.2.4 Cluster functional gene composition
    # ---------------------------------------------------------
    tables = []

    for col in gene_columns:
        table = count_column_composition_by_cluster(
            df_in=df,
            column_col=col,
            split=True,
        )

        table["variable"] = col
        tables.append(table)

    result = pd.concat(tables)
    result.to_csv(f"{out_path}/cluster_counts.csv", sep=";")
    plot.gene_heatmap(result, cluster_col)

    # ---------------------------------------------------------
    # 3.2.5 Cluster functional gene spillover
    # ---------------------------------------------------------
    gene_spillover_summary, cluster_spillover = gene_spillover_analysis(
        df_clustered, cluster_col
    )
    gene_spillover_summary.to_csv(f"{out_path}/gene_spillover.csv", sep=";")
    cluster_spillover.to_csv(f"{out_path}/cluster_spillover.csv", sep=";")


if __name__ == "__main__":
    metadata_df = pd.read_csv("data/metadata.csv", encoding="ISO-8859-1", sep=";")
    clustering_df = pd.read_csv("data/clustering.csv")
    plasmid_df = clean_plasmid_df(metadata_df, clustering_df)
    dataset_overview(plasmid_df)
    cluster_overview(plasmid_df)
