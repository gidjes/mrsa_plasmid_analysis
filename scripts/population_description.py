import pandas as pd
import numpy as np
import os
import geopandas as gp
from itertools import combinations

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from scipy.stats import mannwhitneyu, chi2_contingency
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import (
    DistanceMatrix,
    permanova,
    permdisp,
)
from skbio.stats.ordination import pcoa

import config
import plotting_functions as plot

# ---------------------------------------------------------
# 3.0.0 Configuration
# ---------------------------------------------------------
CLUSTER_COL = config.CLUSTER_COL


# ---------------------------------------------------------
# 3.1.0 Helper Functions
# ---------------------------------------------------------
def plot_population(df_plasmids_in: pd.DataFrame, df_isolates_in: pd.DataFrame):
    """
    Helper function to plot the output from the population
    statistics. Loads and readies map files/geojson

    Key outputs
    ----------
    Figure 1 - isolate data

    Parameters
    ----------
    df_plasmids_in : pd.DataFrame
        Dataframe containing plasmid metadata
    df_isolates_in : pd.DataFrame
        Dataframe containing isolate metadata
    """
    parent_df = df_isolates_in[
        ["KEY", "city", "municipality", "province"]
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
    plot.isolate_data(df_plasmids_in, df_isolates_in, municiple_map, map_boxes)


def save_glm_tables(
    model,
    dispersion,
    name,
    reference_categories,
    term_labels=None,
    correction_method="fdr_bh",
    correction_by_variable=True,
):
    """
    Save publication-ready GLM results as two CSV files.

    Table A:
        Model-level information.

    Table B:
        Exponentiated coefficients (IRRs) with 95% CIs.
        Intercept is excluded.
        Reference categories are explicitly included.
        Multiple-testing adjusted P values are included.

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
        {"origin": "CA-MRSA"}.

    term_labels : dict, optional
        Dictionary to make coefficient names more readable.

    correction_method : str, optional
        Method passed to statsmodels.stats.multitest.multipletests.
        Default is Benjamini-Hochberg FDR ("fdr_bh").

    correction_by_variable : bool, optional
        If True, perform multiple-testing correction separately for
        each categorical variable (e.g. all ST comparisons together
        and all origin comparisons together). If False, all model
        coefficients are corrected together.
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

    # Save table A: Model descriptives
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
    # Multiple-testing correction
    # ---------------------------------------------------------
    results["P_value_adjusted"] = np.nan

    if correction_by_variable:

        # Identify the original model term before cleaning it
        # so that coefficients can be grouped by categorical
        # variable.
        def get_variable(term):
            if term.startswith("C(origin)"):
                return "origin"
            elif term.startswith("C(ST_collapsed)"):
                return "ST_collapsed"
            else:
                return "other"

        results["_correction_group"] = results["Term"].apply(get_variable)

        for group, idx in results.groupby("_correction_group").groups.items():

            if group == "other":
                continue

            pvalues = results.loc[idx, "P_value"].values

            _, p_adjusted, _, _ = multipletests(
                pvalues,
                method=correction_method,
            )

            results.loc[idx, "P_value_adjusted"] = p_adjusted

        # Any coefficients not belonging to a recognised
        # categorical variable are corrected as one group.
        other_idx = results.index[results["_correction_group"] == "other"]

        if len(other_idx) > 0:
            pvalues = results.loc[other_idx, "P_value"].values

            _, p_adjusted, _, _ = multipletests(
                pvalues,
                method=correction_method,
            )

            results.loc[other_idx, "P_value_adjusted"] = p_adjusted

        results = results.drop(columns="_correction_group")

    else:
        pvalues = results["P_value"].values

        _, p_adjusted, _, _ = multipletests(
            pvalues,
            method=correction_method,
        )

        results["P_value_adjusted"] = p_adjusted

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

    # =========================================================
    # ADD REFERENCE CATEGORIES
    # =========================================================
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
                "P_value_adjusted": np.nan,
                "CI_lower": np.nan,
                "CI_upper": np.nan,
            }
        )

    reference_df = pd.DataFrame(reference_rows)

    results = pd.concat(
        [reference_df, results],
        ignore_index=True,
    )

    # =========================================================
    # FORMAT VALUES
    # =========================================================
    results["IRR"] = results["IRR"].round(3)
    results["SE"] = results["SE"].round(3)
    results["z"] = results["z"].round(3)
    results["CI_lower"] = results["CI_lower"].round(3)
    results["CI_upper"] = results["CI_upper"].round(3)

    results["P_value"] = results["P_value"].apply(
        lambda x: ("" if pd.isna(x) else "<0.001" if x < 0.001 else f"{x:.3f}")
    )

    results["P_value_adjusted"] = results["P_value_adjusted"].apply(
        lambda x: ("" if pd.isna(x) else "<0.001" if x < 0.001 else f"{x:.3f}")
    )

    # More explicit column names for supplementary material
    results = results.rename(
        columns={
            "CI_lower": "IRR_95CI_lower",
            "CI_upper": "IRR_95CI_upper",
            "P_value_adjusted": "P_value_FDR",
        }
    )

    # =========================================================
    # SAVE TABLE B
    # =========================================================
    results.to_csv(
        f"results/tables/{name}_B.csv",
        sep=";",
        index=False,
    )

    return model_info, results


def save_plasmidome_tables(
    results,
    output_dir="results/tables",
    table_permanova="S5",
    table_dispersion="S6",
):
    """
    Save publication-ready plasmidome PERMANOVA and dispersion
    results as four supplementary CSV tables.

    Output structure
    ----------------
    Table S5: PERMANOVA
        S5A = Global PERMANOVA
        S5B = Pairwise PERMANOVA

    Table S6: Dispersion
        S6A = Global PERMDISP + group-level dispersion
        S6B = Pairwise dispersion comparisons

    Parameters
    ----------
    results : dict
        Output dictionary returned by
        `compartment_plasmidome_dispersion()`.

    output_dir : str, default="results/tables"
        Directory where CSV files will be saved.

    table_permanova : str, default="S5"
        Supplementary table number assigned to PERMANOVA.

    table_dispersion : str, default="S6"
        Supplementary table number assigned to dispersion analysis.

    Returns
    -------
    dict
        Dictionary containing the four output DataFrames:

        {
            "S5A": global_permanova,
            "S5B": pairwise_permanova,
            "S6A": global_dispersion,
            "S6B": pairwise_dispersion,
        }
    """

    import os
    import numpy as np
    import pandas as pd

    # =========================================================
    # Create output directory
    # =========================================================

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # Helper functions
    # =========================================================

    def format_pvalue(p):
        """
        Publication-style p-value formatting.
        """
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def format_numeric(df, columns, decimals=3):
        """
        Round selected numeric columns if present.
        """
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").round(decimals)
        return df

    # =========================================================
    # Extract results
    # =========================================================

    dm = results["distance_matrix"]
    metadata = results["metadata"]
    isolate_cluster_matrix = results["isolate_cluster_matrix"]

    global_permanova = results["global_permanova"]
    global_permanova_r2 = results["global_permanova_R2"]

    pairwise_permanova = results["pairwise_permanova"].copy()

    permdisp = results["permdisp"]

    dispersion_summary = results["dispersion_summary"].copy()

    dispersion_pairwise = results["dispersion_pairwise"].copy()

    # =========================================================
    # TABLE S5A — GLOBAL PERMANOVA
    # =========================================================

    n_isolates = len(metadata)
    n_compartments = metadata["origin"].nunique()
    n_clusters = isolate_cluster_matrix.shape[1]

    permanova_F = float(global_permanova["test statistic"])

    permanova_p = float(global_permanova["p-value"])

    permanova_permutations = int(global_permanova["number of permutations"])

    table_S5A = pd.DataFrame(
        {
            "Statistic": [
                "No. isolates",
                "No. compartments",
                "No. plasmid backbone clusters",
                "Distance metric",
                "Data representation",
                "Pseudo-F",
                "R2",
                "P_value",
                "Permutations",
            ],
            "Value": [
                n_isolates,
                n_compartments,
                n_clusters,
                "Jaccard",
                "Plasmid backbone-cluster presence/absence",
                f"{permanova_F:.3f}",
                f"{global_permanova_r2:.3f}",
                format_pvalue(permanova_p),
                permanova_permutations,
            ],
        }
    )

    table_S5A = format_numeric(
        table_S5A,
        columns=[
            "Value",
        ],
    )

    # Restore integer/string values that should not be represented
    # as decimal numbers after generic rounding.
    table_S5A.loc[
        table_S5A["Statistic"] == "No. isolates",
        "Value",
    ] = n_isolates

    table_S5A.loc[
        table_S5A["Statistic"] == "No. compartments",
        "Value",
    ] = n_compartments

    table_S5A.loc[
        table_S5A["Statistic"] == "No. plasmid backbone clusters",
        "Value",
    ] = n_clusters

    table_S5A.loc[
        table_S5A["Statistic"] == "Permutations",
        "Value",
    ] = permanova_permutations

    table_S5A.to_csv(
        os.path.join(
            output_dir,
            f"table{table_permanova}_PERMANOVA_A.csv",
        ),
        sep=";",
        index=False,
    )

    # =========================================================
    # =========================================================
    # TABLE S5B — PAIRWISE PERMANOVA
    # =========================================================
    # =========================================================

    # Support both the new column names and the names from
    # earlier versions of the function.
    pairwise_permanova = pairwise_permanova.rename(
        columns={
            "group1": "Group_1",
            "group2": "Group_2",
            "p": "P_value",
            "p_adj": "P_value_BH",
        }
    )

    table_S5B = pairwise_permanova[
        [
            "Group_1",
            "Group_2",
            "F",
            "R2",
            "P_value",
            "P_value_BH",
            "n1",
            "n2",
        ]
    ].copy()

    table_S5B = table_S5B.rename(
        columns={
            "F": "Pseudo_F",
        }
    )

    table_S5B = format_numeric(
        table_S5B,
        columns=[
            "Pseudo_F",
            "R2",
        ],
    )

    table_S5B["P_value"] = table_S5B["P_value"].astype(float).apply(format_pvalue)

    table_S5B["P_value_BH"] = table_S5B["P_value_BH"].astype(float).apply(format_pvalue)

    table_S5B.to_csv(
        os.path.join(
            output_dir,
            f"table{table_permanova}_PERMANOVA_B.csv",
        ),
        sep=";",
        index=False,
    )

    # =========================================================
    # TABLE S6A — GLOBAL PERMDISP + GROUP DISPERSION
    # =========================================================
    permdisp_F = float(permdisp["test statistic"])

    permdisp_p = float(permdisp["p-value"])

    permdisp_permutations = int(permdisp["number of permutations"])

    # ---------------------------------------------------------
    # Global PERMDISP
    # ---------------------------------------------------------

    global_dispersion = pd.DataFrame(
        {
            "Statistic": [
                "Test",
                "Center",
                "No. isolates",
                "No. compartments",
                "F",
                "P_value",
                "Permutations",
            ],
            "Value": [
                "PERMDISP",
                "Centroid",
                str(n_isolates),
                str(n_compartments),
                f"{permdisp_F:.3f}",
                format_pvalue(permdisp_p),
                str(permdisp_permutations),
            ],
        }
    )

    # ---------------------------------------------------------
    # Group-level dispersion
    # ---------------------------------------------------------

    group_dispersion = dispersion_summary.reset_index().rename(
        columns={
            "origin": "Origin",
            "median": "Median_distance_to_centroid",
            "Q1": "Q1_distance_to_centroid",
            "Q3": "Q3_distance_to_centroid",
            "mean": "Mean_distance_to_centroid",
            "SD": "SD_distance_to_centroid",
            "n": "n",
        }
    )

    for column in [
        "Median_distance_to_centroid",
        "Q1_distance_to_centroid",
        "Q3_distance_to_centroid",
        "Mean_distance_to_centroid",
        "SD_distance_to_centroid",
    ]:
        if column in group_dispersion.columns:
            group_dispersion[column] = group_dispersion[column].astype(float).round(3)

    # ---------------------------------------------------------
    # Combine global test + group summary into one CSV.
    #
    # The first section contains the omnibus test.
    # The second section contains group-level dispersion.
    # ---------------------------------------------------------

    global_dispersion_path = os.path.join(
        output_dir,
        f"table{table_dispersion}_dispersion_A.csv",
    )

    with open(
        global_dispersion_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write("Global PERMDISP\n")
        global_dispersion.to_csv(
            f,
            sep=";",
            index=False,
        )

        f.write("\n")
        f.write("Dispersion by compartment\n")
        group_dispersion.to_csv(
            f,
            sep=";",
            index=False,
        )

    # =========================================================
    # =========================================================
    # TABLE S6B — PAIRWISE DISPERSION
    # =========================================================
    # =========================================================

    dispersion_pairwise = dispersion_pairwise.rename(
        columns={
            "group1": "Group_1",
            "group2": "Group_2",
            "p": "P_value",
            "p_adj": "P_value_BH",
        }
    )

    table_S6B = dispersion_pairwise[
        [
            "Group_1",
            "Group_2",
            "Mann_Whitney_U",
            "P_value",
            "P_value_BH",
            "n1",
            "n2",
        ]
    ].copy()

    table_S6B["Mann_Whitney_U"] = (
        pd.to_numeric(
            table_S6B["Mann_Whitney_U"],
            errors="coerce",
        )
        .round(0)
        .astype("Int64")
    )

    table_S6B["P_value"] = table_S6B["P_value"].astype(float).apply(format_pvalue)

    table_S6B["P_value_BH"] = table_S6B["P_value_BH"].astype(float).apply(format_pvalue)

    table_S6B.to_csv(
        os.path.join(
            output_dir,
            f"table{table_dispersion}_dispersion_B.csv",
        ),
        sep=";",
        index=False,
    )

    # =========================================================
    # Return tables
    # =========================================================

    return {
        "S5A": table_S5A,
        "S5B": table_S5B,
        "S6A_global": global_dispersion,
        "S6A_group_summary": group_dispersion,
        "S6B": table_S6B,
    }


def plasmid_carriage_counting(df_in: pd.DataFrame, df_isolates_in: pd.DataFrame):
    """
    Descriptive statistics on the number of plasmids carried by
    isolates

    Parameters
    ----------
    df_in : pd.DataFrame
        dataframe with plasmid metadata

    Key outputs
    ----------
    Figure 2 - Plasmid carriage forest plot
    Table S1 - Poisson model Plasmids_n ~ origin
    Table S2 - Poisson model Plasmids_n ~ ST
    Table S3 - Poisson model Plasmids_n ~ ST + origin
    """
    df = df_in.copy()

    # Create isolate-level data
    isolate_df = (
        df_isolates_in[
            [
                "KEY",
                "ISOLATE_TL_MLST_ST",
                "origin",
            ]
        ]
        .rename(columns={"KEY": "Parent"})
        .drop_duplicates()
    )

    # Determine significant/common STs based on the number of isolates
    isolate_st_counts = isolate_df["ISOLATE_TL_MLST_ST"].value_counts()
    common_STs = isolate_st_counts[isolate_st_counts >= 25].index
    isolate_df["ST_collapsed"] = isolate_df["ISOLATE_TL_MLST_ST"].where(
        isolate_df["ISOLATE_TL_MLST_ST"].isin(common_STs),
        "rare_ST",
    )

    # Count plasmids per isolate
    plasmid_counts = df.groupby("Parent").size().reset_index(name="plasmid_count")

    # Combine plasmid count with isolate metadata
    analysis_df = isolate_df.merge(
        plasmid_counts,
        on="Parent",
        how="left",
    ).fillna(0)

    # Sanity check
    print(analysis_df)
    analysis_df["plasmid_count"].describe().to_csv(
        "results/dataset_overview/carriage_rates.csv", sep=";"
    )
    analysis_df.groupby("origin")["plasmid_count"].describe().to_csv(
        "results/dataset_overview/origin_carriage_rates.csv", sep=";"
    )

    # Save counts
    analysis_df.to_csv(
        "results/plasmid_counts_per_isolate.csv",
        sep=";",
        index=False,
    )

    # Model 1: plasmid count ~ origin
    model_origin = smf.glm(
        "plasmid_count ~ C(origin)",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit()

    pearson_dispersion = sum(model_origin.resid_pearson**2) / model_origin.df_resid
    save_glm_tables(
        model_origin,
        pearson_dispersion,
        "tableS1_plasmids_by_origin",
        reference_categories={
            "origin": "CA-MRSA",
        },
    )

    # Model 2: plasmid count ~ ST
    model_ST = smf.glm(
        "plasmid_count ~ C(ST_collapsed)",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit()

    pearson_dispersion = sum(model_ST.resid_pearson**2) / model_ST.df_resid
    save_glm_tables(
        model_ST,
        pearson_dispersion,
        "tableS2_plasmids_by_ST",
        reference_categories={
            "ST_collapsed": "1",
        },
    )

    # Model 3: plasmid count ~ ST + origin
    model_both = smf.glm(
        "plasmid_count ~ C(ST_collapsed) + C(origin)",
        data=analysis_df,
        family=sm.families.Poisson(),
    ).fit()

    pearson_dispersion = sum(model_both.resid_pearson**2) / model_both.df_resid
    save_glm_tables(
        model_both,
        pearson_dispersion,
        "tableS3_plasmids_by_both",
        reference_categories={
            "origin": "CA-MRSA",
            "ST_collapsed": "1",
        },
    )

    # Plot the model outputs
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


def count_column_composition(
    df_in: pd.DataFrame, col: str, split: bool = False
) -> pd.DataFrame:
    """
    Count the values and percentage of categorical values in a dataframe
    column. Converts to strings to create directly copyable csv.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing values
    col : str
        Column to count
    split : bool, optional
        Should the column values be split on ',', by default False

    Returns
    -------
    pd.DataFrame
        Table with counts and percentages
    """
    df = df_in.copy()

    # If no split needed
    if not split:
        # Count
        counts = df[col].value_counts()

        # Calculate fractions
        percentages = (
            df[col].value_counts(normalize=True).mul(100).round(1).astype(str) + "%"
        )
    else:
        # Split
        s = df[col].dropna().astype(str).str.split(",").explode().str.strip()

        # Count
        counts = s.value_counts()

        # Calculate fractions
        percentages = s.value_counts(normalize=True).mul(100).round(1).astype(str) + "%"

    # Put counts and % together
    col_values = pd.concat([counts, percentages], axis=1)
    col_values.columns = ["count", "percentage"]
    return col_values


def test_gene_origin_association(
    gene: str,
    plasmid_gene_binary: pd.DataFrame,
    origin_col: str = "origin",
    min_positive: int = 5,
    n_permutations: int = 20000,
    random_state: int = 0,
):
    """
    Test whether a gene's presence/absence is associated with plasmid
    epidemiological origin/compartment.

    Statistics reported
    --------------------
    - Pearson chi-square statistic
    - Asymptotic chi-square p-value
    - Monte-Carlo permutation p-value
    - Cramer's V as effect size
    - Expected-count diagnostic
    - Observed gene-positive/gene-negative counts per origin
    - Gene prevalence per origin
    - Pearson standardized residuals

    The permutation p-value is the preferred inferential statistic when
    expected cell counts are sparse.

    Returns
    -------
    dict
        Flat dictionary suitable for conversion to a DataFrame.
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Get gene and origin data
    # ------------------------------------------------------------------
    gene_status = plasmid_gene_binary[gene]
    origin = plasmid_gene_binary[origin_col]

    # Remove missing values
    valid = gene_status.notna() & origin.notna()

    gene_status = gene_status.loc[valid].astype(int)
    origin = origin.loc[valid]

    # ------------------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------------------
    n_pos = int((gene_status == 1).sum())
    n_neg = int((gene_status == 0).sum())

    if n_pos < min_positive:
        return {
            "gene": gene,
            "tested": False,
            "n_gene_positive": n_pos,
            "n_gene_negative": n_neg,
            "reason": "Too few gene-positive plasmids",
        }

    # ------------------------------------------------------------------
    # Contingency table
    # ------------------------------------------------------------------
    contingency = pd.crosstab(
        gene_status,
        origin,
    )

    # Ensure both gene-negative and gene-positive rows exist
    contingency = contingency.reindex(
        index=[0, 1],
        fill_value=0,
    )

    contingency.index = [
        "gene_negative",
        "gene_positive",
    ]

    # Remove pandas axis name ("origin")
    contingency.columns.name = None

    # ------------------------------------------------------------------
    # Pearson chi-square test
    # ------------------------------------------------------------------
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)

    # Proportion of expected cells < 5
    low_expected_counts = bool((expected < 5).mean() > 0.20)

    # ------------------------------------------------------------------
    # Monte-Carlo permutation p-value
    #
    # Origin labels are shuffled while preserving the observed
    # number of plasmids in each origin.
    # ------------------------------------------------------------------
    origin_values = origin.to_numpy()
    gene_values = gene_status.to_numpy()

    perm_chi2 = np.empty(n_permutations)

    for i in range(n_permutations):

        shuffled_origin = rng.permutation(origin_values)

        perm_table = pd.crosstab(
            gene_values,
            shuffled_origin,
        )

        # Ensure exactly the same rows/columns as observed table
        perm_table = perm_table.reindex(
            index=[0, 1],
            columns=contingency.columns,
            fill_value=0,
        )

        perm_chi2[i], _, _, _ = chi2_contingency(perm_table)

    # +1 correction prevents p = 0
    p_chi2_perm = (np.sum(perm_chi2 >= chi2) + 1) / (n_permutations + 1)

    # ------------------------------------------------------------------
    # Cramer's V
    # ------------------------------------------------------------------
    n = contingency.to_numpy().sum()

    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    # ------------------------------------------------------------------
    # Standardized Pearson residuals
    # ------------------------------------------------------------------
    residuals = (contingency.to_numpy() - expected) / np.sqrt(expected)

    residuals = pd.DataFrame(
        residuals,
        index=contingency.index,
        columns=contingency.columns,
    )

    # ------------------------------------------------------------------
    # Start flat result dictionary
    # ------------------------------------------------------------------
    result = {
        "gene": gene,
        "tested": True,
        "n_gene_positive": n_pos,
        "n_gene_negative": n_neg,
        "chi2": chi2,
        "dof": dof,
        "p_chi2": p_chi2,
        "p_chi2_perm": p_chi2_perm,
        "low_expected_counts": low_expected_counts,
        "cramers_v": cramers_v,
    }

    # ------------------------------------------------------------------
    # Add origin-specific counts, prevalence and residuals
    # ------------------------------------------------------------------
    for compartment in contingency.columns:

        n_negative = int(
            contingency.loc[
                "gene_negative",
                compartment,
            ]
        )

        n_positive = int(
            contingency.loc[
                "gene_positive",
                compartment,
            ]
        )

        n_total = n_negative + n_positive

        # Observed counts
        result[f"neg_{compartment}"] = n_negative

        result[f"pos_{compartment}"] = n_positive

        # Prevalence of the gene within the origin
        if n_total > 0:
            result[f"prev_{compartment}"] = n_positive / n_total
        else:
            result[f"prev_{compartment}"] = np.nan

        # Standardized residuals
        result[f"resid_neg_{compartment}"] = residuals.loc[
            "gene_negative",
            compartment,
        ]

        result[f"resid_pos_{compartment}"] = residuals.loc[
            "gene_positive",
            compartment,
        ]

    return result


def gene_origin_enrichment(
    df_in: pd.DataFrame,
    min_positive: int = 5,
    n_permutations: int = 20000,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Test gene presence/absence against plasmid epidemiological origin.

    Returns
    -------
    pd.DataFrame
        Flat, one-row-per-gene results table suitable for supplementary
        material and export to CSV/Excel.
    """

    df = df_in.copy()

    # ------------------------------------------------------------------
    # Gene-list columns
    # ------------------------------------------------------------------
    gene_columns = [
        config.AMR_COL,
        config.VIR_COL,
        config.METAL_COL,
        config.BIOCIDE_COL,
    ]

    # ------------------------------------------------------------------
    # Convert gene lists to long format
    # ------------------------------------------------------------------
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

    # Remove empty gene names
    long_genes = long_genes[long_genes["gene"].ne("")]

    gene_names = sorted(long_genes["gene"].unique())

    print(f"{len(gene_names)} unique genes")

    # ------------------------------------------------------------------
    # Plasmid × gene binary matrix
    # ------------------------------------------------------------------
    plasmid_gene_matrix = long_genes.assign(present=1).pivot_table(
        index="Plasmid",
        columns="gene",
        values="present",
        aggfunc="max",
        fill_value=0,
    )

    all_plasmids = df["Plasmid"].unique()

    plasmid_gene_matrix = plasmid_gene_matrix.reindex(
        all_plasmids,
        fill_value=0,
    )

    # ------------------------------------------------------------------
    # Add origin metadata
    # ------------------------------------------------------------------
    metadata = df[["Plasmid", "origin"]].drop_duplicates().set_index("Plasmid")

    plasmid_gene_binary = plasmid_gene_matrix.join(metadata)

    # ------------------------------------------------------------------
    # Test every gene
    # ------------------------------------------------------------------
    results = []

    for gene in gene_names:

        result = test_gene_origin_association(
            gene=gene,
            plasmid_gene_binary=plasmid_gene_binary,
            origin_col="origin",
            min_positive=min_positive,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        results.append(result)

    # ------------------------------------------------------------------
    # Create flat results DataFrame
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    # Keep only tested genes
    results_df = results_df[results_df["tested"]].copy()

    # ------------------------------------------------------------------
    # Multiple-testing correction across genes
    # ------------------------------------------------------------------
    results_df["p_chi2_adj"] = multipletests(
        results_df["p_chi2"],
        method="fdr_bh",
    )[1]

    results_df["p_chi2_perm_adj"] = multipletests(
        results_df["p_chi2_perm"],
        method="fdr_bh",
    )[1]

    # ------------------------------------------------------------------
    # Sort by primary permutation p-value
    # ------------------------------------------------------------------
    results_df = results_df.sort_values("p_chi2_perm").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Round numerical values for supplementary table
    # ------------------------------------------------------------------
    numeric_cols = results_df.select_dtypes(include="number").columns

    results_df[numeric_cols] = results_df[numeric_cols].round(4)

    # ------------------------------------------------------------------
    # Primary results for console display
    # ------------------------------------------------------------------
    primary_cols = [
        "gene",
        "n_gene_positive",
        "n_gene_negative",
        "chi2",
        "dof",
        "p_chi2_perm",
        "p_chi2_perm_adj",
        "cramers_v",
        "low_expected_counts",
    ]

    print(results_df[primary_cols].to_string(index=False))

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
    """
    Count the values and percentage of categorical values in a dataframe
    column by another column value, default clusters. Percentages represent
    presence/absence fractions for each row. Converts to strings to create
    directly copyable csv.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame with column to count
    column_col : str
        Column name to count each occurence of
    row_col : str, optional
        Column name to count each occurence by, by default config.CLUSTER_COL
    split : bool, optional
        Does the column_col input need to be split on ',', by default False

    Returns
    -------
    pd.DataFrame
        Table with counts and percentages
    """
    df = df_in.copy()

    if not split:
        # Count
        counts = pd.crosstab(
            df[row_col],
            df[column_col],
        )

        # Calculate percentages
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

        # # Remove duplicate category combinations within a row
        # split_df = split_df.drop_duplicates()

        counts = pd.crosstab(
            split_df[row_col],
            split_df[column_col],
        )

        # Denominator = total number of original rows in each row category
        row_totals = df.groupby(row_col).size()
        percentages = counts.div(row_totals, axis=0).mul(100).round(1)

    return counts.astype(str) + " (" + percentages.astype(str) + "%)"


def build_plasmidome_distance(
    df: pd.DataFrame,
    isolate_col: str = "Parent",
    origin_col: str = "origin",
):
    """
    Build a Jaccard distance matrix between isolates based on
    plasmid backbone-cluster presence/absence ("plasmidome").

    Each isolate is represented as a binary vector indicating whether
    each plasmid backbone cluster is present.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing isolate IDs, plasmid backbone clusters,
        and epidemiological origin.
    isolate_col : str, default="Parent"
        Column containing isolate IDs.
    origin_col : str, default="origin"
        Column containing epidemiological origin.

    Returns
    -------
    dm : skbio.DistanceMatrix
        Isolate-by-isolate Jaccard distance matrix.
    metadata : pd.DataFrame
        Metadata indexed by isolate ID.
    isolate_cluster_matrix : pd.DataFrame
        Binary isolate-by-cluster presence/absence matrix.
    """
    cluster_col = str(config.CLUSTER_COL)

    # One observation per isolate × cluster.
    df_unique = df[[isolate_col, cluster_col]].drop_duplicates()

    # Binary plasmidome matrix.
    isolate_cluster_matrix = (
        pd.crosstab(
            df_unique[isolate_col],
            df_unique[cluster_col],
        )
        .astype(bool)
        .astype(int)
    )

    # Metadata corresponding exactly to isolates in the distance matrix.
    metadata = (
        df[[isolate_col, origin_col]]
        .drop_duplicates()
        .set_index(isolate_col)
        .loc[isolate_cluster_matrix.index]
    )

    # Check that every isolate has exactly one origin.
    origin_counts = (
        df[[isolate_col, origin_col]].drop_duplicates().groupby(isolate_col).size()
    )

    if not (origin_counts == 1).all():
        bad = origin_counts[origin_counts != 1]
        raise ValueError(
            "Some isolates are associated with multiple origins:\n" f"{bad}"
        )

    # Jaccard distance on binary plasmidome profiles.
    jaccard_distances = pdist(
        isolate_cluster_matrix.values,
        metric="jaccard",
    )

    dm = DistanceMatrix(
        squareform(jaccard_distances),
        ids=isolate_cluster_matrix.index.tolist(),
    )

    return dm, metadata, isolate_cluster_matrix


def permanova_r2(
    F: float,
    n: int,
    n_groups: int,
) -> float:
    """
    Calculate PERMANOVA R² from the pseudo-F statistic.

    R² represents the proportion of variation in the distance matrix
    attributable to the grouping variable.

    Formula:
        R² = 1 / [1 + ((n - g) / ((g - 1) * F))]

    Parameters
    ----------
    F : float
        PERMANOVA pseudo-F statistic.
    n : int
        Number of observations.
    n_groups : int
        Number of groups.

    Returns
    -------
    float
        PERMANOVA R².
    """
    if F <= 0:
        return 0.0

    return 1.0 / (1.0 + ((n - n_groups) / ((n_groups - 1) * F)))


def pairwise_permanova(
    dm,
    metadata: pd.DataFrame,
    group_col: str = "origin",
    n_perm: int = 999,
    seed: int = 42,
):
    """
    Pairwise PERMANOVA between all groups.

    P-values are corrected using Benjamini-Hochberg FDR.

    Returns pseudo-F, p-value, BH-adjusted p-value, sample sizes,
    and pairwise PERMANOVA R².
    """
    groups = metadata[group_col].dropna().unique()

    rows = []

    for g1, g2 in combinations(groups, 2):

        ids = metadata.index[metadata[group_col].isin([g1, g2])].tolist()

        dm_sub = dm.filter(ids)
        meta_sub = metadata.loc[ids]

        res = permanova(
            distance_matrix=dm_sub,
            grouping=meta_sub[group_col],
            permutations=n_perm,
            seed=seed,
        )

        F = float(res["test statistic"])
        p = float(res["p-value"])

        n1 = int((meta_sub[group_col] == g1).sum())
        n2 = int((meta_sub[group_col] == g2).sum())

        n_total = n1 + n2

        r2 = permanova_r2(
            F=F,
            n=n_total,
            n_groups=2,
        )

        rows.append(
            {
                "Group_1": g1,
                "Group_2": g2,
                "F": F,
                "P_value": p,
                "n1": n1,
                "n2": n2,
                "R2": r2,
            }
        )

    pairwise_df = pd.DataFrame(rows)

    pairwise_df["P_value_BH"] = multipletests(
        pairwise_df["P_value"],
        method="fdr_bh",
    )[1]

    return pairwise_df.sort_values("P_value_BH").reset_index(drop=True)


def distance_to_centroid(
    coords: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
):
    """
    Calculate Euclidean distance from every isolate to the arithmetic
    centroid of its group in PCoA space.

    IMPORTANT
    ---------
    For scikit-bio 0.7.3, `pcoa(...).samples` already contains the
    eigenvalue-scaled principal coordinates. Do NOT multiply the
    coordinates by sqrt(eigenvalue) again.
    """
    id_name = metadata.index.name or "id"

    merged = coords.join(
        metadata[[group_col]],
        how="inner",
    )

    coordinate_columns = coords.columns

    rows = []

    for group, subdf in merged.groupby(group_col):

        if len(subdf) < 2:
            continue

        group_coords = subdf[coordinate_columns]

        # Arithmetic centroid.
        centroid = group_coords.mean(axis=0)

        # Euclidean distance to centroid.
        distances = np.linalg.norm(
            group_coords.values - centroid.values,
            axis=1,
        )

        rows.extend(
            zip(
                subdf.index,
                [group] * len(distances),
                distances,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            id_name,
            group_col,
            "distance_to_centroid",
        ],
    ).set_index(id_name)


def summarize_dispersion(
    disp_df: pd.DataFrame,
    group_col: str = "origin",
):
    """
    Summarize distance-to-centroid distributions by group.
    """
    summary = (
        disp_df.groupby(group_col)["distance_to_centroid"]
        .agg(
            median="median",
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            mean="mean",
            SD="std",
            n="count",
        )
        .sort_values("median", ascending=False)
    )

    return summary


def pairwise_dispersion(
    disp_df: pd.DataFrame,
    group_col: str = "origin",
):
    """
    Pairwise Mann-Whitney tests on distance-to-centroid values.

    These are post-hoc comparisons of the dispersion distributions,
    with Benjamini-Hochberg FDR correction.

    Note:
        These Mann-Whitney tests are descriptive/post-hoc comparisons.
        The formal omnibus multivariate dispersion test is PERMDISP.
    """
    groups = disp_df[group_col].dropna().unique()

    rows = []

    for g1, g2 in combinations(groups, 2):

        x = disp_df.loc[
            disp_df[group_col] == g1,
            "distance_to_centroid",
        ].values

        y = disp_df.loc[
            disp_df[group_col] == g2,
            "distance_to_centroid",
        ].values

        U, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        rows.append(
            {
                "Group_1": g1,
                "Group_2": g2,
                "Mann_Whitney_U": U,
                "P_value": p,
                "n1": len(x),
                "n2": len(y),
            }
        )

    result = pd.DataFrame(rows)

    result["P_value_BH"] = multipletests(
        result["P_value"],
        method="fdr_bh",
    )[1]

    return result.sort_values("P_value_BH").reset_index(drop=True)


def compartment_plasmidome_dispersion(
    df_in: pd.DataFrame,
    n_perm: int = 999,
    seed: int = 42,
    group_col: str = "origin",
):
    """
    Complete plasmidome composition and dispersion analysis.

    Analysis
    --------
    1. Construct isolate × plasmid-backbone-cluster presence/absence matrix.
    2. Calculate pairwise Jaccard distances.
    3. Global PERMANOVA across all compartments.
    4. Pairwise PERMANOVA with BH correction.
    5. PCoA of the Jaccard distance matrix.
    6. Calculate isolate-to-centroid distances using the already-scaled
       PCoA coordinates from scikit-bio 0.7.3.
    7. Summarize dispersion by compartment.
    8. Formal omnibus PERMDISP test.
    9. Pairwise Mann-Whitney comparisons of dispersion with BH correction.

    Returns
    -------
    dict
        All distance, composition, PCoA, PERMANOVA, and dispersion results.
    """

    df = df_in.copy()

    # =========================================================
    # 1. Jaccard plasmidome distance matrix
    # =========================================================
    dm, metadata, isolate_cluster_matrix = build_plasmidome_distance(
        df,
        # group_col=group_col,
    )

    grouping = metadata[group_col]

    # =========================================================
    # 2. Global PERMANOVA
    # =========================================================
    global_res = permanova(
        distance_matrix=dm,
        grouping=grouping,
        permutations=n_perm,
        seed=seed,
    )

    global_F = float(global_res["test statistic"])

    global_p = float(global_res["p-value"])

    n_total = len(metadata)
    n_groups = grouping.nunique()

    global_r2 = permanova_r2(
        F=global_F,
        n=n_total,
        n_groups=n_groups,
    )

    # =========================================================
    # 3. Pairwise PERMANOVA
    # =========================================================
    pairwise_res = pairwise_permanova(
        dm=dm,
        metadata=metadata,
        group_col=group_col,
        n_perm=n_perm,
        seed=seed,
    )

    # =========================================================
    # 4. PCoA
    # =========================================================
    pcoa_res = pcoa(dm)

    # IMPORTANT:
    # scikit-bio 0.7.3 already returns eigenvalue-scaled
    # principal coordinates in pcoa_res.samples.
    coords = pcoa_res.samples.copy()

    # =========================================================
    # 5. Distances to group centroids
    # =========================================================
    disp_df = distance_to_centroid(
        coords=coords,
        metadata=metadata,
        group_col=group_col,
    )

    # =========================================================
    # 6. Dispersion summary
    # =========================================================
    disp_summary = summarize_dispersion(
        disp_df,
        group_col=group_col,
    )

    # =========================================================
    # 7. Formal PERMDISP
    # =========================================================
    permdisp_res = permdisp(
        distance_matrix=dm,
        grouping=grouping,
        test="centroid",
        permutations=n_perm,
        seed=seed,
    )

    # =========================================================
    # 8. Pairwise dispersion comparisons
    # =========================================================
    disp_pairs_df = pairwise_dispersion(
        disp_df,
        group_col=group_col,
    )

    return {
        "distance_matrix": dm,
        "metadata": metadata,
        "isolate_cluster_matrix": isolate_cluster_matrix,
        "pcoa": pcoa_res,
        "pcoa_coordinates": coords,
        "dispersion_distances": disp_df,
        "global_permanova": global_res,
        "global_permanova_R2": global_r2,
        "pairwise_permanova": pairwise_res,
        "permdisp": permdisp_res,
        "dispersion_summary": disp_summary,
        "dispersion_pairwise": disp_pairs_df,
    }


def _build_category_date_index(
    metadata_df: pd.DataFrame, category_col: str, date_col: str
) -> dict:
    """
    Build, per category, a sorted array of sampling-date timestamps
    (int64 ns) for *all* isolates in metadata_df (plasmid-bearing or
    not). Used to binary-search how many isolates of a category were
    sampled prior to a given date, i.e. the surveillance "opportunity"
    window for that category.
    """
    idx = {}
    dated = metadata_df.dropna(subset=[date_col])
    for cat, sub in dated.groupby(category_col):
        idx[cat] = np.sort(
            sub[date_col].values.astype("datetime64[ns]").astype("int64")
        )
    return idx


def _count_prior_isolates(idx: dict, category, date) -> float:
    """Number of isolates of `category` sampled strictly before `date`."""
    if pd.isna(date) or category not in idx:
        return np.nan
    ts = np.datetime64(date).astype("datetime64[ns]").astype("int64")
    return float(np.searchsorted(idx[category], ts, side="left"))


def _attach_dates(
    df: pd.DataFrame, metadata_df: pd.DataFrame, parent_col: str, date_col: str
) -> pd.DataFrame:
    """
    Attach date_col from metadata_df (indexed by parent_col) onto df
    via parent_col. Handles the case where df already has a
    same-named date column (avoids merge-suffix collisions) and
    dtype mismatches between the two parent_col representations.
    """
    if date_col not in metadata_df.columns:
        raise ValueError(f"'{date_col}' not found in metadata_df.")
    if metadata_df.index.name != parent_col:
        raise ValueError(
            f"metadata_df must be indexed by '{parent_col}' "
            f"(got index.name={metadata_df.index.name!r}). "
            f"Did you forget metadata_df.set_index('{parent_col}')?"
        )

    dates = metadata_df[[date_col]].copy()
    dates[date_col] = pd.to_datetime(dates[date_col], errors="coerce")
    dates = dates.rename_axis(parent_col).reset_index()

    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    if df[parent_col].dtype != dates[parent_col].dtype:
        df = df.copy()
        df[parent_col] = df[parent_col].astype(str)
        dates[parent_col] = dates[parent_col].astype(str)

    merged = df.merge(dates, on=parent_col, how="left")

    n_unmatched = merged[date_col].isna().sum()
    if n_unmatched:
        import warnings

        warnings.warn(
            f"{n_unmatched} of {len(merged)} rows had no matching "
            f"'{date_col}' after joining metadata_df on '{parent_col}'."
        )

    return merged


def _cluster_first_seen_per_category(
    df: pd.DataFrame, category_col: str, date_col: str
) -> dict:
    """
    For each (cluster, category), the earliest sampling date at which
    that plasmid cluster was observed in that category — regardless
    of gene content. Used as temporal context for gene-level novel
    appearances: did the gene arrive together with the cluster's
    first appearance in that category, or onto an already-established
    cluster background?

    Returns {(cluster, category): first_date}
    """
    dated = df.dropna(subset=[date_col])
    first_seen = dated.groupby([CLUSTER_COL, category_col])[date_col].min()
    return first_seen.to_dict()


def analyze_cluster_gene_spillover(
    df_in: pd.DataFrame,
    metadata_df: pd.DataFrame,
    category_col: str,
    parent_col: str = "Parent",
    date_col: str = "sampling_date",
    min_restricted_carriers: int = 5,
    min_prior_isolates_for_novel: int = 10,
    cluster_cooccurrence_window_days: int = 14,
):
    """
    Analyze functional-gene distribution across plasmid clusters, with
    a lightweight temporal read on spillover events.

    Spillover:
        A gene is present in plasmids from >1 category within the
        same plasmid cluster.

    Category restriction:
        A gene is present in exactly 1 category within the same
        plasmid cluster and is carried by at least
        `min_restricted_carriers` plasmids. Genes present in exactly
        1 category but below this threshold are neither spillover nor
        restricted — they're separated, but too rare (yet) to treat
        as an established category-specific pattern.

    Neither classification implies HGT or transmission.

    Temporal context (lightweight):
        `metadata_df` provides the *full* isolate-level sampling
        record (including isolates without any plasmid), keyed by
        `parent_col`, and is used purely to establish, per category,
        how much surveillance had happened by a given date.

        For each category that carries a given gene within a cluster,
        we find the date it first appears there and count how many
        isolates of that same category had already been sampled by
        that point. A category's first appearance is flagged
        `novel_appearance` when the gene/cluster was already present
        in another category before this category's first-seen date,
        AND at least `min_prior_isolates_for_novel` isolates of this
        category had already been sampled without it showing up. This
        distinguishes a plausible new introduction from simply not
        having sampled that category much yet.

        We additionally check, per category, whether the gene's first
        appearance coincides with the plasmid *cluster's* first
        appearance in that category (within
        `cluster_cooccurrence_window_days`) or whether the cluster was
        already established there beforehand — see
        `gene_cooccurs_with_cluster_first_appearance` in
        cluster_gene_category_df.

        This is descriptive only, not a transmission or HGT claim,
        and is intended as a triage signal ahead of the more rigorous
        downstream analysis.

    Parameters
    ----------
    df_in : pd.DataFrame
        Must contain: plasmid, cluster, parent_col, category_col,
        amr, virulence, metal, biocide.
    metadata_df : pd.DataFrame
        Isolate-level metadata (wider than df_in — includes isolates
        with no plasmid), indexed by parent_col, containing
        category_col and date_col.
    category_col : str
        Column defining the categories across which spillover is
        assessed, e.g. "ST" or "origin". Must already exist in df_in.
    parent_col : str
        Column in df_in identifying the parent isolate, used to join
        sampling dates from metadata_df.
    date_col : str
        Sampling date column in metadata_df.
    min_restricted_carriers : int
        Minimum number of plasmids carrying a gene for it to be
        classified as category restricted.
    min_prior_isolates_for_novel : int
        Minimum number of previously-sampled isolates of a category
        required for a later first-appearance to be flagged as a
        novel appearance rather than just undersampling.
    cluster_cooccurrence_window_days : int
        Max gap (days) between the gene's first-seen date and the
        cluster's first-seen date in a category for the gene to be
        considered as having arrived together with the cluster.

    Returns
    -------
    cluster_gene_df : pd.DataFrame
        One row per cluster x gene_function x gene. Passed to
        plot.spillover_summary.
    cluster_gene_category_df : pd.DataFrame
        One row per cluster x gene_function x gene x category —
        the per-category detail (dates, novel_appearance, cluster
        co-occurrence context) behind the summary table above.
    """
    df = df_in.copy()

    gene_function_cols = ["amr", "virulence", "metal", "biocide"]

    required_cols = [
        "Plasmid",
        CLUSTER_COL,
        parent_col,
        category_col,
        *gene_function_cols,
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df_in: {missing_cols}")

    meta_required = [category_col, date_col]
    missing_meta = [c for c in meta_required if c not in metadata_df.columns]
    if missing_meta:
        raise ValueError(f"Missing required columns in metadata_df: {missing_meta}")

    df = _attach_dates(df, metadata_df, parent_col, date_col)

    category_date_idx = _build_category_date_index(metadata_df, category_col, date_col)
    cluster_first_seen = _cluster_first_seen_per_category(df, category_col, date_col)

    # ---------------------------------------------------------
    # Convert comma-separated gene lists into long format
    # ---------------------------------------------------------
    gene_records = []

    for _, row in df.iterrows():
        for gene_function in gene_function_cols:
            gene_string = row[gene_function]

            if pd.isna(gene_string):
                continue
            gene_string = str(gene_string).strip()
            if not gene_string:
                continue

            genes = [g.strip() for g in gene_string.split(",") if g.strip()]

            for gene in genes:
                gene_records.append(
                    {
                        "Plasmid": row["Plasmid"],
                        CLUSTER_COL: row[CLUSTER_COL],
                        category_col: row[category_col],
                        date_col: row[date_col],
                        "gene": gene,
                        "gene_function": gene_function,
                    }
                )

    gene_df = pd.DataFrame(gene_records)

    if gene_df.empty:
        raise ValueError("No genes were detected in the functional gene columns.")

    gene_df = gene_df.drop_duplicates(
        subset=["Plasmid", CLUSTER_COL, category_col, "gene", "gene_function"]
    )

    cluster_sizes = df.groupby(CLUSTER_COL)["Plasmid"].nunique().rename("n_total")

    # ---------------------------------------------------------
    # Analyze each cluster x functional category x gene
    # ---------------------------------------------------------
    results = []
    category_results = []

    for (cluster, gene_function, gene), sub in gene_df.groupby(
        [CLUSTER_COL, "gene_function", "gene"]
    ):
        n_present = sub["Plasmid"].nunique()
        n_total = cluster_sizes.loc[cluster]
        percent_present = n_present / n_total * 100

        category_counts = (
            sub.groupby(category_col)["Plasmid"].nunique().sort_values(ascending=False)
        )
        n_categories_with_gene = len(category_counts)
        categories_with_gene = list(category_counts.index)

        spillover = n_categories_with_gene > 1
        category_restricted = (
            n_categories_with_gene == 1 and n_present >= min_restricted_carriers
        )
        restricted_category = categories_with_gene[0] if category_restricted else None

        # -------------------------------------------------
        # Temporal summary
        # -------------------------------------------------
        sub_dated = sub.dropna(subset=[date_col])
        category_first_seen = (
            sub_dated.groupby(category_col)[date_col].min().sort_values()
        )

        if not category_first_seen.empty:
            first_seen_category = category_first_seen.index[0]
            first_seen_date = category_first_seen.iloc[0]
        else:
            first_seen_category = None
            first_seen_date = pd.NaT

        novel_introduction_categories = []
        novel_cluster_introduction = False
        novel_gene_introduction = False

        for cat in categories_with_gene:
            cat_sub = sub[sub[category_col] == cat]
            cat_sub_dated = sub_dated[sub_dated[category_col] == cat]
            cat_first_date = (
                cat_sub_dated[date_col].min() if not cat_sub_dated.empty else pd.NaT
            )

            n_prior = _count_prior_isolates(category_date_idx, cat, cat_first_date)
            is_first = spillover and cat == first_seen_category

            novel_appearance = bool(
                spillover
                and not is_first
                and not pd.isna(n_prior)
                and n_prior >= min_prior_isolates_for_novel
            )

            # Did the gene arrive together with the cluster's own
            # first appearance in this category, or was the cluster
            # already established there beforehand?
            cluster_first_date = cluster_first_seen.get((cluster, cat), pd.NaT)
            if pd.isna(cluster_first_date) or pd.isna(cat_first_date):
                cooccurs_with_cluster = None
                days_after_cluster_first_seen = None
            else:
                days_after_cluster_first_seen = (
                    cat_first_date - cluster_first_date
                ).days
                cooccurs_with_cluster = (
                    days_after_cluster_first_seen <= cluster_cooccurrence_window_days
                )

            if novel_appearance:
                novel_introduction_categories.append(cat)
                if cooccurs_with_cluster:
                    novel_cluster_introduction = True
                else:
                    novel_gene_introduction = True

            category_results.append(
                {
                    CLUSTER_COL: cluster,
                    "gene_function": gene_function,
                    "gene": gene,
                    category_col: cat,
                    "n_present": cat_sub["Plasmid"].nunique(),
                    "first_date": cat_first_date,
                    "n_prior_isolates_in_category": n_prior,
                    "is_presumed_first_category": is_first,
                    "novel_appearance": novel_appearance,
                    "cluster_first_seen_in_category": cluster_first_date,
                    "days_after_cluster_first_seen": days_after_cluster_first_seen,
                    "gene_cooccurs_with_cluster_first_appearance": cooccurs_with_cluster,
                }
            )

        results.append(
            {
                CLUSTER_COL: cluster,
                "gene_function": gene_function,
                "gene": gene,
                "n_present": n_present,
                "n_total": n_total,
                "percent_present": percent_present,
                "n_categories_with_gene": n_categories_with_gene,
                "categories_with_gene": categories_with_gene,
                "spillover": spillover,
                "category_restricted": category_restricted,
                "restricted_category": restricted_category,
                "first_seen_date": first_seen_date,
                "first_seen_category": first_seen_category,
                "n_novel_introductions": len(novel_introduction_categories),
                "novel_introduction_categories": novel_introduction_categories,
                "novel_cluster_introduction": novel_cluster_introduction,
                "novel_gene_introduction": novel_gene_introduction,
            }
        )

    cluster_gene_df = pd.DataFrame(results)
    cluster_gene_category_df = pd.DataFrame(category_results)

    if category_col == config.ORIGIN_COL:
        main_figure = False
    else:
        main_figure = True

    plot.spillover_summary(cluster_gene_df, main_figure)

    return cluster_gene_df, cluster_gene_category_df


# ---------------------------------------------------------
# 3.1 Dataset Overview
# ---------------------------------------------------------
def dataset_overview(df_plasmids_in: pd.DataFrame, df_isolates_in: pd.DataFrame):
    df_plasmids = df_plasmids_in.copy()
    df_isolates = df_isolates_in.copy()

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
    n_isolates = df_isolates["KEY"].nunique()
    n_people = df_isolates[config.ID_COL].nunique()
    n_plasmids = df_plasmids[config.PLASMID_COL].nunique()
    n_isolates_with_plasmids = df_plasmids[config.ISOLATE_COL].nunique()

    # Temporal
    start_date = df_isolates[config.DATE_COL].min()
    end_date = df_isolates[config.DATE_COL].max()

    # Geo
    n_cities = df_isolates["city"].nunique()
    n_municipalities = df_isolates["municipality"].nunique()
    n_provinces = df_isolates["province"].nunique()
    with open(f"{out_path}/output.txt", "w") as f:
        f.write(f"n_isolates = {n_isolates}\n")
        f.write(f"n_patients = {n_people}\n")
        f.write(f"n_plasmids = {n_plasmids}\n")
        f.write(f"n_isolates with plasmids = {n_isolates_with_plasmids}\n")
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
    origin_iso = count_column_composition(df_isolates, "origin", False)
    origin_pls = count_column_composition(df_plasmids, "origin", False)
    origin = origin_iso.merge(
        origin_pls, on="origin", suffixes=("_isolates", "_plasmids")
    )
    origin.to_csv(f"{out_path}/origin_distribution.csv", sep=";")

    # Species origin
    species_iso = count_column_composition(df_isolates, config.SPECIES_COL, False)
    species_pls = count_column_composition(df_plasmids, config.SPECIES_COL, False)
    species = species_iso.merge(
        species_pls, on=config.SPECIES_COL, suffixes=("_isolates", "_plasmids")
    )
    species.to_csv(f"{out_path}/species_distribution.csv", sep=";")

    origin_sts = count_column_composition_by_cluster(
        df_isolates, config.ST_COL, "origin"
    )
    origin_sts.to_csv(f"{out_path}/origin_st_composition.csv", sep=";")

    # ---------------------------------------------------------
    # 3.1.3 Plasmid carriage statistics
    # ---------------------------------------------------------
    plasmid_carriage_counting(df_plasmids, df_isolates)
    plot_population(df_plasmids, df_isolates)

    # ---------------------------------------------------------
    # 3.1.4 Functional gene analysis
    # ---------------------------------------------------------
    # Mobility
    mobility = count_column_composition(df_plasmids, "mobility", False)
    mobility.to_csv(f"{out_path}/mobility_distribution.csv", sep=";")

    # Replicon
    replicon_full = count_column_composition(df_plasmids, "replicon", False)
    replicon_full.to_csv(f"{out_path}/replicon_full_distribution.csv", sep=";")
    replicon = count_column_composition(df_plasmids, "replicon", True)
    replicon.to_csv(f"{out_path}/replicon_distribution.csv", sep=";")

    # Functional genes
    for col in gene_columns:
        col_presence = count_column_composition(df_plasmids, f"{col}_plasmid", False)
        col_presence.to_csv(f"{out_path}/{col}_presence_distribution.csv", sep=";")
        colgenes = count_column_composition(df_plasmids, f"{col}", True)
        colgenes.to_csv(f"{out_path}/{col}_gene_distribution.csv", sep=";")
        df_plasmids[f"{col}_count"].describe().to_csv(
            f"results/dataset_overview/{col}_carriage_rates.csv", sep=";"
        )
        df_plasmids.groupby(f"{col}_plasmid")[f"{col}_count"].describe().to_csv(
            f"results/dataset_overview/{col}_only_carriage_rates.csv", sep=";"
        )

    gene_statics = gene_origin_enrichment(df_plasmids)
    gene_statics.to_csv(f"results/dataset_overview/gene_distribution_full.csv", sep=";")
    primary_cols = [
        "gene",
        "p_chi2_adj",
        "p_chi2_perm_adj",
        "cramers_v",
        "resid_pos_CA-MRSA",
        "resid_pos_HA-MRSA",
        "resid_pos_LA-MRSA",
        "resid_pos_MSSA",
        "resid_pos_Sar",
    ]

    supplementary_df = gene_statics[primary_cols]
    supplementary_df.to_csv(
        f"results/tables/tableS4_gene_distribution.csv", sep=";", index=False
    )

    tables = []
    for col in gene_columns:
        table = count_column_composition_by_cluster(
            df_in=df_plasmids,
            column_col=col,
            row_col="origin",
            split=True,
        )
        table["variable"] = col
        tables.append(table)

    result = pd.concat(tables)
    result.to_csv(f"{out_path}/origin_gene_counts.csv", sep=";")
    result = pd.read_csv(f"{out_path}/origin_gene_counts.csv", sep=";")
    plot.gene_heatmap(result, "origin")


# ---------------------------------------------------------
# 3.2 Plasmid Cluster Composition
# ---------------------------------------------------------
def cluster_overview(df_in: pd.DataFrame, metadata_df: pd.DataFrame):
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

    df_clustered = df.loc[df["clustered"] == "Clustered"]
    sizes = df_clustered.groupby(cluster_col).size()

    # Determine significant/common STs based on the number of isolates
    isolate_st_counts = metadata_df[config.ST_COL].value_counts()
    common_STs = isolate_st_counts[isolate_st_counts >= 25].index
    df_clustered["ST_collapsed"] = df_clustered["ISOLATE_TL_MLST_ST"].where(
        df_clustered["ISOLATE_TL_MLST_ST"].isin(common_STs),
        "rare_ST",
    )
    print(df_clustered["ST_collapsed"].unique())
    q1 = sizes.quantile(0.25)
    q2 = sizes.median()
    q3 = sizes.quantile(0.75)
    with open(f"{out_path}/cluster_stats.txt", "w") as f:
        f.write(f"Number of clusters:   {sizes.size}\n")
        f.write(f"Mean cluster size:    {sizes.mean():.2f}\n")
        f.write(f"Q1:                   {q1:.2f}\n")
        f.write(f"Median:               {q2:.2f}\n")
        f.write(f"Q3:                   {q3:.2f}\n")
        f.write(f"IQR:                  {q3 - q1:.2f}\n")
        f.write(f"Std:                  {sizes.std():.2f}\n")
        f.write(f"Min:                  {sizes.min()}\n")
        f.write(f"Max:                  {sizes.max()}\n")
        f.write("\n")

        # Number of unique categorical values per cluster
        categorical_cols = ["origin", config.ST_COL, "ST_collapsed"]
        for col in categorical_cols:
            counts = df_clustered.groupby(cluster_col)[col].nunique()

            q1 = counts.quantile(0.25)
            q2 = counts.median()
            q3 = counts.quantile(0.75)

            f.write(f"Number of unique {col} values per cluster:\n")
            f.write(f"  Mean:    {counts.mean():.2f}\n")
            f.write(f"  Q1:      {q1:.2f}\n")
            f.write(f"  Median:  {q2:.2f}\n")
            f.write(f"  Q3:      {q3:.2f}\n")
            f.write(f"  IQR:     {q3 - q1:.2f}\n")
            f.write(f"  Std:     {counts.std():.2f}\n")
            f.write(f"  Min:     {counts.min()}\n")
            f.write(f"  Max:     {counts.max()}\n")
            f.write("\n")

    plot.tsne_by_cluster(df)

    # ---------------------------------------------------------
    # 3.2.2 Clustering compositions
    # ---------------------------------------------------------
    clusters_per_species = df_clustered.groupby("origin")[cluster_col].nunique()
    clusters_per_STs = df_clustered.groupby(config.ST_COL)[cluster_col].nunique()
    clusters_per_ST_collapsed = df_clustered.groupby("ST_collapsed")[
        cluster_col
    ].nunique()

    q1 = clusters_per_STs.quantile(0.25)
    q2 = clusters_per_STs.median()
    q3 = clusters_per_STs.quantile(0.75)

    q1_col = clusters_per_ST_collapsed.quantile(0.25)
    q2_col = clusters_per_ST_collapsed.median()
    q3_col = clusters_per_ST_collapsed.quantile(0.75)
    with open(f"{out_path}/column_cluster_stats.txt", "w") as f:
        f.write("Number of clusters per species:\n")
        for species, count in clusters_per_species.items():
            f.write(f"{species}: {count}\n")

        # Species -> number of clusters
        f.write("\nNumber of clusters per ST:\n")
        f.write(f"Number of STs:         {clusters_per_STs.size}\n")
        f.write(f"Mean:                  {clusters_per_STs.mean():.2f}\n")
        f.write(f"Q1:                    {q1:.2f}\n")
        f.write(f"Median:                {q2:.2f}\n")
        f.write(f"Q3:                    {q3:.2f}\n")
        f.write(f"IQR:                   {q3 - q1:.2f}\n")
        f.write(f"Std:                   {clusters_per_STs.std():.2f}\n")
        f.write(f"Min:                   {clusters_per_STs.min()}\n")
        f.write(f"Max:                   {clusters_per_STs.max()}\n")

        f.write("\nNumber of clusters per common ST:\n")
        f.write(f"Number of STs:         {clusters_per_ST_collapsed.size}\n")
        f.write(f"Mean:                  {clusters_per_ST_collapsed.mean():.2f}\n")
        f.write(f"Q1:                    {q1_col:.2f}\n")
        f.write(f"Median:                {q2_col}\n")
        f.write(f"Q3:                    {q3_col}\n")
        f.write(f"IQR:                   {q3_col - q1_col:.2f}\n")
        f.write(f"Std:                   {clusters_per_ST_collapsed.std():.2f}\n")
        f.write(f"Min:                   {clusters_per_ST_collapsed.min()}\n")
        f.write(f"Max:                   {clusters_per_ST_collapsed.max()}\n")

    for col in ["origin", "replicon", "mobility", "AMR_plasmid"]:
        table = count_column_composition_by_cluster(
            df_in=df,
            column_col=col,
            split=True,
        )
        table.to_csv(f"{out_path}/{col}_cluster_counts.csv", sep=";")

    plot.composition_by_cluster(df)

    # ---------------------------------------------------------
    # 3.2.3 Plasmidome differences
    # ---------------------------------------------------------
    composition_dict = compartment_plasmidome_dispersion(df_clustered)
    tables = save_plasmidome_tables(
        composition_dict,
    )

    # ---------------------------------------------------------
    # 3.2.4 Cluster functional gene composition
    # ---------------------------------------------------------
    tables = []

    for col in gene_columns:
        table = count_column_composition_by_cluster(
            df_in=df_clustered,
            column_col=col,
            split=True,
        )
        table["variable"] = col
        tables.append(table)

    result = pd.concat(tables)
    result.to_csv(f"{out_path}/cluster_gene_counts.csv", sep=";")
    # result = pd.read_csv(f"{out_path}/cluster_gene_counts.csv", sep=";")
    # plot.gene_heatmap(result, cluster_col)

    # ---------------------------------------------------------
    # 3.2.5 Cluster functional gene spillover
    # ---------------------------------------------------------
    metadata_df["Parent"] = metadata_df["KEY"]
    metadata_df = metadata_df.set_index("Parent")
    cluster_gene_df, gene_spillover_summary = analyze_cluster_gene_spillover(
        df_clustered, metadata_df, config.ORIGIN_COL, date_col=config.DATE_COL
    )
    cluster_gene_df.to_csv(
        f"{out_path}/gene_spillover_origin_summary.csv", sep=";", index=False
    )
    gene_spillover_summary.to_csv(
        f"{out_path}/gene_spillover_origin.csv", sep=";", index=False
    )

    cluster_gene_df, gene_spillover_summary = analyze_cluster_gene_spillover(
        df_clustered, metadata_df, config.ST_COL, date_col=config.DATE_COL
    )
    cluster_gene_df.to_csv(
        f"{out_path}/gene_spillover_ST_summary.csv", sep=";", index=False
    )
    gene_spillover_summary.to_csv(
        f"{out_path}/gene_spillover_ST.csv", sep=";", index=False
    )


if __name__ == "__main__":
    print(
        "Functions related to analysing the MRSA plasmid dataset and cluster composistion."
    )
    print(
        "Used by mrsa_plasmid_analyis.py. Produces figures 1-5, S1-2 and tables S1-7."
    )
