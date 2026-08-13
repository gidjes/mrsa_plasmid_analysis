import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import seaborn as sns
from textwrap import wrap
import geopandas as gp
import os
import config
import baltic as bt

# ---------------------------------------------------------
# Declare variables
# ---------------------------------------------------------
ORIGIN_COL = config.ORIGIN_COL
SPECIES_COL = config.SPECIES_COL
ST_COL = config.ST_COL

CLUSTER_COL = config.CLUSTER_COL
TSNE1D = config.TSNE1D
TSNE2D = config.TSNE2D

ORIGIN_PALETTE_FULL = config.ORIGIN_PALETTE_FULL
ORIGIN_PALETTE = config.ORIGIN_PALETTE
SPECIES_PALETTE = config.SPECIES_PALETTE


# ---------------------------------------------------------
# Palette functions
# ---------------------------------------------------------
def create_palette(
    categories,
    assigned_cat: str,
    assigned_colour: str = "#FFFFFF",
    palette_name: str = "husl",
):
    # Normalize all categories to strings (Python-level)
    # categories = [str(c) for c in categories]
    # assigned_cat = str(assigned_cat)
    categories.sort()
    # Generate seaborn colors for all non-assigned categories
    num_colors = len(categories) - 1
    palette_colors = sns.color_palette(palette_name, num_colors)
    palette_colors_hex = [mcolors.to_hex(c) for c in palette_colors]

    # Assign colors
    palette_array = []
    color_idx = 0

    for cat in categories:
        if cat == assigned_cat:
            palette_array.append(assigned_colour)
        else:
            palette_array.append(palette_colors_hex[color_idx])
            color_idx += 1

    # Build dict
    palette_dict = dict(zip(categories, palette_array))
    return palette_dict


def lighten_color(color, amount):
    """
    Lightens the given color by moving it toward white.
    amount: 0 -> no change, 1 -> white
    """
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return mcolors.to_hex((1 - amount) * c + amount * white)


def darken_color(color, amount):
    """
    Darkens the given color by moving it toward black.
    amount: 0 -> no change, 1 -> black
    """
    c = np.array(mcolors.to_rgb(color))
    black = np.array([0, 0, 0])
    return mcolors.to_hex((1 - amount) * c + amount * black)


def custom_hierarchical_palette(colors, df, category_col, subcategory_col):
    """
    Generate a hierarchical palette with both lighter and darker shades
    for subcategories within each main category.

    Parameters:
        colors (list): Base colors to interpolate from (e.g., ['red', 'blue', 'green']).
        df (pd.DataFrame): DataFrame containing the data.
        category_col (str): Column name for main categories.
        subcategory_col (str): Column name for subcategories.

    Returns:
        dict: {subcategory_name: hex_color}
    """
    df = df[[category_col, subcategory_col]].drop_duplicates()
    main_categories = df[category_col].unique()
    main_palette = sns.color_palette(colors, n_colors=len(main_categories)).as_hex()

    color_map = {}
    main_color_dict = dict(zip(main_categories, main_palette))

    for main_cat in main_categories:
        sub_df = df[df[category_col] == main_cat]
        subcategories = sub_df[subcategory_col].tolist()
        n_sub = len(subcategories)

        base_color = main_color_dict[main_cat]

        if n_sub > 1:
            # Split subcategories around base color: darker → base → lighter
            mid_idx = n_sub // 2
            amounts = np.linspace(0.3, 0, mid_idx, endpoint=False)[::-1]
            dark_shades = [darken_color(base_color, amt) for amt in amounts]

            amounts = np.linspace(0, 0.3, n_sub - mid_idx)
            light_shades = [lighten_color(base_color, amt) for amt in amounts]

            shades = dark_shades + light_shades
        else:
            shades = [base_color]

        color_map.update(dict(zip(subcategories, shades)))

    return color_map


def generate_colour_dict(
    df: pd.DataFrame,
    cat_col: str,
    color_list: list,
    id_col: str = "Plasmid",
    subcat_col: str = None,
    use_hierarchical_palette: bool = False,
):
    """
    Generates color dictionaries for categories and IDs.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cat_col (str): The main category column.
        color_list (list): List of base colors.
        id_col (str): Column containing unique IDs (e.g., sample or data point ID).
        subcat_col (str): Optional subcategory column for hierarchical coloring.
        use_hierarchical_palette (bool): Whether to use a hierarchical color palette.

    Returns:
        id_color_dict (dict): {id: color}
        category_color_dict (dict): {category_or_subcategory: color}
    """
    if use_hierarchical_palette and subcat_col:
        # Use hierarchical palette function
        category_color_dict = custom_hierarchical_palette(
            color_list, df, category_col=cat_col, subcategory_col=subcat_col
        )
        id_color_dict = {
            row[id_col]: category_color_dict[row[subcat_col]]
            for _, row in df.iterrows()
        }
    else:
        # Flat color assignment by main category
        categories = sorted(df[cat_col].unique())
        category_color_dict = {
            cat: color_list[i % len(color_list)] for i, cat in enumerate(categories)
        }
        id_color_dict = {
            row[id_col]: category_color_dict[row[cat_col]] for _, row in df.iterrows()
        }

    return id_color_dict, category_color_dict


# ---------------------------------------------------------
# Manual legend helper
# ---------------------------------------------------------
def add_manual_legend(
    ax,
    palette,
    labels,
    title=None,
    ncol=4,
):
    handles = [
        Patch(
            facecolor=palette[label],
            edgecolor="none",
            label=str(label),
        )
        for label in labels
    ]

    ax.legend(
        handles=handles,
        title=title,
        ncol=ncol,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        borderaxespad=0,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=8,
        title_fontsize=9,
    )


def plot_count_map(
    df_in: pd.DataFrame,
    map_boxes: pd.DataFrame,
    ax,
    # sampleless_locations: list,
    # empty_map: pd.DataFrame | None = None,
    palette="",
):
    df = df_in.copy()

    # Set the palette
    if palette == "":
        cmap = "jet"
        vmin = df["count"].min()
        vmax = df["count"].max()
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    # Or set the external palette
    else:
        norm = mcolors.Normalize(vmin=0, vmax=palette)
        cmap = plt.get_cmap("jet")  # or any other colormap

    # plot the map
    df.plot(
        ax=ax,
        column="count",
        cmap=cmap,
        norm=norm,
        markersize=70,
        edgecolor="lightgrey",
        linewidth=0.2,
        legend=True,
        missing_kwds={
            "color": "whitesmoke",
            "edgecolor": "lightcoral",
            "linewidth": 0.2,
            "hatch": "///",
            "label": "None",
        },
    )

    # Put boxes around the overseas territories/insets
    map_boxes["regio_naam"] = map_boxes["regio_naam"].str.replace(
        "-eilanden",
        "\nislands",
        case=False,
        regex=False,
    )
    map_boxes.plot(facecolor="none", edgecolor="black", ax=ax)
    # Label each inset below its box
    for _, row in map_boxes.iterrows():
        x = row.geometry.centroid.x
        y = row.geometry.bounds[1]

        ax.annotate(
            row["regio_naam"],
            xy=(x, y),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
        )

    # Remove the axes
    ax.axis("off")
    return ax


def isolate_data(df_in, geo_df_in, map_boxes, origin_col: str = ORIGIN_COL):
    cluster_col = CLUSTER_COL
    species_col = SPECIES_COL
    origin_col = ORIGIN_COL

    df = df_in.copy()
    geo_df = geo_df_in.copy()
    df[species_col] = df[species_col].replace(
        {
            "Staphylococcus aureus": r"$\it{Staphyloccocus}$ $\it{aureus}$",
            "Staphylococcus argenteus": r"$\it{Staphyloccocus}$ $\it{argenteus}$",
            "Staphylococcus schweitzeri": r"$\it{Staphyloccocus}$ $\it{argenteus}$",
        }
    )
    df[origin_col] = df[origin_col].replace(
        {
            "CA-MRSA": "Community-associated\nMRSA",
            "HA-MRSA": "Hopsital-associated\nMRSA",
            "LA-MRSA": "Livestock-associated\nMRSA",
            "MSSA": r"Sensitive $\it{S. aureus}$",
            "Sar": r"$\it{S. argenteus}$",
            "Ssc": r"$\it{S. argenteus}$",
        }
    )
    plasmid_count = df.groupby("Parent").agg(
        plasmid_count=("Plasmid", "nunique"),
        total_amr_genes=("amr_count", "sum"),
        origin=(origin_col, "first"),
        isolate_species=(species_col, "first"),
        sampling_date=("MATERIAL_SAMPLINGDATE", "first"),
    )
    plasmid_count.origin = pd.Categorical(
        plasmid_count[cluster_col],
        categories=[
            "Community-associated\nMRSA",
            "Hopsital-associated\nMRSA",
            "Livestock-associated\nMRSA",
            r"Sensitive $\it{S. aureus}$",
            r"$\it{S. argenteus}$",
        ],
        ordered=True,
    )

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    plasmid_count = plasmid_count.rename(columns={"isolate_species": "Species"})
    sns.histplot(
        plasmid_count,
        x="sampling_date",
        hue="Species",
        palette=SPECIES_PALETTE,
        ax=ax1,
        bins=100,
        multiple="stack",
    )
    ax1.grid()
    ax1.set_ylabel(r"Isolates ($\it{N}$)")
    ax1.set_xlabel("Sampling Date")
    ax1.set_title("A", loc="left")

    plasmid_count.origin = plasmid_count.origin.replace(
        {
            "CA-MRSA": "Community-associated\nMRSA",
            "HA-MRSA": "Hopsital-associated\nMRSA",
            "LA-MRSA": "Livestock-associated\nMRSA",
            "MSSA": r"Sensitive $\it{S. aureus}$",
            "Sar": r"$\it{S. argenteus}$",
            "Ssc": r"$\it{S. argenteus}$",
        }
    )
    sns.violinplot(
        data=plasmid_count,
        x=cluster_col,
        y="plasmid_count",
        hue=cluster_col,
        palette=ORIGIN_PALETTE_FULL,
        # split=True,
        bw_method=1,
        cut=0,
        gap=0.1,
        inner="quart",
        ax=ax3,
    )
    ax3.grid()
    ax3.set_ylabel(r"Plasmids ($\it{N}$)")
    ax3.set_xlabel("Epidiomological origin")
    ax3.set_title("C", loc="left")

    ax2 = plot_count_map(geo_df, map_boxes, ax2)
    ax2.set_title("B", loc="left")

    plt.tight_layout()

    plt.savefig(f"results/figures/figure1_isolate_plots.png", dpi=600)
    plt.clf()


def glm_forest(
    models,
    panel_titles,
    reference_categories,
    term_labels=None,
    output_name="plasmid_count_forest_plot",
    figsize=(8.27, 11.69),
    origin_col: str = ORIGIN_COL,
):
    """
    Publication-ready three-panel forest plot.

    Layout:
        A. Origin
        B. ST
        C. ST + origin

    The top panel spans the full width and is deliberately
    shorter than the two lower panels.
    """
    cluster_col = CLUSTER_COL

    # =========================================================
    # Extract results
    # =========================================================
    all_results = []
    for model, refs in zip(
        models,
        reference_categories,
    ):
        ci = model.conf_int()
        results = pd.DataFrame(
            {
                "term": model.params.index,
                "IRR": np.exp(model.params.values),
                "CI_lower": np.exp(ci[0].values),
                "CI_upper": np.exp(ci[1].values),
                "p_value": model.pvalues.values,
            }
        )

        # -----------------------------------------------------
        # Remove intercept
        # -----------------------------------------------------
        results = results[results["term"] != "Intercept"].copy()

        # -----------------------------------------------------
        # Clean coefficient names
        # -----------------------------------------------------
        def clean_term(term):

            term = term.replace(
                "C(origin)[T.",
                "",
            )
            term = term.replace(
                "C(ST_collapsed)[T.",
                "",
            )
            term = term.replace("]", "")
            return term

        results["term"] = results["term"].apply(clean_term)
        if term_labels is not None:
            results["term"] = results["term"].replace(term_labels)

        # -----------------------------------------------------
        # Add reference categories
        # -----------------------------------------------------
        reference_rows = []
        for variable, category in refs.items():
            variable_label = {
                cluster_col: cluster_col,
                "ST_collapsed": "ST",
            }.get(variable, variable)
            reference_rows.append(
                {
                    "term": (f"{variable_label}: " f"{category} (reference)"),
                    "IRR": 1.0,
                    "CI_lower": np.nan,
                    "CI_upper": np.nan,
                    "p_value": np.nan,
                }
            )

        if reference_rows:
            reference_df = pd.DataFrame(reference_rows)
            results = pd.concat(
                [
                    reference_df,
                    results,
                ],
                ignore_index=True,
            )
        all_results.append(results)

    # =========================================================
    # Determine common x-axis
    # =========================================================
    lower_values = []
    upper_values = []

    for results in all_results:
        lower_values.extend(results["CI_lower"].dropna().tolist())
        upper_values.extend(results["CI_upper"].dropna().tolist())

    xmin = min(lower_values) / 1.5
    xmax = max(upper_values) * 1.5

    # =========================================================
    # Figure and GridSpec
    # =========================================================
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        # Small top panel, large bottom panels
        height_ratios=[0.35, 1.55],
        # Small gap between panels
        hspace=0.25,
        wspace=0.45,
    )

    ax_origin = fig.add_subplot(gs[0, :])
    ax_ST = fig.add_subplot(gs[1, 0])
    ax_both = fig.add_subplot(
        gs[1, 1],
        sharex=ax_ST,
    )
    axes = [
        ax_origin,
        ax_ST,
        ax_both,
    ]

    # =========================================================
    # Plotting function for individual panels
    # =========================================================
    def plot_panel(
        ax,
        results,
        title,
    ):
        # Reverse so first row appears at top
        results = results.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(results))

        # -----------------------------------------------------
        # Non-reference terms
        # -----------------------------------------------------
        non_reference = results["CI_lower"].notna()
        plot_results = results[non_reference]
        y_plot = y[non_reference]

        # -----------------------------------------------------
        # Error bars
        # -----------------------------------------------------
        if len(plot_results) > 0:
            xerr = np.vstack(
                [
                    plot_results["IRR"] - plot_results["CI_lower"],
                    plot_results["CI_upper"] - plot_results["IRR"],
                ]
            )
            ax.errorbar(
                plot_results["IRR"],
                y_plot,
                xerr=xerr,
                fmt="o",
                capsize=3,
                markersize=5,
                linewidth=1.2,
            )

        # -----------------------------------------------------
        # Null effect
        # -----------------------------------------------------
        ax.axvline(
            1,
            linestyle="--",
            linewidth=1,
        )

        # -----------------------------------------------------
        # Axes
        # -----------------------------------------------------
        ax.set_xscale("log")
        ax.set_xlim(
            xmin,
            xmax,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(results["term"])
        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            pad=6,
        )

        # Grid
        ax.grid(
            axis="x",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

        # Remove unnecessary tick marks
        ax.tick_params(
            axis="y",
            length=0,
        )

        # Keep the panels visually clean
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # =========================================================
    # Draw panels
    # =========================================================
    plot_panel(
        ax_origin,
        all_results[0],
        panel_titles[0],
    )

    plot_panel(
        ax_ST,
        all_results[1],
        panel_titles[1],
    )

    plot_panel(
        ax_both,
        all_results[2],
        panel_titles[2],
    )

    # =========================================================
    # Axis labels
    # =========================================================
    ax_origin.set_xlabel("Incidence rate ratio (IRR)")
    ax_ST.set_xlabel("Incidence rate ratio (IRR)")
    ax_both.set_xlabel("Incidence rate ratio (IRR)")

    # =========================================================
    # Layout
    # =========================================================
    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    fig.savefig(
        f"results/figures/{output_name}.png",
        dpi=600,
        bbox_inches="tight",
    )
    return fig, axes


def gene_heatmap(result: pd.DataFrame, cluster_col: str):
    # Columns containing countries
    country_cols = [
        col for col in result.columns if col not in ["variable", cluster_col]
    ]

    # Extract percentage from "N (%)"
    heatmap_data = result.copy()

    for col in country_cols:
        heatmap_data[col] = (
            heatmap_data[col].str.extract(r"\(([\d.]+)%\)", expand=False).astype(float)
        )

    variables = heatmap_data["variable"].unique()

    # Number of columns of panels
    ncols = 4

    # Split variables into rows of panels
    variable_rows = [variables[i : i + ncols] for i in range(0, len(variables), ncols)]

    # Height of each panel row = maximum number of clusters
    # among the variables in that row
    row_heights = []

    for variable_row in variable_rows:
        max_rows = max(
            len(
                heatmap_data[heatmap_data["variable"] == variable][cluster_col].unique()
            )
            for variable in variable_row
        )

        row_heights.append(max_rows)

    # Number of rows in each variable
    n_rows = [
        heatmap_data[heatmap_data["variable"] == variable]["cluster"].nunique()
        for variable in variables
    ]

    fig = plt.figure(figsize=(15, sum(n_rows) * 0.35))

    gs = gridspec.GridSpec(
        nrows=len(variables),
        ncols=3,
        figure=fig,
        height_ratios=n_rows,
        hspace=0.6,
        wspace=0.3,
    )

    # Plot panels
    for row_idx, variable_row in enumerate(variable_rows):

        for col_idx, variable in enumerate(variable_row):

            ax = fig.add_subplot(gs[row_idx, col_idx])

            data = heatmap_data[heatmap_data["variable"] == variable].set_index(
                cluster_col
            )[country_cols]

            sns.heatmap(
                data,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                vmin=0,
                vmax=100,
                cbar=False,
                ax=ax,
            )

            ax.set_title(variable)
            ax.set_xlabel("Gene")
            ax.set_ylabel("Cluster")

    # Remove unused axes in final row
    for col_idx in range(len(variable_rows[-1]), ncols):
        ax = fig.add_subplot(gs[-1, col_idx])
        ax.remove()

    plt.tight_layout()
    plt.savefig(f"results/figures/figureS1_gene_counts.png", dpi=600)
    plt.clf()


def spillover_summary(df_in):
    df = df_in.copy()

    plt.figure(figsize=(10, 8))
    # fig, ax = plt.subplots(ncols=2, figsize=(10, 8), sharey=True)
    sns.scatterplot(
        data=df,
        x="total_plasmids_with_gene",
        y="spillover_fraction",
        size="n_clusters",
        # ax=ax[0],
    )
    plt.tight_layout()
    plt.savefig("results/figures/figureS2_gene_spillover.png", dpi=300)


def add_ellipse(df, ax, column_name):
    """Funtion to add elipses around a particular category in a dataframe plotted in a e.g. scatterplot.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe used for plot
    ax : ax object
        plot or ax slice to place it in
    column_name : str
        name of column with categories to use
    """
    all_values = list(set(df[f"{column_name}"].values))
    if "-1" in all_values:
        all_values.remove("-1")
    if -1 in all_values:
        all_values.remove(-1)
    if "-" in all_values:
        all_values.remove("-")
    if np.nan in all_values:
        all_values.remove(np.nan)
    for value in all_values:
        df_it = df.loc[df[f"{column_name}"] == value]
        x_coor = df_it[f"{ax.get_xlabel()}"].values
        x1 = min(x_coor)
        x2 = max(x_coor)
        y_coor = df_it[f"{ax.get_ylabel()}"].values
        y1 = min(y_coor)
        y2 = max(y_coor)
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2
        y_dir = (y2 - y1) / 2
        x_dir = (x2 - x1) / 2
        ellipse = Ellipse(
            (x0, y0), (x2 - x1 + 1), (y2 - y1 + 1), fill=False, label=value
        )
        ax.add_patch(ellipse)


def tsne_by_cluster(
    df_in: pd.DataFrame,
    cluster_col: str = CLUSTER_COL,
    tsne1D: str = TSNE1D,
    tsne2D: str = TSNE2D,
):
    df = df_in.copy()
    origin_col = ORIGIN_COL

    df = df.loc[df[cluster_col] != "-"]
    df[tsne1D] = pd.to_numeric(df[tsne1D], downcast="float", errors="coerce")
    df[tsne2D] = pd.to_numeric(df[tsne2D], downcast="float", errors="coerce")

    categories = df[cluster_col].unique().tolist()
    palette = create_palette(categories, "-1", "#8C8C8C")

    fig = plt.figure(figsize=(8.27, 11.69))
    gs = gridspec.GridSpec(
        2,
        2,
        # height_ratios=[1, 1.2], width_ratios=[1, 2]
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    sns.scatterplot(
        df,
        x=tsne1D,
        y=tsne2D,
        hue=cluster_col,
        alpha=0.5,
        palette=palette,
        ax=ax1,
        legend=False,
    )
    add_ellipse(df, ax1, cluster_col)
    # ax1.legend(ncol=3)
    ax1.set_ylabel("t-SNE 2D")
    ax1.set_xlabel("t-SNE 1D")
    ax1.set_title("A", loc="left")

    df = df.loc[df[cluster_col] != "-1"]
    df[origin_col] = df[origin_col].replace(
        {
            "Sar": r"$\it{S. argenteus}$",
        }
    )
    df[origin_col] = pd.Categorical(
        df[origin_col],
        categories=[
            "CA-MRSA",
            "HA-MRSA",
            "LA-MRSA",
            "MSSA",
            r"$\it{S. argenteus}$",
        ],
        ordered=True,
    )
    sns.scatterplot(
        df,
        x=tsne1D,
        y=tsne2D,
        hue=origin_col,
        alpha=0.5,
        palette=ORIGIN_PALETTE,
        legend=True,
        ax=ax2,
    )
    ax2.set_ylabel("t-SNE 2D")
    ax2.set_xlabel("t-SNE 1D")
    ax2.set_title("B", loc="left")
    ax2.legend(loc="lower left")

    df.AMR_plasmid = df.AMR_plasmid.replace(
        {
            1: "AMR plasmid",
            0: "Plasmid",
        }
    )
    palette = create_palette(df["AMR_plasmid"].unique(), "Plasmid", "#8C8C8C")
    sns.scatterplot(
        df,
        x=tsne1D,
        y=tsne2D,
        hue="AMR_plasmid",
        alpha=0.5,
        legend=True,
        palette=palette,
        ax=ax3,
    )
    ax3.set_ylabel("t-SNE 2D")
    ax3.set_xlabel("t-SNE 1D")
    ax3.set_title("C", loc="left")
    ax3.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"results/figures/figure3_tnse.png", dpi=300)
    plt.clf()


def composition_by_cluster(
    df_in,
    cluster_col: str = CLUSTER_COL,
    origin_col: str = ORIGIN_COL,
    st_col: str = ST_COL,
):
    df = df_in.copy()
    df = df.loc[~df[cluster_col].isin(["-", "-1"])]

    # ---------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------
    fig, axs = plt.subplots(
        4,
        figsize=(8.27, 11.69),
        sharex=True,
    )

    # ---------------------------------------------------------
    # Plot 1: Plasmid count
    # ---------------------------------------------------------
    sns.histplot(
        df,
        x=cluster_col,
        ax=axs[0],
    )
    axs[0].set_ylabel(r"Plasmid count ($\it{N}$)")
    axs[0].set_title("A", loc="left", x=-0.05)

    # ---------------------------------------------------------
    # Plot 2: Origin
    # ---------------------------------------------------------
    df[cluster_col] = df[cluster_col].replace(
        {
            "Sar": r"$\it{S. argenteus}$",
        }
    )

    origin_categories = [
        "CA-MRSA",
        "HA-MRSA",
        "LA-MRSA",
        "MSSA",
        r"$\it{S. argenteus}$",
    ]

    origin_palette = ORIGIN_PALETTE

    df[cluster_col] = pd.Categorical(
        df[cluster_col],
        categories=origin_categories,
        ordered=True,
    )

    sns.histplot(
        df,
        x=cluster_col,
        stat="percent",
        hue=cluster_col,
        palette=origin_palette,
        multiple="fill",
        legend=False,
        ax=axs[1],
    )
    axs[1].set_ylabel("Origin")
    axs[1].set_title("B", loc="left", pad=15, x=-0.05)

    # ---------------------------------------------------------
    # Plot 3: Isolate ST
    # ---------------------------------------------------------

    # Determine common STs based on NUMBER OF ISOLATES.
    # Parent is used so that multiple rows belonging to the
    # same isolate are only counted once.
    isolate_df = df[
        [
            "Parent",
            st_col,
        ]
    ].drop_duplicates()

    isolate_st_counts = isolate_df[st_col].value_counts()

    common_STs = isolate_st_counts[isolate_st_counts >= 10].index

    df[st_col] = df[st_col].where(
        df[st_col].isin(common_STs),
        "Rare ST",
    )

    # Order STs by abundance rather than alphabetically.
    st_counts = df[st_col].value_counts()

    st_categories = [st for st in st_counts.index if st != "Rare ST"]

    # Put Rare ST at the end of the legend.
    if "Rare ST" in st_counts.index:
        st_categories.append("Rare ST")

    st_palette = create_palette(
        st_categories,
        "Rare ST",
        "#8C8C8C",
    )

    sns.histplot(
        df,
        x=cluster_col,
        stat="percent",
        hue=st_col,
        multiple="fill",
        palette=st_palette,
        legend=False,
        ax=axs[2],
    )
    axs[2].set_ylabel("Isolate ST")
    axs[2].set_title("C", loc="left", pad=15, x=-0.05)

    # ---------------------------------------------------------
    # Plot 4: ARG count
    # ---------------------------------------------------------
    amr_categories = list(range(0, 6))

    amr_palette = create_palette(
        amr_categories,
        0,
        "#8C8C8C",
        "flare",
    )

    sns.histplot(
        df,
        x=cluster_col,
        stat="percent",
        hue="amr_count",
        multiple="fill",
        palette=amr_palette,
        legend=False,
        ax=axs[3],
    )
    axs[3].set_ylabel(r"ARGs ($\it{N}$)")
    axs[3].set_title("D", loc="left", pad=15, x=-0.05)

    axs[3].tick_params(axis="x", rotation=90, labelsize=8)

    # ---------------------------------------------------------
    # Manual legends
    # ---------------------------------------------------------

    # Origin: one row
    add_manual_legend(
        axs[1],
        origin_palette,
        origin_categories,
        title=cluster_col,
        ncol=4,
    )

    # Isolate ST: compact multi-row legend
    add_manual_legend(
        axs[2],
        st_palette,
        st_categories,
        title="Isolate ST",
        ncol=16,
    )

    # ARG count: one row
    add_manual_legend(
        axs[3],
        amr_palette,
        amr_categories,
        title="ARG count",
        ncol=6,
    )

    # ---------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------
    for ax in axs:
        ax.get_yaxis().set_label_coords(
            -0.06,
            0.5,
        )

    # ---------------------------------------------------------
    # Explicit figure spacing
    # ---------------------------------------------------------
    # Avoid tight_layout(), which tends to allocate excessive
    # space around the multi-row ST legend.
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.98,
        bottom=0.05,
        hspace=0.30,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    fig.savefig(
        "results/figures/figure4_cluster_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_tanglegram_full(
    df_in: pd.DataFrame,
    ax: Axes,
    cluster: str | int,
    node_left: str,
    node_right: str,
    tangles: str,
    palette: str | list = "",
    outline: bool = False,
    outline_left: str = "",
    outline_right: str = "",
    shared_x_scale: bool = False,
    outlier_only: bool = False,
):
    """
    Generalised function to plot tanglegrams
    Plasmids will be on the left, isolates will be on the right

    Parameters
    ----------
    df_in : pd.DataFrame
        _description_
    ax : Axes
        _description_
    cluster : str | int
        _description_
    node_left : str
        _description_
    node_right : str
        _description_
    tangles : str
        _description_
    palette : str | list, optional
        _description_, by default ""
    outline : bool, optional
        _description_, by default False
    outline_left : str, optional
        _description_, by default ""
    outline_right : str, optional
        _description_, by default ""

    Returns
    -------
    _type_
        _description_
    """

    # Prepare dataframe
    df = df_in.copy()
    df.fillna({node_right: "Unknown"}, inplace=True)

    # Define colourschemes
    preset_colours = [
        "#a90061",
        "#275937",
        "#007bc7",
        "#f9e11e",
        "#d52b1e",
        "#777b00",
        "#76d2b6",
        "#673327",
        "#552c6f",
        "#f092cd",
        "#154273",
        "#94710a",
        "#7c796f",
        "#1baa62",
    ]
    colors = [
        "#FF0000",  # Red
        "#00FF00",  # Lime
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FFA500",  # Orange
        "#800080",  # Purple
        "#00FFFF",  # Cyan
        "#FFC0CB",  # Pink
        "#A52A2A",  # Brown
        "#808080",  # Gray
        "#000000",  # Black
        "#FFFFFF",  # White
        "#008000",  # Green
        "#800000",  # Maroon
        "#008080",  # Teal
        "#000080",  # Navy
        "#FFD700",  # Gold
        "#4B0082",  # Indigo
        "#7FFF00",  # Chartreuse
        "#DC143C",  # Crimson
        "#00CED1",  # DarkTurquoise
        "#FF1493",  # DeepPink
        "#1E90FF",  # DodgerBlue
        "#B22222",  # FireBrick
        "#228B22",  # ForestGreen
        "#DAA520",  # GoldenRod
        "#ADFF2F",  # GreenYellow
        "#F08080",  # LightCoral
        "#90EE90",  # LightGreen
        "#20B2AA",  # LightSeaGreen
        "#87CEFA",  # LightSkyBlue
    ]
    outbreak_colours = ["#00CED1", "#C0C0C0", "#800080", "#F08080", "#ADFF2F"]
    species_colours = ["seagreen", "gold"]
    outline_colours = [
        "firebrick",
        "goldenrod",
        "darkcyan",
    ]
    left_colours = [
        "#C0C0C0",
        "#007bc7",
        "#154273",
        "#a90061",
    ]
    df["Parent"] = df["Parent"].astype(str)

    # Generate the dictionary containing the colours for each point
    if palette == "":
        tree1_colour_dict, tree1_legend = generate_colour_dict(
            df,
            node_left,
            left_colours,
            "Plasmid",
        )
        tree2_colour_dict, tree2_legend = generate_colour_dict(
            df,
            node_right,
            colors,
            "Parent",
        )
    # Or extact it from an external source (supplementary figs)
    else:
        tree1_colour_dict, tree1_legend = palette[0][0], palette[0][1]
        tree2_colour_dict, tree2_legend = palette[1][0], palette[1][1]

    if outline:
        tree1_outline_dict, tree1_outline_legend = generate_colour_dict(
            df,
            outline_left,
            colors,
            "Plasmid",
        )

        tree2_outline_dict, tree2_outline_legend = generate_colour_dict(
            df,
            outline_right,
            outline_colours,
            "Parent",
        )
    elif outline and palette != "":
        tree1_outline_dict, tree1_outline_legend = palette[0][0], palette[0][1]
        tree2_outline_dict, tree2_outline_legend = palette[1][0], palette[1][1]
    else:
        tree1_outline_dict = {}
        tree2_outline_dict = {}

    # Generate the tangle colours
    if pd.api.types.is_numeric_dtype(df[tangles]):
        values = df[tangles].astype(float)

        # norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        if "wgmlst" in df["host_dist_source"].unique():
            norm = mcolors.Normalize(vmin=0, vmax=2500)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=values.max())
        base_cmap = cm.get_cmap("nipy_spectral")

        # Skip darkest part of cmap
        colors = base_cmap(np.linspace(0.08, 1, 256))

        cmap = mcolors.LinearSegmentedColormap.from_list("trimmed_spectral", colors)
        cmap.set_bad(color="lightgrey")

        # Map sample name -> colour
        tangle_colour_dict = {
            row["Plasmid"]: cmap(norm(row[tangles]))
            for _, row in df[["Plasmid", tangles]].iterrows()
        }

        tangle_legend = None
    else:
        tangle_colour_dict, tangle_legend = generate_colour_dict(
            df, tangles, outbreak_colours
        )

    # Load trees
    path_to_tree1 = f"mashtree/{cluster}_tree.dnd"
    path_to_tree2 = f"mashtree/{cluster}_chr_tree.dnd"

    tree1 = bt.loadNewick(path_to_tree1, absoluteTime=False)
    tree2 = bt.loadNewick(path_to_tree2, absoluteTime=False)

    # Get the maximum tree height across both trees
    if shared_x_scale:
        max_height1 = max_height2 = max(tree1.treeHeight, tree2.treeHeight)
    else:
        max_height1 = tree1.treeHeight
        max_height2 = tree2.treeHeight

    # Pad each tree so they “fit” the same x-scale visually
    xpad = 0.1  # left/right padding as a fraction of width
    ypad = 1.2  # vertical buffer around node clusters

    # Define widths (can force equal width if desired)
    total_plot_width = 80
    gap_ratio = 0.20
    gap = total_plot_width * gap_ratio
    available_width = total_plot_width - gap

    tree1_width = available_width / 2
    tree2_width = available_width / 2

    # Define normalization functions using same scale + padding
    def x_attr1(k):
        scaled = (k.height / max_height1) * (1 - 2 * xpad)
        return xpad * tree1_width + scaled * tree1_width

    def x_attr2(k):
        scaled = (k.height / max_height2) * (1 - 2 * xpad)
        return tree1_width + gap + (1 - xpad - scaled) * tree2_width

    # Plot trees once first (to initialize .y if needed)
    tree1.plotTree(ax, x_attr=x_attr1)
    tree2.plotTree(ax, x_attr=x_attr2)

    # Compute old min/max from those initial coordinates
    ymin1 = min(k.y for k in tree1.Objects if hasattr(k, "y"))
    ymax1 = max(k.y for k in tree1.Objects if hasattr(k, "y"))
    ymin2 = min(k.y for k in tree2.Objects if hasattr(k, "y"))
    ymax2 = max(k.y for k in tree2.Objects if hasattr(k, "y"))

    # Determine the global shared y-range
    global_ymin = min(ymin1, ymin2)
    global_ymax = max(ymax1, ymax2)

    # Rescale both trees’ y-values *before re-plotting*
    def rescale_y(tree, old_min, old_max, new_min, new_max):
        scale = (new_max - new_min) / (old_max - old_min)
        for k in tree.Objects:
            if hasattr(k, "y"):
                k.y = (k.y - old_min) * scale + new_min

    rescale_y(tree1, ymin1, ymax1, global_ymin, global_ymax)
    rescale_y(tree2, ymin2, ymax2, global_ymin, global_ymax)

    # Clear any previous tree lines
    ax.clear()

    # Replot the trees and points using rescaled y-coordinates
    tree1.plotTree(ax, x_attr=x_attr1)
    tree1.plotPoints(
        ax,
        x_attr=x_attr1,
        target=lambda k: k.branchType == "leaf",
        colour=lambda k: tree1_colour_dict[k.name],
        outline=outline,
        outline_colour=lambda k: (
            tree1_outline_dict[k.name] if k.name in tree1_outline_dict else "black"
        ),
        zorder=100,
    )

    tree2.plotTree(ax, x_attr=x_attr2)
    tree2.plotPoints(
        ax,
        x_attr=x_attr2,
        target=lambda k: k.branchType == "leaf",
        colour=lambda k: tree2_colour_dict[k.name],
        outline=outline,
        outline_colour=lambda k: (
            tree2_outline_dict[k.name] if k.name in tree2_outline_dict else "black"
        ),
        zorder=100,
    )

    # Create the tangles
    # Build lookup of outlier isolates
    if outlier_only:
        outlier_names = set(
            df.loc[df["is_outlier"], "plasmid"]  # or whatever key matches k.name
        )
    else:
        outlier_names = set(df["plasmid"])

    for k in filter(lambda x: x.branchType == "leaf", tree1.Objects):
        if k.name not in outlier_names:
            continue  ## grab leaf objects in tree1
        x = k.height  ## get height
        y = k.y  ## get y position

        matching_tip = tree2.getBranches(
            lambda x: x.branchType == "leaf"
            and x.name.split("_")[0] == k.name.split("_")[0]
        )  ## fetch corresponding branch in tree2

        # X coordinated for each line
        match_y = matching_tip.y
        x1 = x_attr1(k)
        x4 = x_attr2(matching_tip)

        # create line path
        xs = [
            x1,
            tree1_width + 0.15 * gap,
            tree1_width + 0.85 * gap,
            x4,
        ]
        ys = [y, y, match_y, match_y]  ## y coordinates for tangleline
        line_colour = tangle_colour_dict[k.name]
        ax.plot(xs, ys, color=line_colour, alpha=0.3)  ## plot tangleline

    # Create room for legends
    ax.set_xlim(-0.20 * total_plot_width, 1.20 * total_plot_width)

    # Create legends
    if palette == "":
        # Tree 1 legend on the left
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in tree1_legend.values()
        ]
        labels = ["\n".join(wrap(l, 15)) for l in tree1_legend.keys()]
        legend1 = ax.legend(
            markers,
            labels,
            title=f"Node colour\n{node_left}",
            numpoints=1,
            loc="upper left",
            bbox_to_anchor=(-0.08, 1),
        )
        ax.add_artist(legend1)

        # Tree 2 legend on the right
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in tree2_legend.values()
        ]
        labels = ["\n".join(wrap(l, 15)) for l in tree2_legend.keys()]
        legend2 = ax.legend(
            markers,
            labels,
            title=f"Node colour\n{node_right}",
            numpoints=1,
            loc="upper right",
            bbox_to_anchor=(1.08, 1.04),
            # borderaxespad=3,
        )
        ax.add_artist(legend2)

    if outline:
        # Get bounding box of first legend (in display coords)
        bbox = legend1.get_window_extent()
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        y_bottom = bbox_axes.y0
        spacing = 0.01

        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in tree1_outline_legend.values()
        ]
        labels = ["\n".join(wrap(l, 15)) for l in tree1_outline_legend.keys()]
        legend1b = ax.legend(
            markers,
            labels,
            title=f"Outline colour\n{outline_left}",
            numpoints=1,
            loc="upper left",
            bbox_to_anchor=(-0.08, y_bottom - spacing),  # same x as legend1
        )
        ax.add_artist(legend1b)

        # Get bounding box of first legend (in display coords)
        bbox = legend2.get_window_extent()
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        y_bottom = bbox_axes.y0
        spacing = 0.01
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in tree2_outline_legend.values()
        ]
        labels = ["\n".join(wrap(l, 15)) for l in tree2_outline_legend.keys()]
        legend2b = ax.legend(
            markers,
            labels,
            title=f"Outline colour\n{outline_right}",
            loc="upper right",
            bbox_to_anchor=(1.08, y_bottom - spacing),  # same x as legend2
            # borderaxespad=3,
        )
        ax.add_artist(legend2b)

        # Tangle legend on top
        if pd.api.types.is_numeric_dtype(df[tangles]):
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(tangles)
        else:
            markers = [
                plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
                for color in tangle_legend.values()
            ]
            ax.legend(
                markers,
                tangle_legend.keys(),
                title=f"Tangle colour     {tangles}",
                numpoints=1,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(tangle_legend.keys()),
            )

    # Remove axes
    ax.axis("off")
    return ax


def plot_single_tangle(
    df_in: pd.DataFrame,
    cluster_id: str,
    left_node: str,
    right_node: str,
    tangle: str,
    outline_left: str,
    outline_right: str,
):
    tangle_df = df_in.copy()

    fig, ax = plt.subplots(figsize=(12, 20))
    plot_tanglegram_full(
        f"mashtree/{cluster_id}_tree.dnd",
        f"mashtree/{cluster_id}_chr_tree.dnd",
        tangle_df,
        ax,
        left_node,
        right_node,
        tangle,
        outline=True,
        outline_left=outline_left,
        outline_right=outline_right,
    )
    plt.savefig(
        f"results/outlier_tangles/{cluster_id}.png",
        # bbox_inches="tight",
        dpi=600,
    )
    plt.clf()
