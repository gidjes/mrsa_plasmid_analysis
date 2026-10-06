import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns
from textwrap import wrap
import geopandas as gp
import config
import baltic as bt

# ---------------------------------------------------------
# Declare variables
# ---------------------------------------------------------
ORIGIN_COL = config.ORIGIN_COL
SPECIES_COL = config.SPECIES_COL
ST_COL = config.ST_COL
DATE_COL = config.DATE_COL

CLUSTER_COL = config.CLUSTER_COL
TSNE1D = config.TSNE1D
TSNE2D = config.TSNE2D

ORIGIN_PALETTE_FULL = config.ORIGIN_PALETTE_FULL
ORIGIN_PALETTE = config.ORIGIN_PALETTE
SPECIES_PALETTE = config.SPECIES_PALETTE

FIGSIZE_LANDSCAPE = (11.69, 8.27)
FIGSIZE_PORTRAIT = (8.27, 11.69)


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
    color_list: list | dict,
    id_col: str = "Plasmid",
    subcat_col: str = None,
    use_hierarchical_palette: bool = False,
):
    """
    Generates color dictionaries for categories and IDs.

    Parameters:
        df (pd.DataFrame):
            The input DataFrame.

        cat_col (str):
            The main category column.

        color_list (list | dict):
            Either:
            - a list of colours, which are assigned sequentially; or
            - a dictionary mapping category names to colours.

        id_col (str):
            Column containing unique IDs (e.g., sample or data point ID).

        subcat_col (str, optional):
            Optional subcategory column for hierarchical coloring.

        use_hierarchical_palette (bool):
            Whether to use a hierarchical color palette.

    Returns:
        id_color_dict (dict):
            {id: color}

        category_color_dict (dict):
            {category_or_subcategory: color}
    """

    # ------------------------------------------------------------------
    # Hierarchical palette
    # ------------------------------------------------------------------
    if use_hierarchical_palette and subcat_col:
        if isinstance(color_list, dict):
            # Named colours: use the supplied colours directly
            category_color_dict = {
                subcat: color_list[subcat]
                for subcat in df[subcat_col].dropna().unique()
            }
        else:
            # Existing behaviour
            category_color_dict = custom_hierarchical_palette(
                color_list,
                df,
                category_col=cat_col,
                subcategory_col=subcat_col,
            )

        id_color_dict = {
            row[id_col]: (
                category_color_dict[row[subcat_col]]
                if pd.notna(row[subcat_col])
                else "lightgrey"
            )
            for _, row in df.iterrows()
        }

    # ------------------------------------------------------------------
    # Flat palette
    # ------------------------------------------------------------------
    else:
        categories = sorted(df[cat_col].dropna().unique())

        if isinstance(color_list, dict):
            # Named palette
            missing_categories = set(categories) - set(color_list)

            if missing_categories:
                raise ValueError(
                    f"No colour defined for categories: "
                    f"{sorted(missing_categories)}"
                )

            category_color_dict = {cat: color_list[cat] for cat in categories}

        else:
            # Existing sequential palette behaviour
            if not color_list:
                raise ValueError("color_list cannot be empty.")

            category_color_dict = {
                cat: color_list[i % len(color_list)] for i, cat in enumerate(categories)
            }

        id_color_dict = {
            row[id_col]: (
                category_color_dict[row[cat_col]]
                if pd.notna(row[cat_col])
                else "lightgrey"
            )
            for _, row in df.iterrows()
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


def isolate_data(
    df_plasmids_in, df_isolates_in, geo_df_in, map_boxes, origin_col: str = ORIGIN_COL
):
    cluster_col = CLUSTER_COL
    species_col = SPECIES_COL
    origin_col = ORIGIN_COL

    df_plasmids = df_plasmids_in.copy()
    df_isolates = df_isolates_in.copy()
    geo_df = geo_df_in.copy()
    df_plasmids[species_col] = df_plasmids[species_col].replace(
        {
            "Staphylococcus aureus": r"$\it{Staphyloccocus}$ $\it{aureus}$",
            "Staphylococcus argenteus": r"$\it{Staphyloccocus}$ $\it{argenteus}$",
            "Staphylococcus schweitzeri": r"$\it{Staphyloccocus}$ $\it{argenteus}$",
        }
    )
    df_plasmids[origin_col] = df_plasmids[origin_col].replace(
        {
            "CA-MRSA": "Community-associated\nMRSA",
            "HA-MRSA": "Hopsital-associated\nMRSA",
            "LA-MRSA": "Livestock-associated\nMRSA",
            "MSSA": r"Sensitive $\it{S. aureus}$",
            "Sar": r"$\it{S. argenteus}$",
            "Ssc": r"$\it{S. argenteus}$",
        }
    )

    df_isolates[species_col] = df_isolates[species_col].replace(
        {
            "Staphylococcus aureus": r"$\it{Staphyloccocus}$ $\it{aureus}$",
            "Staphylococcus argenteus": r"$\it{Staphyloccocus}$ $\it{argenteus}$",
            "Staphylococcus schweitzeri": r"$\it{Staphyloccocus}$ $\it{argenteus}$",
        }
    )
    df_isolates[origin_col] = df_isolates[origin_col].replace(
        {
            "CA-MRSA": "Community-associated\nMRSA",
            "HA-MRSA": "Hopsital-associated\nMRSA",
            "LA-MRSA": "Livestock-associated\nMRSA",
            "MSSA": r"Sensitive $\it{S. aureus}$",
            "Sar": r"$\it{S. argenteus}$",
            "Ssc": r"$\it{S. argenteus}$",
        }
    )

    # Count plasmids per isolate
    plasmid_counts = (
        df_plasmids.groupby("Parent").size().reset_index(name="plasmid_count")
    )

    # Combine plasmid count with isolate metadata
    plasmid_count = (
        df_isolates[["KEY", origin_col, DATE_COL]]
        .merge(
            plasmid_counts,
            left_on="KEY",
            right_on="Parent",
            how="left",
        )
        .fillna(0)
    )
    print(plasmid_count)
    print(plasmid_count.columns)

    plasmid_count.origin = pd.Categorical(
        plasmid_count[origin_col],
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

    df_isolates = df_isolates.rename(columns={SPECIES_COL: "Species"})
    sns.histplot(
        df_isolates,
        x=DATE_COL,
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
        x=origin_col,
        y="plasmid_count",
        hue=origin_col,
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
    ax3.tick_params(axis="x", labelsize=8)
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

    # Move only the bottom-left panel to the left
    pos = ax_ST.get_position()
    ax_ST.set_position(
        [
            pos.x0 - 0.1,  # move left
            pos.y0,
            pos.width,
            pos.height,
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


def gene_heatmap(result: pd.DataFrame, x_col: str):
    # Set variables for origin vs cluster plot
    if x_col == ORIGIN_COL:
        rstring = r"^([\d.]+)"
        annotate = True
        colorbar_label = "Count"
        xlabel = "Origin"
        outpath = "results/figures/figureS1_gene_count_origin.png"

    else:
        raise ValueError(f"Unsupported x_col: {x_col}")

    # Columns containing genes
    gene_cols = [col for col in result.columns if col not in ["variable", x_col]]

    heatmap_data = result.copy()

    # ---------------------------------------------------------
    # Convert values to counts or percentages
    # ---------------------------------------------------------
    for col in gene_cols:
        if heatmap_data[col].dtype == "object":
            extracted = heatmap_data[col].astype(str).str.extract(rstring, expand=False)

            heatmap_data[col] = pd.to_numeric(
                extracted,
                errors="coerce",
            )
        else:
            heatmap_data[col] = pd.to_numeric(
                heatmap_data[col],
                errors="coerce",
            )

    variables = heatmap_data["variable"].unique()

    # ---------------------------------------------------------
    # Find genes actually containing data for each variable
    # ---------------------------------------------------------
    variable_gene_data = {}

    for variable in variables:
        data = heatmap_data[heatmap_data["variable"] == variable]

        genes = [gene for gene in gene_cols if data[gene].notna().any()]

        variable_gene_data[variable] = genes

    # Height proportional to number of genes
    panel_heights = [
        max(len(variable_gene_data[variable]), 1) for variable in variables
    ]

    # ---------------------------------------------------------
    # Figure + GridSpec
    #
    # Column 0: colourbar
    # Column 1: heatmaps
    # ---------------------------------------------------------
    fig = plt.figure(figsize=FIGSIZE_PORTRAIT)

    gs = gridspec.GridSpec(
        nrows=len(variables),
        ncols=2,
        figure=fig,
        width_ratios=[0.02, 1],
        height_ratios=panel_heights,
        wspace=0.3,
        hspace=0.12,
    )

    # Dedicated axis for the single shared colourbar
    cbar_ax = fig.add_subplot(gs[:, 0])

    # ---------------------------------------------------------
    # Shared colour scale
    # ---------------------------------------------------------
    cmap = plt.get_cmap("flare")

    if x_col == ORIGIN_COL:
        vmax = heatmap_data[gene_cols].max().max()
    elif x_col == CLUSTER_COL:
        vmax = 100

    norm = plt.Normalize(vmin=0, vmax=vmax)

    # Keep reference to the last heatmap
    last_ax = None

    # ---------------------------------------------------------
    # Plot each variable as a vertically stacked heatmap
    # ---------------------------------------------------------
    for row_idx, variable in enumerate(variables):

        ax = fig.add_subplot(gs[row_idx, 1])
        last_ax = ax

        data = heatmap_data[heatmap_data["variable"] == variable].copy()

        genes = variable_gene_data[variable]

        # Keep only relevant genes and cluster
        data = data[[x_col] + genes]

        # Cluster -> x-axis
        # Gene -> y-axis
        data = data.set_index(x_col).T

        # -----------------------------------------------------
        # Make zero annotations blank
        # -----------------------------------------------------
        annot_data = (
            data.map(lambda x: "" if pd.isna(x) or x == 0 else f"{int(x)}")
            if annotate
            else None
        )

        # Don't display zeros as coloured cells
        data = data.replace({0: np.nan})

        sns.heatmap(
            data,
            annot=annot_data,
            annot_kws={"fontsize": 5},
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        ax.set_title(
            variable.upper(),
            loc="left",
            fontweight="bold",
        )

        # -----------------------------------------------------
        # Explicitly set ALL tick positions/labels
        # -----------------------------------------------------
        ax.set_xticks(np.arange(data.shape[1]) + 0.5)
        ax.set_xticklabels(
            data.columns,
            rotation=90,
            fontsize=6,
        )

        ax.set_yticks(np.arange(data.shape[0]) + 0.5)
        ax.set_yticklabels(
            data.index,
            rotation=0,
            fontsize=5,
        )

        ax.set_ylabel("")

        # -----------------------------------------------------
        # X-axis
        # Only show x-ticks on the bottom heatmap
        # -----------------------------------------------------
        if row_idx == len(variables) - 1:
            ax.set_xlabel(xlabel)
            ax.tick_params(
                axis="x",
                bottom=True,
                labelbottom=True,
                labelsize=6,
            )
        else:
            ax.set_xlabel("")
            ax.tick_params(
                axis="x",
                bottom=False,
                labelbottom=False,
            )

    # ---------------------------------------------------------
    # Single shared colourbar
    # ---------------------------------------------------------
    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=9,
    )

    cbar.ax.tick_params(
        labelsize=6,
    )

    # Optional: put colourbar ticks on the left side
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    fig.savefig(
        outpath,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def spillover_summary(df_in: pd.DataFrame, main_figure: bool):
    df = df_in.copy()

    variables = [
        "amr",
        "virulence",
        "metal",
        "biocide",
    ]

    cluster_order = sorted(df[CLUSTER_COL].dropna().unique())

    variable_gene_data = {}
    for variable in variables:
        data = df[df["gene_function"] == variable]
        variable_gene_data[variable] = data["gene"].drop_duplicates().tolist()

    panel_heights = [
        max(len(variable_gene_data[variable]), 1) for variable in variables
    ]

    fig = plt.figure(figsize=FIGSIZE_PORTRAIT)

    gs = gridspec.GridSpec(
        nrows=len(variables),
        ncols=2,
        figure=fig,
        width_ratios=[0.02, 1],
        height_ratios=panel_heights,
        wspace=0.3,
        hspace=0.12,
    )

    # -----------------------------------------------------------
    # Split the colourbar column into a colourbar (top) and a
    # legend (bottom), so the colourbar doesn't span the full
    # figure height.
    # -----------------------------------------------------------
    cbar_col_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[:, 0],
        height_ratios=[3, 1],
        hspace=0.15,
    )
    cbar_ax = fig.add_subplot(cbar_col_gs[0])
    legend_ax = fig.add_subplot(cbar_col_gs[1])
    legend_ax.axis("off")

    cmap = plt.get_cmap("flare")
    norm = plt.Normalize(vmin=0, vmax=100)

    # -----------------------------------------------------------
    # Marker style definitions (single source of truth, also
    # used to build the legend)
    # -----------------------------------------------------------
    MARKER_STYLES = {
        "category_restricted": dict(
            marker="s", c="#B0AEAE", edgecolors="black", label="Category\nrestricted"
        ),
        "novel_cluster_introduction": dict(
            marker="o", c="#FF3131", edgecolors="k", label="Novel cluster\nintroduction"
        ),
        "novel_gene_introduction": dict(
            marker="^", c="#5BFF3A", edgecolors="k", label="Novel gene\nintroduction"
        ),
    }

    for row_idx, variable in enumerate(variables):

        ax = fig.add_subplot(gs[row_idx, 1])

        data = df[df["gene_function"] == variable].copy()

        heatmap_data = data.pivot(
            index="gene",
            columns=CLUSTER_COL,
            values="percent_present",
        )

        heatmap_data = heatmap_data.reindex(columns=cluster_order)

        if not heatmap_data.empty:
            gene_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
            heatmap_data = heatmap_data.loc[gene_order]

        heatmap_plot = heatmap_data.replace({0: np.nan})

        sns.heatmap(
            heatmap_plot,
            cmap=cmap,
            vmin=0,
            vmax=100,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        ax.set_title(variable.upper(), loc="left", fontweight="bold")

        ax.set_xticks(np.arange(len(cluster_order)) + 0.5)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=6)

        ax.set_yticks(np.arange(heatmap_data.shape[0]) + 0.5)
        ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=5)
        ax.set_ylabel("")

        if row_idx == len(variables) - 1:
            ax.set_xlabel("Plasmid Cluster")
            ax.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=6)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # =====================================================
        # Overlay spillover sub-type / category restriction
        # =====================================================

        marker_data = (
            data[
                [
                    "gene",
                    CLUSTER_COL,
                    "spillover",
                    "category_restricted",
                    "novel_cluster_introduction",
                    "novel_gene_introduction",
                ]
            ]
            .drop_duplicates(subset=["gene", CLUSTER_COL])
            .set_index(["gene", CLUSTER_COL])
        )

        for i, gene in enumerate(heatmap_data.index):

            for j, cluster in enumerate(cluster_order):

                key = (gene, cluster)

                if key not in marker_data.index:
                    continue

                cell = marker_data.loc[key]

                # ---------------------------------------------
                # Priority: category restriction > novel cluster
                # introduction > novel gene introduction.
                # Plain, established spillover (no novel signal
                # in any category) gets no marker — the heatmap
                # colour alone is enough for that case.
                # ---------------------------------------------
                if cell["category_restricted"]:
                    style_key = "category_restricted"
                elif cell["spillover"] and cell["novel_cluster_introduction"]:
                    style_key = "novel_cluster_introduction"
                elif cell["spillover"] and cell["novel_gene_introduction"]:
                    style_key = "novel_gene_introduction"
                else:
                    continue

                style = MARKER_STYLES[style_key]

                ax.scatter(
                    j + 0.5,
                    i + 0.5,
                    s=15,
                    c=style["c"],
                    marker=style["marker"],
                    edgecolors=style["edgecolors"],
                    linewidth=0.4,
                    zorder=10,
                    alpha=0.9,
                )

    # =========================================================
    # Shared colourbar (top portion of column 0)
    # =========================================================

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Percentage of plasmids", fontsize=9)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    # =========================================================
    # Legend (bottom portion of column 0)
    # =========================================================

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["c"],
            markeredgecolor=style["edgecolors"],
            markeredgewidth=0.5,
            markersize=6,
            label=style["label"],
        )
        for style in MARKER_STYLES.values()
    ]

    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        fontsize=6,
        handletextpad=0.5,
        labelspacing=1.0,
    )

    # =========================================================
    # Save
    # =========================================================
    if main_figure:
        outpath = "results/figures/figure5_gene_spillover_ST.png"
    else:
        outpath = "results/figures/figureS2_gene_spillover_origin.png"

    fig.savefig(
        outpath,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


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
    ax2.legend(loc="upper left")

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
    ax3.legend(loc="upper left")

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
    df[origin_col] = df[origin_col].replace(
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

    df[origin_col] = pd.Categorical(
        df[origin_col],
        categories=origin_categories,
        ordered=True,
    )

    sns.histplot(
        df,
        x=cluster_col,
        stat="percent",
        hue=origin_col,
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

    common_STs = isolate_st_counts[isolate_st_counts >= 25].index

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
    amr_categories = list(range(0, df["amr_count"].max() + 1))

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

    axs[3].set(xlabel="Cluster")

    # ---------------------------------------------------------
    # Manual legends
    # ---------------------------------------------------------

    # Origin: one row
    add_manual_legend(
        axs[1],
        origin_palette,
        origin_categories,
        title="Epidemiological Compartment",
        ncol=5,
    )

    # Isolate ST: compact multi-row legend
    add_manual_legend(
        axs[2],
        st_palette,
        st_categories,
        title="Isolate ST",
        ncol=14,
    )

    # ARG count: one row
    add_manual_legend(
        axs[3],
        amr_palette,
        amr_categories,
        title="ARG count",
        ncol=12,
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
        hspace=0.32,
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
    path_to_tree1: str,
    path_to_tree2: str,
    df_in: pd.DataFrame,
    ax: Axes,
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
    colors = [
        "#1E90FF",  # DodgerBlue
        "#FFD700",  # Gold
        "#800080",  # Purple
        "#F08080",  # LightCoral
        "#000000",  # Black
        "#20B2AA",  # LightSeaGreen
        "#DC143C",  # Crimson
        "#154273",
        "#ADFF2F",  # GreenYellow
        "#FFA500",  # Orange
        "#007bc7",
        "#800000",  # Maroon
        "#90EE90",  # LightGreen
        "#4B0082",  # Indigo
        "#00CED1",  # DarkTurquoise
        "#275937",
        "#FF1493",  # DeepPink
        "#673327",
        "#0000FF",  # Blue
        "#552c6f",
        "#7FFF00",  # Chartreuse
        "#008000",  # Green
        "#f092cd",
        "#000080",  # Navy
        "#A52A2A",  # Brown
        "#76d2b6",
        "#00FF00",  # Lime
        "#94710a",
        "#B22222",  # FireBrick
        "#a90061",
        "#7c796f",
        "#FFFF00",  # Yellow
        "#1baa62",
        "#FFC0CB",  # Pink
        "#00FFFF",  # Cyan
        "#777b00",
        "#87CEFA",  # LightSkyBlue
        "#d52b1e",
        # "#808080",  # Gray
        "#DAA520",  # GoldenRod
        "#008080",  # Teal
    ]

    tangle_colours = config.CLONALITY_PALETTE
    df["Parent"] = df["Parent"].astype(str)
    # Generate the dictionary containing the colours for each point
    if palette == "":
        colours_bin = colors
        list_index = (len(df[node_left].unique()) - 1) % 40
        colours_bin[list_index] = "#C2C2C2"
        tree1_colour_dict, tree1_legend = generate_colour_dict(
            df,
            node_left,
            colors,
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
            config.MOBILITY_PALETTE,
            "Plasmid",
        )

        tree2_outline_dict, tree2_outline_legend = generate_colour_dict(
            df,
            outline_right,
            config.ORIGIN_PALETTE_SHORT,
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
            df, tangles, tangle_colours
        )

    # Load trees
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
        linewidth=3,
        zorder=100,
    )

    # Create the tangles
    # Build lookup of outlier isolates
    if outlier_only:
        outlier_names = set(
            df.loc[
                df[node_left] != "unbinned", "Plasmid"
            ]  # or whatever key matches k.name
        )
    else:
        outlier_names = set(df["Plasmid"])

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
            bbox_to_anchor=(-0.08, 1.02),
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
            bbox_to_anchor=(1.08, 1.02),
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
        elif len(tangle_legend.values()) == 0:
            print("")
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

    tangle_df = tangle_df.rename(
        columns={
            tangle: "Group Host Lineages",
            left_node: "Group index",
            right_node: "Isolate ST",
        }
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_PORTRAIT)
    plot_tanglegram_full(
        f"output/mashtree/{cluster_id}/{cluster_id}_tree.dnd",
        f"output/mashtree/{cluster_id}/{cluster_id}_chr_tree.dnd",
        tangle_df,
        ax,
        "Group index",
        "Isolate ST",
        "Group Host Lineages",
        # outline=True,
        # outline_left=outline_left,
        # outline_right=outline_right,
        outlier_only=True,
    )
    plt.savefig(
        f"results/trees/outlier_tangles/{cluster_id}.png",
        # bbox_inches="tight",
        dpi=600,
    )
    plt.clf()
