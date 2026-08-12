import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import geopandas as gp
import os
import config


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


def isolate_data(df_in, geo_df_in, map_boxes, origin_col: str = config.ORIGIN_COL):
    species_col = config.SPECIES_COL
    origin_col = config.ORIGIN_COL

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
        palette=config.SPECIES_PALETTE,
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
        palette=config.ORIGIN_PALETTE_FULL,
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
    origin_col: str = config.ORIGIN_COL,
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
    import numpy as np
    from matplotlib.patches import Ellipse

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
    cluster_col: str = config.CLUSTER_COL,
    tsne1D: str = config.TSNE1D,
    tsne2D: str = config.TSNE2D,
):
    df = df_in.copy()
    origin_col = config.ORIGIN_COL

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
        palette=config.ORIGIN_PALETTE,
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
    cluster_col: str = config.CLUSTER_COL,
    origin_col: str = config.ORIGIN_COL,
    st_col: str = config.ST_COL,
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

    origin_palette = config.ORIGIN_PALETTE

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
