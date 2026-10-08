# ==============================================================================
# Cluster tree visualisation
#
# Draws one annotated phylogenetic tree for a given cluster, with:
#
#   * Tree tips  coloured by REPLICON
#   * Block 1    Sequence Type (ST)      - one column per ST, tile shows
#                                           presence/absence
#   * Block 2    Functional genes        - one column per gene, pooled from
#                                           the amr / virulence / metal /
#                                           biocide columns, tile coloured by
#                                           which of those columns the gene
#                                           came from
#   * Block 3    Origin                  - one column per origin, tile
#                                           coloured by origin
#
# The tree file for a cluster is expected at: mashtree/{cluster}_tree.dnd
#
# USAGE
# -----
# From the command line:
#   Rscript plot_cluster_tree.R 56
#
# From an interactive R session / another script:
#   source("plot_cluster_tree.R")
#   plasmid_df <- load_plasmid_data("data/plasmid_data.csv")
#   plot_cluster_tree("56", plasmid_df)
#
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. PACKAGES
# ------------------------------------------------------------------------------
# ggtree/ggtreeExtra/ggstar/ggnewscale drive the tree + ring-annotation plot,
# dplyr/tidyr do the data wrangling, colorspace/scales help build palettes.

required_packages <- c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "ape",
  "ggtree",
  "ggtreeExtra",
  "ggstar",
  "ggnewscale",
  "ggtext",
  "svglite",
  "scales",
  "colorspace"
)

install_missing_packages <- function(packages) {
  installed <- rownames(installed.packages())
  missing <- setdiff(packages, installed)
  for (pkg in missing) {
    message(sprintf("Installing missing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

install_missing_packages(required_packages)
invisible(lapply(required_packages, library, character.only = TRUE))

# ------------------------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------------------------
# Base colours used to build every discrete palette in the plot. Edit here to
# re-theme the whole figure.

categorical_colours <- c(
  "#a90061", "#275937", "#007bc7", "#f9e11e", "#d52b1e",
  "#777b00", "#76d2b6", "#673327", "#552c6f", "#f092cd",
  "#154273", "#94710a", "#7c796f", "#1baa62"
)

# Column names in plasmid_df that feed each part of the figure. Change these
# if the new project's data uses different column names - nothing below the
# CONFIGURATION section needs to change.
DEFAULT_COLUMNS <- list(
  id       = "Plasmid", # unique identifier that matches the tree tip labels
  cluster  = "Standard_Cluster_mrsa", # column used to subset plasmid_df to one cluster
  st       = "ISOLATE_TL_MLST_ST", # sequence type column
  genes    = c("amr", "virulence", "metal", "biocide"), # gene-presence columns (comma-separated lists)
  origin   = "origin", # sample origin column
  replicon = "replicon" # tip-level replicon column (comma-separated list allowed)
)


# ------------------------------------------------------------------------------
# 3. COLOUR HELPERS
# ------------------------------------------------------------------------------

#' Interpolate `n` colours out of a small base colour list.
discrete_palette <- function(n, color_list) {
  colorRampPalette(color_list)(n)
}

#' ggplot2 discrete colour scale built from discrete_palette().
scale_color_custom_discrete <- function(n, color_list) {
  discrete_scale("color", palette = function(n) discrete_palette(n, color_list))
}

#' ggplot2 discrete fill scale built from discrete_palette().
scale_fill_custom_discrete <- function(n, color_list) {
  discrete_scale("fill", palette = function(n) discrete_palette(n, color_list))
}

#' Two-level hierarchical palette: each main category gets a base colour from
#' `colors`, and its subcategories get progressively lighter shades of that
#' base colour. Handy for e.g. province -> municipality colour schemes.
#' Kept as a general-purpose utility; not used by the default figure below.
custom_hierarchical_palette <- function(df, main_categories_name, subcategories_name, colors) {
  families <- split(df[, subcategories_name], df[, main_categories_name])
  subcategories <- lapply(families, unique)

  main_categories <- sort(unique(df[, main_categories_name]))
  main_palette <- colorRampPalette(colors)(length(main_categories))
  category_colors <- c()

  for (i in seq_along(main_categories)) {
    base_color <- main_palette[i]
    subs <- subcategories[[main_categories[i]]]
    n_subs <- length(subs)
    if (n_subs > 1) {
      shade_vals <- seq(0, 0.5, length.out = n_subs)
      sub_palette <- sapply(shade_vals, function(x) lighten(base_color, x))
    } else {
      sub_palette <- base_color
    }
    category_colors <- c(category_colors, setNames(sub_palette, subs))
  }

  category_colors
}


# ------------------------------------------------------------------------------
# 4. ANNOTATION-MATRIX BUILDERS
# ------------------------------------------------------------------------------
# geom_fruit() wants long-format data: one row per (id, category) pair that
# is "present". Each unique value of `category` becomes its own ring column
# in the plot. These helpers turn a wide metadata column (optionally a
# comma-separated list, e.g. multiple replicons/genes per row) into that
# long format.

#' Turn one categorical column into a long presence table.
#'
#' @param df             source data frame (e.g. plasmid_df)
#' @param id_col         column matching the tree tip labels
#' @param category_col   column to expand into one-category-per-column tiles
#' @param sep             if not NULL, split category_col on this separator
#'                        first (e.g. "," for comma-separated gene lists)
#' @param category_label value to store in the `source` column, useful when
#'                        combining several category_cols together (see
#'                        build_gene_matrix())
build_presence_matrix <- function(df, id_col, category_col, sep = NULL,
                                  category_label = category_col) {
  data <- df[, c(id_col, category_col)]
  colnames(data) <- c("id", "category")
  data <- data[!is.na(data$category) & data$category != "", , drop = FALSE]

  if (!is.null(sep)) {
    data <- tidyr::separate_rows(data, category, sep = sep)
  }

  data$category <- trimws(data$category)
  data <- data[data$category != "", , drop = FALSE]
  data <- unique(data)

  data$present <- "Present"
  data$source <- category_label
  as.data.frame(data)
}

#' Pool several gene-presence columns (amr, virulence, metal, biocide, ...)
#' into one long table with one row per (id, gene), tagging each gene with
#' the column it came from (`gene_type`) so the tile block can be coloured
#' by gene category.
build_gene_matrix <- function(df, id_col, gene_cols, sep = ",") {
  pieces <- lapply(gene_cols, function(col) {
    build_presence_matrix(df, id_col, col, sep = sep, category_label = col)
  })
  out <- do.call(rbind, pieces)
  colnames(out)[colnames(out) == "category"] <- "gene"
  colnames(out)[colnames(out) == "source"] <- "gene_type"

  out$gene_type <- factor(out$gene_type, levels = gene_cols)

  gene_levels <- unique(out$gene[order(out$gene_type, out$gene)])
  out$gene <- factor(out$gene, levels = gene_levels)

  out
}

#' Collapse a possibly comma-separated, possibly multi-row column down to a
#' single display value per id (e.g. several replicons per isolate joined
#' into one string for the tip colour/legend).
collapse_to_single_value <- function(df, id_col, value_col, sep = ",") {
  data <- df[, c(id_col, value_col)]
  colnames(data) <- c("id", "value")
  data <- data[!is.na(data$value) & data$value != "", , drop = FALSE]
  data$value <- vapply(strsplit(data$value, sep), function(x) {
    paste(sort(unique(trimws(x))), collapse = ", ")
  }, character(1))
  data <- unique(data)
  colnames(data) <- c(id_col, value_col)
  data
}


# ------------------------------------------------------------------------------
# 5. DATA LOADING
# ------------------------------------------------------------------------------

#' Read the raw plasmid metadata table and add any derived columns shared by
#' every figure.
load_plasmid_data <- function(path, sep = ";") {
  plasmid_df <- read.csv(path, header = TRUE, sep = sep)
  plasmid_df$Genus <- sub(" .*$", "", plasmid_df$ISOLATE_TL_SPECIES)
  plasmid_df
}

# ------------------------------------------------------------------------------
# 6. HELPER FUNCTION FOR PLOTTING SPARSE DATA
# ------------------------------------------------------------------------------
# Handles three situations:
#
#   0 rows
#       -> leave unchanged; the caller skips geom_fruit()
#
#   1 row
#       -> duplicate the row with alpha = 0 so geom_fruit() does not calculate
#          var() on a single observation
#
#   >1 rows but only 1 unique x category
#       -> add an invisible dummy x category so geom_fruit() has a non-degenerate
#          x scale. The dummy is explicitly removed from the displayed axis.
#
add_geom_fruit_safety_data <- function(dat, x_col) {
  # ---- No annotation data ---------------------------------------------------
  if (nrow(dat) == 0) {
    dat$.geom_fruit_alpha <- numeric(0)
    dat$.geom_fruit_dummy <- logical(0)
    return(dat)
  }

  # Store original state
  dat$.geom_fruit_alpha <- 1
  dat$.geom_fruit_dummy <- FALSE

  # Unique non-NA x values
  x_values <- unique(
    dat[[x_col]][!is.na(dat[[x_col]])]
  )

  # ---- Exactly one row or multiple rows but only one x category-------------
  # Add one dummy x category. It is marked explicitly so that the plotting
  # code can exclude it from the displayed axis and from pwidth calculations.
  #
  if (nrow(dat) == 1 || length(x_values) == 1) {
    dummy <- dat[1, , drop = FALSE]

    dummy[[x_col]] <- paste0(
      "__"
    )

    dummy$.geom_fruit_alpha <- 0
    dummy$.geom_fruit_dummy <- TRUE

    return(
      rbind(dat, dummy)
    )
  }


  # ---- Normal case ----------------------------------------------------------
  dat
}


# ------------------------------------------------------------------------------
# 7. MAIN PLOTTING FUNCTION
# ------------------------------------------------------------------------------

#' Build and save the annotated tree for one cluster.
#'
#' @param cluster     cluster identifier, e.g. "56". Used both to find the
#'                     tree file (mashtree/{cluster}_tree.dnd) and to subset
#'                     plasmid_df (columns$cluster == cluster).
#' @param plasmid_df  full metadata table (see load_plasmid_data())
#' @param columns     list of column-name overrides, see DEFAULT_COLUMNS
#' @param tree_dir    directory containing the *_tree.dnd files
#' @param output_dir  directory the svg/png outputs are written to
#' @param width,height,units  passed straight to ggsave()
plot_cluster_tree <- function(cluster,
                              plasmid_df,
                              columns = DEFAULT_COLUMNS,
                              tree_dir = "output/mashtree",
                              output_dir = "results/trees/annotated",
                              width = 210, height = 297, units = "mm") {
  id_col <- columns$id
  cluster_col <- columns$cluster
  st_col <- columns$st
  gene_cols <- columns$genes
  origin_col <- columns$origin
  replicon_col <- columns$replicon

  # ---- 6.1 Read the tree for this cluster ----------------------------------
  tree_path <- file.path(tree_dir, sprintf("%s", cluster), sprintf("%s_tree.dnd", cluster))
  message(sprintf("Reading tree: %s", tree_path))
  tree <- read.tree(tree_path)

  # ---- 6.2 Subset the metadata to this cluster ------------------------------
  cluster_df <- plasmid_df[plasmid_df[[cluster_col]] == cluster, , drop = FALSE]
  if (nrow(cluster_df) == 0) {
    stop(sprintf("No rows in plasmid_df where %s == '%s'", cluster_col, cluster))
  }

  # ---- 6.3 Attach tip-level replicon info directly onto the tree -----------
  # (so ggtree can map it with aes() without a further join downstream)
  replicon_lookup <- collapse_to_single_value(cluster_df, id_col, replicon_col)
  colnames(replicon_lookup)[colnames(replicon_lookup) == id_col] <- "label"
  tree <- dplyr::full_join(tree, replicon_lookup, by = "label")

  # ---- 6.4 Build the three ring-annotation blocks ---------------------------
  st_matrix <- build_presence_matrix(cluster_df, id_col, st_col)
  gene_matrix <- build_gene_matrix(cluster_df, id_col, gene_cols)
  origin_matrix <- build_presence_matrix(cluster_df, id_col, origin_col)
  st_matrix <- add_geom_fruit_safety_data(
    st_matrix,
    "category"
  )
  st_categories <- unique(
    st_matrix$category[
      !st_matrix$.geom_fruit_dummy
    ]
  )

  gene_matrix <- add_geom_fruit_safety_data(
    gene_matrix,
    "gene"
  )
  gene_categories <- unique(
    gene_matrix$gene[
      !gene_matrix$.geom_fruit_dummy
    ]
  )

  origin_matrix <- add_geom_fruit_safety_data(
    origin_matrix,
    "category"
  )
  origin_categories <- unique(
    origin_matrix$category[
      !origin_matrix$.geom_fruit_dummy
    ]
  )

  GENE_PALETTE <- c(
    "amr" = "#BB297A",
    "virulence" = "#56B4E9",
    "metal" = "#9B9B9B",
    "biocide" = "#009E73"
  )

  gene_palette <- GENE_PALETTE

  ORIGIN_PALETTE <- c(
    "LA-MRSA" = "#009E73",
    "HA-MRSA" = "#D55E00",
    "CA-MRSA" = "#0072B2",
    "MSSA" = "#737070",
    "Sar" = "#F0E442"
  )

  origin_palette <- ORIGIN_PALETTE

  n <- nrow(cluster_df)

  linewidth <- if (n <= 100) {
    1
  } else {
    1 - ((n - 100) / 700)^0.5 * 0.9
  }

  # ---- 6.5 Base tree, tips coloured by replicon -----------------------------
  tree_plot <- ggtree(tree) +
    geom_treescale(y = -2) +
    labs(title = "") +
    theme(
      panel.background   = element_blank(),
      panel.grid         = element_blank(),
      panel.border       = element_blank(),
      axis.line          = element_blank(),
      axis.ticks         = element_blank(),
      axis.text          = element_blank(),
      axis.title         = element_blank(),
      plot.title         = element_blank(),
      legend.background  = element_blank()
    )

  tree_tips <- tree_plot +
    geom_tippoint(aes(colour = .data[[replicon_col]]), size = 2.5, na.rm = TRUE) +
    guides(colour = guide_legend(ncol = 1, title = "Replicon", order = 1))

  # ---- 6.6 Block 1: Sequence Type (ST), one column per ST -------------------
  if (nrow(st_matrix) > 0) {
    tree_st <- tree_tips +
      new_scale_fill() +
      geom_fruit(
        data = st_matrix,
        geom = geom_tile,
        mapping = aes(
          y = id,
          x = category,
          fill = present,
          alpha = .geom_fruit_alpha
        ),
        axis.params = list(
          axis = "x",
          text.angle = -60,
          text.size = 3,
          line.size = 0.2,
          vjust = 0.5,
          hjust = 0,
          title = "ST",
          title.size = 4,
          title.height = 0.02
        ),
        size = linewidth,
        pwidth = max(2, length(st_categories)) / 10, ,
        offset = 0.12,
        color = "#FFFFFF",
        grid.params = list(
          linetype = 1,
          size = 0.2
        )
      ) +
      scale_fill_manual(
        values = c("Present" = "black"),
        na.translate = FALSE
      ) +
      scale_alpha_identity() +
      guides(
        fill = guide_legend(
          title = "Isolate ST",
          order = 2
        )
      )
  } else {
    tree_st <- tree_tips
  }

  # ---- 6.7 Block 2: functional genes, coloured by gene type -----------------
  if (nrow(gene_matrix) > 0) {
    tree_genes <- tree_st +
      new_scale_fill() +
      geom_fruit(
        data = gene_matrix,
        geom = geom_tile,
        mapping = aes(
          y = id,
          x = gene,
          fill = gene_type,
          alpha = .geom_fruit_alpha
        ),
        axis.params = list(
          axis = "x",
          text.angle = -60,
          text.size = 3,
          line.size = 0.2,
          vjust = 0.5,
          hjust = 0,
          title = "Genes",
          title.size = 4,
          title.height = 0.02
        ),
        size = linewidth,
        pwidth = max(2, length(gene_categories)) / 10,
        offset = 0.12,
        color = "#FFFFFF",
        grid.params = list(
          linetype = 1,
          size = 0.2
        )
      ) +
      scale_fill_manual(
        values = gene_palette,
        name = "Gene type"
      ) +
      scale_alpha_identity() +
      guides(
        fill = guide_legend(
          title = "Gene type",
          order = 3
        )
      )
  } else {
    tree_genes <- tree_st
  }

  # ---- 6.8 Block 3: origin, one column per origin, coloured by origin -------
  if (nrow(origin_matrix) > 0) {
    tree_full <- tree_genes +
      new_scale_fill() +
      geom_fruit(
        data = origin_matrix,
        geom = geom_tile,
        mapping = aes(
          y = id,
          x = category,
          fill = category,
          alpha = .geom_fruit_alpha
        ),
        axis.params = list(
          axis = "x",
          text.angle = -60,
          text.size = 3,
          line.size = 0.2,
          vjust = 0.5,
          hjust = 0,
          title = "Origin",
          title.size = 4,
          title.height = 0.02
        ),
        size = linewidth,
        pwidth = max(2, length(origin_categories)) / 10,
        offset = 0.12,
        color = "#FFFFFF",
        grid.params = list(
          linetype = 1,
          size = 0.2
        )
      ) +
      scale_fill_manual(
        values = origin_palette,
        name = "Origin",
        na.translate = FALSE
      ) +
      scale_alpha_identity() +
      guides(
        fill = guide_legend(
          title = "Origin",
          order = 4
        )
      )
  } else {
    tree_full <- tree_genes
  }

  tree_full <- tree_full +
    coord_cartesian(clip = "off") +
    theme(
      legend.text = element_markdown(size = 6),
      legend.title = element_text(size = 8),
      plot.margin = margin(
        t = 10,
        r = 10,
        b = 100,
        l = 10
      )
    )

  # ---- 6.9 Save outputs -------------------------------------------------------
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  svg_path <- file.path(output_dir, sprintf("%s_tree.svg", cluster))
  png_path <- file.path(output_dir, sprintf("%s_tree.png", cluster))

  # ggsave(svg_path, plot = tree_full, width = width, height = height, units = units)
  ggsave(png_path, plot = tree_full, width = width, height = height, units = units)
  message(sprintf("Saved: %s and %s", svg_path, png_path))

  if (cluster == "11") {
    png_path <- file.path("results/figures/figure6_cluster_11_tree.png")
    svg_path <- file.path("results/figures/figure6_cluster_11_tree.svg")
    ggsave(png_path, plot = tree_full, width = width, height = height, units = units)
    ggsave(svg_path, plot = tree_full, width = width, height = height, units = units)
  }

  tree_full
}



# ------------------------------------------------------------------------------
# 8. COMMAND-LINE ENTRY POINT
# ------------------------------------------------------------------------------
# Lets the script be run directly as:  Rscript plot_cluster_tree.R <cluster_id>
# Sourcing the file (e.g. `source("plot_cluster_tree.R")`) will NOT trigger
# this block, so you can safely source it and call plot_cluster_tree()
# yourself with custom arguments.
if (sys.nframe() == 0 && !interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    stop("Usage: Rscript scripts/annotated_tree.R <cluster_id>  e.g. Rscript scripts/annotated_tree.R 56")
  }
  cluster_id <- args[1]

  plasmid_df <- load_plasmid_data("data/merged_data.csv")
  plot_cluster_tree(cluster_id, plasmid_df)
}
