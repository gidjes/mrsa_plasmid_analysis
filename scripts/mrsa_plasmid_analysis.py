import pandas as pd
import os
from multiprocessing import set_start_method

import config
import mge_bootstrap
import population_description
import nearest_neighbour_analysis
import bin_dynamics
from helper_functions import clean_plasmid_df

NJOBS = config.NJOBS
CLUSTER_COL = config.CLUSTER_COL


def mrsa_plasmid_analysis():
    """
    Running function for all components included in the paper:
    1. Open data
    2. Mge-cluster bootstrap to create scheme
    3. General population statistics/description
    4. Cluster statistics/description
    5. Nearly-identical neighbour analysis
    """
    # ---------------------------------------------------------
    # 1. Open data
    # ---------------------------------------------------------
    # Collect metadata and fasta sample lists
    metadata_df = pd.read_csv("data/metadata.csv", encoding="ISO-8859-1", sep=";")

    # Make necessary output directories
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # ---------------------------------------------------------
    # 2. mge-bootstrap -> create scheme
    # ---------------------------------------------------------
    # Make required directories
    os.makedirs("logs/mge/", exist_ok=True)
    os.makedirs("mge_bootstrap/", exist_ok=True)

    # Run the bootstrap optimiser
    mge_bootstrap.create_input_file()
    mge_bootstrap.bootstrap_mge_cluster()

    # Get optimised paramter values
    perplexity = mge_bootstrap.bootstrap_rand("perplexity")
    clustersize = mge_bootstrap.bootstrap_rand("clustersize")

    # Perform optimised mge-cluster
    mge_bootstrap.optimised_mge(perplexity, clustersize)

    # ---------------------------------------------------------
    # 3. Population description
    # ---------------------------------------------------------
    # Load and clean required files
    genome_df = pd.read_csv("data/genome_data.csv", sep=";")
    plasmid_df = genome_df.merge(
        metadata_df, left_on="Parent", right_on="KEY", how="left"
    )
    clustering_df = pd.read_csv(
        "output/mge_bootstrap/final_model/mge-cluster_results.csv"
    )
    plasmid_df = clean_plasmid_df(plasmid_df, clustering_df, True)
    metadata_df = clean_plasmid_df(metadata_df, clustering_df, False)
    plasmid_df.to_csv("data/merged_data.csv", sep=";", index=False)

    # Run population-level analysis
    population_description.dataset_overview(plasmid_df, metadata_df)

    # ---------------------------------------------------------
    # 4. Scheme / cluster-level description
    # ---------------------------------------------------------
    # Run cluster-level analysis
    population_description.cluster_overview(plasmid_df, metadata_df)

    # ---------------------------------------------------------
    # 5. Near-identical plasmid bin analysis
    # ---------------------------------------------------------
    # Calculate mash distances and create phylogenetic trees
    os.makedirs("output/mashtree", exist_ok=True)
    nearest_neighbour_analysis.mashtree_builder(plasmid_df)

    # Create the wgMLST distance files
    os.makedirs("output/wgmlst", exist_ok=True)
    nearest_neighbour_analysis.wgMLST_converter()
    nearest_neighbour_analysis.wgMLST_prepper(plasmid_df)

    # Calculate mash distances and create phylogenetic trees
    os.makedirs("results/trees/outlier_tangles/", exist_ok=True)
    nearest_neighbour_analysis.run_nn_analysis(plasmid_df)

    # Run sequence-level analysis
    os.makedirs("results/bin_dynamics", exist_ok=True)
    bin_dynamics.bin_post_hoc(plasmid_df)


if __name__ == "__main__":
    set_start_method("spawn")
    mrsa_plasmid_analysis()
