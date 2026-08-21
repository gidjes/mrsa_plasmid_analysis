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
    1. Verify all required files exist
    2. Mge-cluster bootstrap to create scheme
    3. General population statistics/description
    4. Cluster statistics/description
    5. Nearly-identical neighbour analysis
    """
    # ---------------------------------------------------------
    # 1. Verify fastas + metadata
    # ---------------------------------------------------------
    # Collect metadata and fasta sample lists
    metadata_df = pd.read_csv("data/metadata.csv", encoding="ISO-8859-1", sep=";")
    fastas = os.listdir("fastas")
    iso_fastas = os.listdir("fastas_chr")

    # Remove extension
    files = [int(x).split(".fasta")[0] for x in fastas]
    iso_files = [int(x).split(".fasta")[0] for x in iso_fastas]

    # Find missing items between the lists
    missing_fastas = [x for x in fastas if x not in metadata_df["Plasmid"].values]
    missing_fastas += [x for x in iso_fastas if x not in metadata_df["Parent"].values]

    missing_metadata = [x for x in metadata_df["Plasmid"].values if x not in files]
    missing_metadata += [x for x in metadata_df["Parent"].values if x not in iso_files]

    # Report completeness
    if len(missing_fastas) > 0:
        print("All fasta files have corresponding metadata")
        print("\n")
    else:
        print(f"Missing metadata for {len(missing_fastas)} files:")
        print(missing_fastas)
        print("\n")

    if len(missing_metadata) > 0:
        print("All metadata have corresponding fasta files")
        print("\n")
    else:
        print(f"Missing fastas for {len(missing_metadata)} plasmids:")
        print(missing_metadata)
        print("\n")

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
        metadata_df, left_on="Parent", right_on="Isolate_ID", how="left"
    )
    clustering_df = pd.read_csv("mge_bootstrap/final_model/mge-cluster_results.csv")
    plasmid_df = clean_plasmid_df(metadata_df, clustering_df)

    # Run population-level analysis
    population_description.dataset_overview(plasmid_df)

    # ---------------------------------------------------------
    # 4. Scheme / cluster-level description
    # ---------------------------------------------------------
    # Run cluster-level analysis
    population_description.cluster_overview(plasmid_df)

    # ---------------------------------------------------------
    # 5. Nearly-identical plasmid analysis
    # ---------------------------------------------------------
    # Calculate mash distances and create phylogenetic trees
    nearest_neighbour_analysis.mashtree_builder(plasmid_df, NJOBS, CLUSTER_COL)

    # Run sequence-level analysis
    bin_dynamics.bin_post_hoc(plasmid_df)


if __name__ == "__main__":
    set_start_method("spawn")
    mrsa_plasmid_analysis()
