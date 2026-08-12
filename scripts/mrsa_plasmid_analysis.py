import pandas as pd
import os

from mge_cluster.scripts.main import main as mge_main
from helper_functions import clean_plasmid_df
import population_description


def mrsa_plasmid_analysis():
    # ---------------------------------------------------------
    # 1. Verify fastas + metadata
    # ---------------------------------------------------------
    # Collect metadata and fasta sample lists
    metadata_df = pd.read_csv("data/metadata.csv", encoding="ISO-8859-1", sep=";")
    fastas = os.listdir("fastas")

    # Remove extension
    files = [int(x).split(".fasta")[0] for x in fastas]

    # Find missing items between the lists
    missing_fastas = [x for x in fastas if x not in metadata_df["Plasmid"].values]
    missing_metadata = [x for x in metadata_df["Plasmid"].values if x not in files]

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
    mge_main()

    # ---------------------------------------------------------
    # 3. Population description
    # ---------------------------------------------------------
    clustering_df = pd.read_csv("data/clustering.csv")
    plasmid_df = clean_plasmid_df(metadata_df, clustering_df)
    population_description.dataset_overview(plasmid_df)
    population_description.cluster_overview(plasmid_df)

    # ---------------------------------------------------------
    # 4. Scheme / cluster-level description
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 5. Nearly-identical plasmid analysis
    # ---------------------------------------------------------


if __name__ == "__main__":
    mrsa_plasmid_analysis()
