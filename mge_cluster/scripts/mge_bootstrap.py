## Imports
import os
import subprocess
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import set_start_method, Pool
from typing import Literal
from sklearn.metrics.cluster import adjusted_rand_score


def lsf_hpcify_cmd(
    bash_string: str, log_file: str, threads: int = 6, mem: int = 6, time: int = 15
) -> str:
    """Function that takes an input string and prefixes all the LSF stuff
    This way the function can be easily commented out when shared outside
    the RIVM (or even IDS, since idk their queue and stuff).

    Parameters
    ----------
    bash_string : str
        bash command to run
    log_file : str
        path/to/log/file.log
    threads : int, optional
        threads to use, by default 6
    mem : int, optional
        memory to use in G, by default 6
    time : int, optional
        maximum runtime in minutes, by default 15

    Returns
    -------
    subprocess_cmd : str
        Input string with all the requested LSF configuration prefixed
    """

    hpc_string = (
        f"bsub -q bio -K -oo {log_file} -n {threads} -R 'rusage[mem={mem}G]' -W {time}"
    )
    list_string = [hpc_string, bash_string]
    subprocess_cmd = " ".join(list_string)
    return subprocess_cmd


def create_input_file():
    fastas_dir = os.listdir("fastas")
    fastas = [x for x in fastas_dir if x.endswith(".fasta")]
    # metadata_df = pd.read_csv("../data/mrsa_plasmid_data.csv", sep=";")
    # short_pls = metadata_df.loc[metadata_df["length"] >= 1000]
    # fastas = [x for x in fastas if x.split(".fasta")[0] in short_pls["Plasmid"].values]
    fastas = [f".fastas/{x}" for x in fastas]
    # rest = os.listdir("../gplas_bins")
    # rest = [f"../gplas_bins/{x}" for x in rest]
    # fastas += rest
    with open("mge_input.txt", "w") as f:
        for fasta in fastas:
            f.write(f"{fasta}" + "\n")


def generate_mge_cmd(
    boot_param: Literal["perplexity", "clustersize"], boot_value: int
) -> str | None:
    """
    Generate a single bash command for mge_cluster, unless the expected output
    file already exists. If the output exists, return None to skip rerunning.
    """

    mge_cmd_base = "mge_cluster --create --input mge_input.txt --threads 8"

    # --- Set parameter and directories ---
    if boot_param == "perplexity":
        param = f"--perplexity {boot_value}"
        sub_name = "perplex"
        outdir = f"{boot_param}/{sub_name}_{boot_value}/"
    elif boot_param == "clustersize":
        param = f"--min_cluster {boot_value}"
        sub_name = "cs"
        outdir = f"{boot_param}/{sub_name}_{boot_value}/"
    else:
        print("Incorrect parameter chosen for bootstrap.")
        exit()

    # --- Output file to check ---
    final_output = os.path.join(outdir, "mge-cluster_results.csv")

    # --- Skip if final output already exists ---
    if os.path.exists(final_output):
        print(f"Skipping {outdir}: results already exist.")
        return None

    # --- Ensure directory exists ---
    os.makedirs(outdir, exist_ok=True)

    # --- Construct command ---
    mge_cmd = f"{mge_cmd_base} {param} --outdir {outdir}"

    # mge_cmd = lsf_hpcify_cmd(
    #     mge_cmd, f"../logs/mge/mge_{boot_param}_{boot_value}.log", 12, 100, 120
    # )

    return mge_cmd


def bootstrap_mge_cluster(
    per_base: int = 30,
    per_max: int = 400,
    per_inc: int = 5,
    cs_base: int = 5,
    cs_max: int = 50,
    cs_inc: int = 1,
):
    ## Definie variables
    # Perplexity
    perplexity_base = per_base
    perplexity_increment = per_inc
    perplexity_max = per_max
    perplexity_range = list(
        range(perplexity_base, perplexity_max + 1, perplexity_increment)
    )
    print(perplexity_range)
    # Clustersize
    clustersize_base = cs_base
    clustersize_increment = cs_inc
    clusterize_max = cs_max  # <- 50
    clustersize_range = list(
        range(clustersize_base, clusterize_max + 1, clustersize_increment)
    )
    print(clustersize_range)

    # Generate commandline strings
    mge_cmd_cs = [
        cmd for x in clustersize_range if (cmd := generate_mge_cmd("clustersize", x))
    ]

    mge_cmd_plex = [
        cmd for x in perplexity_range if (cmd := generate_mge_cmd("perplexity", x))
    ]

    mge_cmds = mge_cmd_cs + mge_cmd_plex

    # Run bootstrap iteration (in parallel)
    set_start_method("spawn")
    n_jobs = 50
    worker = partial(subprocess.call, shell=True, stderr=subprocess.STDOUT)
    with Pool(processes=n_jobs, maxtasksperchild=1) as pool:
        pool.map(worker, mge_cmds)
    return None


def open_and_mark(input_file_path: str, marker: str) -> pd.DataFrame:
    """
    Function to open a mge clustering file and mark it with the bootstrap name

    Parameters
    ----------
    input_file_path : str
        path/to/clustering/file
    marker : str
        marker to add

    Returns
    -------
    pd.DataFrame
        df containing clustering column marked by bootstrap name
    """
    df = pd.read_csv(f"{input_file_path}", sep=",")
    df.rename(columns={"Standard_Cluster": f"{marker}"}, inplace=True)
    df.set_index("Sample_Name", inplace=True)
    return df[f"{marker}"]


def bootstrap_rand(
    parameter: Literal["perplexity", "clustersize"],
    # key_col: str,
    # cluster_col: str,
):
    """function to calculate the rand index for the bootstrap of a parameter value

    Parameters
    ----------
    input_path : str
        path/to/input/dir/
    filename : str
        name of the file containing the cluster
    key_col : str
        column name containing the key/seq names
    cluster_col : str
        column name where the clusters are stored
    """
    # Open clustering files and concat to single df
    boots = os.listdir(parameter)
    file_paths = [f"{parameter}/{x}/mge-cluster_results.csv" for x in boots]
    dfs = [open_and_mark(x, x.split("/")[1].split("_")[-1]) for x in file_paths]
    df = pd.concat(dfs, axis=1)

    # Create empty df to store the ARI values
    methods = df.columns
    ari_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    # Fill in ARI matrix
    for m1, m2 in itertools.product(methods, methods):
        ari_matrix.loc[m1, m2] = adjusted_rand_score(df[m1], df[m2])

    # Extract mean best ARI and parameter
    ari_no_diag = ari_matrix.copy()
    np.fill_diagonal(ari_no_diag.values, np.nan)
    mean_ari = ari_no_diag.mean()
    best_parameter = mean_ari.idxmax()

    # Sort the dataframe to create correct order for the plot
    df_sorted = ari_matrix.reindex(
        sorted(ari_matrix.index, key=lambda s: int(s)), axis=0
    )
    df_sorted = df_sorted.reindex(
        sorted(df_sorted.columns, key=lambda s: int(s)), axis=1
    )

    # Plot the heatmap
    sns.set_theme(font_scale=0.7)
    sns.heatmap(
        df_sorted,
        cmap=sns.color_palette("magma", as_cmap=True),
        vmin=0,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        square=True,
        linecolor="white",
        linewidth="0.5",
    ).set(title=f"{parameter}")
    plt.savefig(f"{parameter}_ARI.png")
    plt.clf()
    ari_matrix.to_csv(f"{parameter}_ARI.csv", sep=";")
    return best_parameter


def optimised_mge(perplexity, clustersize):
    base = "mge_cluster --create --input mge_input.txt --threads 8 --outdir final_model"
    params = f"--min_cluster {clustersize} --perplexity {perplexity}"
    if not os.path.exists("final_model/"):
        os.makedirs("final_model")
    cmd = f"{base} {params}"
    # cmd = lsf_hpcify_cmd(
    #     f"{cmd}", f"../logs/mge/mge_optimised.log", 12, 100, 120
    # )
    subprocess.call(f"{cmd}", shell=True, stderr=subprocess.STDOUT)
