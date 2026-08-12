import pandas as pd
import os


def qc_set():
    bn_metadata = pd.read_csv("data/mrsa_metadata.csv", sep=";")
    print(bn_metadata)


if __name__ == "__main__":
    qc_set()
