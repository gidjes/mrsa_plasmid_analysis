import pandas as pd
import config

# ---------------------------------------------------------
# Declare variables
# ---------------------------------------------------------
ORIGIN_COL = config.ORIGIN_COL
SPECIES_COL = config.SPECIES_COL
ST_COL = config.ST_COL
LA_COL = config.LA_COL
SUBMITTER_TYPE = config.SUBMITTER_TYPE
MEC_COL = config.MEC_COL

CITY_COL = config.CITY_COL
CITY_COL_BACKUP = config.CITY_COL_BACKUP
PROVINCE_COL = config.PROVINCE_COL
PROVINCE_COL_BACKUP = config.PROVINCE_COL_BACKUP
DATE_COL = config.DATE_COL


def infer_origin(df_in: pd.DataFrame) -> pd.DataFrame:
    # Default to 'MRSA'
    df = df_in.copy()
    df[ORIGIN_COL] = "CA-MRSA"

    # Update to 'LA-MRSA' where condition is met
    df.loc[df[LA_COL] == "LA-MRSA", ORIGIN_COL] = "LA-MRSA"
    df.loc[df[ST_COL] == "1232", ORIGIN_COL] = "CA-MRSA"
    df.loc[
        (df[SUBMITTER_TYPE] == "Ziekenhuis") & (df[ORIGIN_COL] != "LA-MRSA"),
        ORIGIN_COL,
    ] = "HA-MRSA"
    df.loc[df[MEC_COL] == "Negatief", ORIGIN_COL] = "MSSA"
    df.loc[df[SPECIES_COL] == "Staphylococcus schweitzeri", ORIGIN_COL] = "Sar"
    df.loc[df[SPECIES_COL] == "Staphylococcus argenteus", ORIGIN_COL] = "Sar"

    df.loc[df[ORIGIN_COL] == "Sar", ST_COL] = (
        df.loc[df[ORIGIN_COL] == "Sar", ST_COL] + "_sar"
    )

    return df


def attach_municipalities(df_in):
    df = df_in.copy()

    # Load geodata and clean
    NL_geo_key = pd.read_csv("data/WoonplaatsenCodes.csv", sep=";")
    NL_geo_key.drop(
        columns=[
            "Woonplaatscode",
            "Gemeente|Code ",
            "Provincie|Code",
            "Landsdeel|Naam",
            "Landsdeel|Code",
        ],
        inplace=True,
    )
    NL_geo_key.rename(
        columns={
            "Gemeente|Naam ": "municipality",
            "Woonplaatsen": "city",
            "Provincie|Naam": "province",
        },
        inplace=True,
    )
    NL_geo_key["municipality"] = NL_geo_key["municipality"].replace(
        " ", "_", regex=True
    )
    # merge with incoming data
    df = pd.merge(
        df,
        NL_geo_key,
        on=["city", "province"],
        how="left",
    )

    # Show cities that did not get a municipality
    unmatched = (
        df.loc[df["municipality"].isna(), ["city", "province"]]
        .drop_duplicates()
        .sort_values(["province", "city"])
    )

    print(f"{len(unmatched)} unique city/province combinations did not match:")
    print(unmatched.to_string(index=False))

    return df


def infer_region(row):
    city = row[CITY_COL]
    province = row[PROVINCE_COL]
    if city == "" or pd.isna(city):
        city = row[CITY_COL_BACKUP]
        province = row[PROVINCE_COL_BACKUP]
    return city, province


def clean_plasmid_df(
    df_meta: pd.DataFrame,
    df_clustering: pd.DataFrame,
    plasmid_level: bool = True,
) -> pd.DataFrame:
    """
    Funtion to clean up and fill up the MRSA plasmid data file

    Parameters
    ----------
    df_meta : pd.DataFrame
        DF with plasmid genomic and epi data
    df_clustering : pd.DataFrame
        DF with MRSA assigned clusters
    plasmid_level : bool
        Is the input DF at the plasmid level, by default True

    Returns
    -------
    pd.DataFrame
        DF with genomic, epi, and cluster data
    """
    df_meta[DATE_COL] = pd.to_datetime(
        df_meta[DATE_COL],
        dayfirst=True,
        errors="coerce",  # converts invalid/empty values to NaT
    )
    df_meta[ST_COL] = [
        str(int(float(x))) if pd.notna(x) else x for x in df_meta[ST_COL]
    ]

    # Fix the MRSA type column; origin
    df_origin_fixed = infer_origin(df_meta)

    # Fill empty region values (GPs) and create municipality
    df_origin_fixed[["city", "province"]] = df_origin_fixed.apply(
        infer_region, axis=1, result_type="expand"
    )
    df_geo_loc_fixed = attach_municipalities(df_origin_fixed)

    if plasmid_level:
        # Clean the small linear pieces
        df_thresh = df_geo_loc_fixed.loc[df_geo_loc_fixed["bp_length"] >= 1000].copy()

        # Add AMR_status and gene counts
        df_thresh.loc[:, "amr_plasmid"] = df_thresh["amr"].notna().astype(int)
        df_thresh.loc[:, "amr_count"] = df_thresh["amr"].apply(
            lambda x: 0 if pd.isna(x) else len(x.split(","))
        )
        df_thresh.loc[:, "virulence_plasmid"] = (
            df_thresh["virulence"].notna().astype(int)
        )
        df_thresh.loc[:, "virulence_count"] = df_thresh["virulence"].apply(
            lambda x: 0 if pd.isna(x) else len(x.split(","))
        )
        df_thresh.loc[:, "metal_plasmid"] = df_thresh["metal"].notna().astype(int)
        df_thresh.loc[:, "metal_count"] = df_thresh["metal"].apply(
            lambda x: 0 if pd.isna(x) else len(x.split(","))
        )
        df_thresh.loc[:, "biocide_plasmid"] = df_thresh["biocide"].notna().astype(int)
        df_thresh.loc[:, "biocide_count"] = df_thresh["biocide"].apply(
            lambda x: 0 if pd.isna(x) else len(x.split(","))
        )

        # Add the MRSA mge-clusters
        df_clustering = df_clustering.add_suffix("_mrsa")
        df_clustering = df_clustering.rename(columns={"Sample_Name_mrsa": "Plasmid"})
        print(df_clustering)
        df_geo_loc_fixed = df_thresh.merge(
            df_clustering, left_on="Plasmid", right_on="Plasmid"
        )

    return df_geo_loc_fixed


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
