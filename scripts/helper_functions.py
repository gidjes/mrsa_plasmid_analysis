import pandas as pd
import config


def infer_origin(df_in: pd.DataFrame) -> pd.DataFrame:
    # Default to 'MRSA'
    df = df_in.copy()
    df["origin"] = "CA-MRSA"

    # Update to 'LA-MRSA' where condition is met
    df.loc[df[config.LA_COL] == "LA-MRSA", "origin"] = "LA-MRSA"
    df.loc[df[config.ST_COL] == "1232", "origin"] = "CA-MRSA"
    df.loc[
        (df[config.SUBMITTER_TYPE] == "Ziekenhuis") & (df["origin"] != "LA-MRSA"),
        "origin",
    ] = "HA-MRSA"
    # df.loc[
    #     (df[config.SUBMITTER_TYPE] == "Onbekend") & (df["origin"] != "LA-MRSA"),
    #     "origin",
    # ] = "HA-MRSA"
    df.loc[df[config.SPECIES_COL] == "Staphylococcus schweitzeri", "origin"] = "Sar"
    df.loc[df[config.SPECIES_COL] == "Staphylococcus argenteus", "origin"] = "Sar"

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

    return df


def infer_region(row):
    city = row[config.CITY_COL]
    province = row[config.PROVINCE_COL]
    if city == "" or pd.isna(city):
        city = row[config.CITY_COL_BACKUP]
        province = row[config.PROVINCE_COL_BACKUP]
    return city, province


def clean_plasmid_df(
    df_meta: pd.DataFrame, df_clustering: pd.DataFrame
) -> pd.DataFrame:
    """
    Funtion to clean up and fill up the MRSA plasmid data file

    Parameters
    ----------
    df_meta : pd.DataFrame
        DF with plasmid genomic and epi data
    df_clustering : pd.DataFrame
        DF with MRSA assigned clusters

    Returns
    -------
    pd.DataFrame
        DF with genomic, epi, and cluster data
    """
    # Clean the small linear pieces
    df_thresh = df_meta.loc[df_meta["length"] >= 1000].copy()

    # Add AMR_status and gene counts
    df_thresh.loc[:, "amr_status"] = df_thresh["amr"].notna().astype(int)
    df_thresh.loc[:, "amr_count"] = df_thresh["amr"].apply(
        lambda x: 0 if pd.isna(x) else len(x.split(","))
    )
    df_thresh.loc[:, "vir_count"] = df_thresh["virulence"].apply(
        lambda x: 0 if pd.isna(x) else len(x.split(","))
    )
    df_thresh.loc[:, "metal_count"] = df_thresh["metal"].apply(
        lambda x: 0 if pd.isna(x) else len(x.split(","))
    )
    df_thresh.loc[:, "biocide_count"] = df_thresh["biocide"].apply(
        lambda x: 0 if pd.isna(x) else len(x.split(","))
    )

    df_thresh[config.DATE_COL] = pd.to_datetime(
        df_thresh[config.DATE_COL],
        dayfirst=True,
        errors="coerce",  # converts invalid/empty values to NaT
    )

    # Fix the MRSA type column; origin
    df_origin_fixed = infer_origin(df_thresh)

    # Fill empty region values (GPs) and create municipality
    df_origin_fixed[["city", "province"]] = df_origin_fixed.apply(
        infer_region, axis=1, result_type="expand"
    )
    df_geo_loc_fixed = attach_municipalities(df_origin_fixed)

    # Add the MRSA mge-clusters
    df_clustering = df_clustering.add_suffix("_mrsa")
    df_clustering = df_clustering.rename(columns={"Sample_Name_mrsa": "Plasmid"})
    merged_df = df_geo_loc_fixed.merge(
        df_clustering, left_on="Plasmid", right_on="Plasmid"
    )

    return merged_df
