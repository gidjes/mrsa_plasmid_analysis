# Global configuration

# Column names
## General
PLASMID_COL = "Plasmid"
ISOLATE_COL = "Parent"
ID_COL = "PERSON_UNIFIEDPERSONID"

## Meta cols
SPECIES_COL = "ISOLATE_TL_SPECIES"
ST_COL = "ISOLATE_TL_MLST_ST"
LA_COL = "ISOLATE_TL_SA_CLASS"
ORIGIN_COL = "origin"

## Genome cols
CLUSTER_COL = "Standard_Cluster_mrsa"
TSNE1D = "tsne_1D_mrsa"
TSNE2D = "tsne_2D_mrsa"
AMR_COL = "amr"
VIR_COL = "virulence"
METAL_COL = "metal"
BIOCIDE_COL = "biocide"

## Epi cols
DATE_COL = "MATERIAL_SAMPLING_DATE"
PROVINCE_COL = "MATERIAL_SUBMITTER_PROVINCE"
PROVINCE_COL_BACKUP = "PERSON_PROVINCE"
CITY_COL = "MATERIAL_SUBMITTER_CITY"
CITY_COL_BACKUP = "PERSON_CITY"
SUBMITTER_TYPE = "MATERIAL_SUBMITTER_TYPE"


# Palettes
SPECIES_PALETTE = {
    r"$\it{Staphyloccocus}$ $\it{aureus}$": "#007bc7",
    r"$\it{Staphyloccocus}$ $\it{argenteus}$": "#C6C83A",
}
ORIGIN_PALETTE = {
    "LA-MRSA": "#1baa62",
    "HA-MRSA": "#d52b1e",
    "CA-MRSA": "#007bc7",
    "MSSA": "#737070",
    r"$\it{S. argenteus}$": "#C6C83A",
}
ORIGIN_PALETTE_FULL = {
    "Livestock-associated\nMRSA": "#1baa62",
    "Hopsital-associated\nMRSA": "#d52b1e",
    "Community-associated\nMRSA": "#007bc7",
    r"Sensitive $\it{S. aureus}$": "#737070",
    r"$\it{S. argenteus}$": "#C6C83A",
}
