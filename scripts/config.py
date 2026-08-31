# Global configuration

# Column names
## General
NJOBS = 50

## Meta cols
PLASMID_COL = "Plasmid"
ISOLATE_COL = "Parent"
ID_COL = "PERSON_UNIFIEDPERSONID"
SPECIES_COL = "ISOLATE_TL_SPECIES"
ST_COL = "ISOLATE_TL_MLST_ST"
LA_COL = "ISOLATE_TL_SA_CLASS"
MEC_COL = "ISOLATE_TL_PCR_MEC"
ORIGIN_COL = "origin"

## Genome cols
CLUSTER_COL = "Standard_Cluster_mrsa"
TSNE1D = "tsne1D_mrsa"
TSNE2D = "tsne2D_mrsa"
MOBILITY_COL = "mobility"
REPLICON_COL = "replicon"
AMR_COL = "amr"
VIR_COL = "virulence"
METAL_COL = "metal"
BIOCIDE_COL = "biocide"

## Epi cols
DATE_COL = "MATERIAL_SAMPLINGDATE"
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
