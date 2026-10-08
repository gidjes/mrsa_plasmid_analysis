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
    r"$\it{Staphyloccocus}$ $\it{aureus}$": "#0072B2",
    r"$\it{Staphyloccocus}$ $\it{argenteus}$": "#F0E442",
}
ORIGIN_PALETTE = {
    "LA-MRSA": "#009E73",
    "HA-MRSA": "#D55E00",
    "CA-MRSA": "#0072B2",
    "MSSA": "#737070",
    r"$\it{S. argenteus}$": "#F0E442",
}
ORIGIN_PALETTE_FULL = {
    "Livestock-associated\nMRSA": "#009E73",
    "Hopsital-associated\nMRSA": "#D55E00",
    "Community-associated\nMRSA": "#0072B2",
    r"Sensitive $\it{S. aureus}$": "#737070",
    r"$\it{S. argenteus}$": "#F0E442",
}
ORIGIN_PALETTE_SHORT = {
    "LA-MRSA": "#009E73",
    "HA-MRSA": "#D55E00",
    "CA-MRSA": "#0072B2",
    "MSSA": "#737070",
    "Sar": "#F0E442",
}

MOBILITY_PALETTE = {
    "non-mobilizable": "#E69F00",
    "mobilizable": "#BB297A",
    "conjugative": "#2BC985",
}
CLONALITY_PALETTE = {
    "Clonal": "#009E73",
    "Same genogroup": "#56B4E9",
    "Distant lineages": "#BB297A",
}
