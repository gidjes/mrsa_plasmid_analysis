import os
import shutil
from pathlib import Path

import pandas as pd

FASTA_EXTENSIONS = {".fa", ".fasta", ".fas"}


def find_files(
    source_dir,
    prefixes,
    fasta_only=False,
    exclude_suffixes=None,
    required_substrings=None,
):
    """
    Recursively find files whose filename starts with one of the prefixes.

    Optional filtering can require substrings to occur in the filename
    and/or exclude filenames ending in specific suffixes.
    """
    source_dir = Path(source_dir)
    prefixes = set(prefixes)

    if exclude_suffixes is None:
        exclude_suffixes = []

    if required_substrings is None:
        required_substrings = []

    exclude_suffixes = tuple(suffix.lower() for suffix in exclude_suffixes)

    found = {prefix: [] for prefix in prefixes}

    for root, _, files in os.walk(source_dir):
        for filename in files:
            path = Path(root) / filename
            filename_lower = filename.lower()

            # Exclude unwanted suffixes
            if filename_lower.endswith(exclude_suffixes):
                continue

            # Require specified strings to occur in filename
            if not all(
                substring.lower() in filename_lower for substring in required_substrings
            ):
                continue

            # FASTA-only filtering
            if fasta_only and path.suffix.lower() not in FASTA_EXTENSIONS:
                continue

            # Prefix matching
            for prefix in prefixes:
                if filename.startswith(prefix):
                    found[prefix].append(path)
                    break

    return {prefix: paths for prefix, paths in found.items() if paths}


def copy_files(files, destination_dir):
    """Copy all files to destination_dir."""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for source in files:
        shutil.copy2(
            source,
            destination_dir / source.name,
        )


def qc_set():
    # ---------------------------------------------------------
    # Directories
    # ---------------------------------------------------------

    # Primary pair
    directory_a = Path("../../fastas/RIVM/mrsa_chr")
    directory_b = Path("../../fastas/RIVM/mrsa")

    # Backup pair
    directory_e = Path("../../plasmid_reconstruction/output/isolate_fastas")
    directory_f = Path("../../plasmid_reconstruction/output/final_results")

    # Output directories
    directory_c = Path("fastas_chr/")
    directory_d = Path("fastas")

    missing_csv = Path("data/missing_pairs.csv")
    no_plasmids_csv = Path("data/isolates_without_plasmids.csv")

    directory_c.mkdir(parents=True, exist_ok=True)
    directory_d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Read metadata
    # ---------------------------------------------------------

    bn_metadata = pd.read_csv(
        "data/mrsa_metadata.csv",
        sep=";",
    )

    bn_metadata = bn_metadata.loc[bn_metadata["ISOLATE_BN_NGS_STATUS"] == "Vrijgegeven"]
    bn_metadata = bn_metadata.loc[
        bn_metadata["ISOLATE_BN_STATUS_DATA"] == "In Type-Ned"
    ]
    bn_metadata = bn_metadata.loc[bn_metadata["ISOLATE_BN_STATUS"] != "Verontreinigd"]
    bn_metadata = bn_metadata.loc[
        ~bn_metadata["ISOLATE_TL_SPECIES"].isin(
            ["Acinetobacter baumannii ", "Klebsiella pneumoniae "]
        )
    ]
    bn_metadata["MATERIAL_SAMPLINGDATE"] = pd.to_datetime(
        bn_metadata["MATERIAL_SAMPLINGDATE"], dayfirst=True, errors="coerce"
    )
    bn_metadata = bn_metadata.loc[bn_metadata["MATERIAL_SAMPLINGDATE"] < "1-8-2026"]

    keys = bn_metadata["KEY"].dropna().astype(str).unique().tolist()
    keys = [key for key in keys if not key.startswith("19")]

    print(f"Number of unique keys: {len(keys)}")

    # B/F use the part of KEY before the first underscore
    b_prefixes = {key.split("_", 1)[0] for key in keys}

    # ---------------------------------------------------------
    # Search directories
    # ---------------------------------------------------------

    print("Searching directory A...")
    files_a = find_files(
        directory_a,
        keys,
        fasta_only=True,
    )

    print("Searching directory B...")
    files_b = find_files(
        directory_b,
        b_prefixes,
        required_substrings=["_bin_"],
        # exclude_suffixes=["_Unbinned.fasta"],
    )

    print("Searching directory E...")
    files_e = find_files(
        directory_e,
        keys,
        fasta_only=True,
    )

    print("Searching directory F...")
    files_f = find_files(
        directory_f,
        b_prefixes,
        required_substrings=["_bin_"],
        # exclude_suffixes=["_Unbinned.fasta"],
    )
    # ---------------------------------------------------------
    # Match and copy pairs
    # ---------------------------------------------------------

    paired = []
    missing = []

    for key in keys:
        b_prefix = key.split("_", 1)[0]

        a_files = files_a.get(key, [])
        e_files = files_e.get(key, [])

        # B/F can contain multiple files
        b_matching = files_b.get(b_prefix, [])
        f_matching = files_f.get(b_prefix, [])

        # -----------------------------------------------------
        # First priority: A + B
        # -----------------------------------------------------

        if a_files and b_matching:
            fasta_file = a_files[0]

            copy_files(
                [fasta_file],
                directory_c,
            )

            copy_files(
                b_matching,
                directory_d,
            )

            paired.append(
                {
                    "KEY": key,
                    "source": "A+B",
                    "fasta_file": fasta_file.name,
                    "number_of_B_files": len(b_matching),
                }
            )

            continue

        # -----------------------------------------------------
        # Second priority: E + F
        # -----------------------------------------------------

        if e_files and f_matching:
            fasta_file = e_files[0]

            copy_files(
                [fasta_file],
                directory_c,
            )

            copy_files(
                f_matching,
                directory_d,
            )

            paired.append(
                {
                    "KEY": key,
                    "source": "E+F",
                    "fasta_file": fasta_file.name,
                    "number_of_F_files": len(f_matching),
                }
            )

            continue

        # ---------------------------------------------------------
        # Identify isolates from directory E with no plasmids
        # ---------------------------------------------------------
        no_plasmids = []

        for key in keys:
            e_files = files_e.get(key, [])

            # Only consider isolates that actually have a chromosome
            # FASTA in directory E
            if not e_files:
                continue

            b_prefix = key.split("_", 1)[0]
            f_matching = files_f.get(b_prefix, [])

            if not f_matching:
                no_plasmids.append(
                    {
                        "KEY": key,
                        "chromosome_file": ";".join(str(path) for path in e_files),
                        "plasmid_count": 0,
                    }
                )

        no_plasmids_df = pd.DataFrame(no_plasmids)

        no_plasmids_df.to_csv(
            no_plasmids_csv,
            sep=";",
            index=False,
        )

        # -----------------------------------------------------
        # No complete pair
        # -----------------------------------------------------

        missing.append(
            {
                "KEY": key,
                "B_F_prefix": b_prefix,
                "A_found": bool(a_files),
                "B_found": bool(b_matching),
                "E_found": bool(e_files),
                "F_found": bool(f_matching),
                "A_files": ";".join(str(path) for path in a_files),
                "B_files": ";".join(str(path) for path in b_matching),
                "E_files": ";".join(str(path) for path in e_files),
                "F_files": ";".join(str(path) for path in f_matching),
            }
        )

    # ---------------------------------------------------------
    # Write missing-pair CSV
    # ---------------------------------------------------------

    missing_df = pd.DataFrame(missing)

    missing_df.to_csv(
        missing_csv,
        sep=";",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    primary_count = sum(pair["source"] == "A+B" for pair in paired)

    backup_count = sum(pair["source"] == "E+F" for pair in paired)

    print("\nResults")
    print("-------")
    print(f"Total keys:                                 {len(keys)}")
    print(f"Complete A+B pairs:                         {primary_count}")
    print(f"Complete E+F pairs:                         {backup_count}")
    print(f"Missing complete pairs:                     {len(missing)}")
    print(f"Missing report:                             {missing_csv}")
    print(f"Isolates from directory E with no plasmids: {len(no_plasmids)}")
    print(f"No-plasmid report:                          {no_plasmids_csv}")

    # ---------------------------------------------------------
    # Reduce metadata to keys with a complete pair
    # ---------------------------------------------------------

    found_keys = {pair["KEY"] for pair in paired}
    found_keys += no_plasmids

    bn_metadata = bn_metadata[bn_metadata["KEY"].isin(found_keys)].copy()

    print(f"Metadata rows after filtering: {len(bn_metadata)}")
    print(f"Unique keys after filtering: {bn_metadata['KEY'].nunique()}")

    cols_to_keep = [
        "KEY",
        "ISOLATE_TL_SPECIES",
        "ISOLATE_TL_MLST_ST",
        "ISOLATE_TL_PCR_MEC",
        "ISOLATE_TL_PCR_PVL",
        "ISOLATE_TL_SA_CLASS",
        "MATERIAL_SUBMITTER",
        "MATERIAL_SUBMITTER_CITY",
        "MATERIAL_SUBMITTER_PROVINCE",
        "MATERIAL_SUBMITTER_TYPE",
        "MATERIAL_SAMPLINGDATE",
        "PERSON_CITY",
        "PERSON_PROVINCE",
        "PERSON_UNIFIEDPERSONID",
        "EPI_COUNTRY",
    ]

    bn_metadata[cols_to_keep].to_csv(
        "data/metadata.csv",
        sep=";",
        index=False,
    )

    metadata_old = pd.read_csv("../../metadata/RIVM/mrsa/mrsa_plasmidNL.csv", sep=";")
    metadata_old = metadata_old[metadata_old["Parent"].isin(found_keys)].copy()
    metadata_new = pd.read_csv(
        "../../PlasmidNL_typing_public/PlasmidNL_report.csv", sep=";"
    )
    metadata_new = metadata_new[metadata_new["Parent"].isin(found_keys)].copy()
    genome_data = pd.concat([metadata_old, metadata_new])
    genome_data.to_csv(
        "data/genome_data.csv",
        sep=";",
        index=False,
    )


if __name__ == "__main__":
    qc_set()
