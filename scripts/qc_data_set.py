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

    # Redone pair
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

    bn_metadata["KEY"] = bn_metadata["KEY"].astype("string").str.strip()

    bn_metadata = bn_metadata.loc[bn_metadata["ISOLATE_BN_NGS_STATUS"] == "Vrijgegeven"]

    bn_metadata = bn_metadata.loc[
        bn_metadata["ISOLATE_BN_STATUS_DATA"] == "In Type-Ned"
    ]

    bn_metadata = bn_metadata.loc[bn_metadata["ISOLATE_BN_STATUS"] != "Verontreinigd"]

    bn_metadata = bn_metadata.loc[
        ~bn_metadata["ISOLATE_TL_SPECIES"].isin(
            [
                "Acinetobacter baumannii ",
                "Klebsiella pneumoniae ",
            ]
        )
    ]

    bn_metadata["MATERIAL_SAMPLINGDATE"] = pd.to_datetime(
        bn_metadata["MATERIAL_SAMPLINGDATE"],
        dayfirst=True,
        errors="coerce",
    )

    bn_metadata = bn_metadata.loc[bn_metadata["MATERIAL_SAMPLINGDATE"] < "2026-08-01"]

    # ---------------------------------------------------------
    # Keys
    # ---------------------------------------------------------

    keys = bn_metadata["KEY"].dropna().astype(str).unique().tolist()

    keys = [key for key in keys if not key.startswith("19")]

    print(f"Number of unique keys: {len(keys)}")

    # ---------------------------------------------------------
    # Prefixes used by B/F
    #
    # KEY itself is the isolate identifier.
    #
    # Example:
    #     KEY = AAA
    #
    # B/F filenames:
    #     AAA_bin_1.fasta
    #     AAA_bin_2.fasta
    #
    # No conversion of the KEY is required.
    # ---------------------------------------------------------

    prefixes = set(keys)

    # ---------------------------------------------------------
    # Search E/F FIRST
    #
    # This is deliberately done before A/B.
    #
    # If a key occurs in E/F, that isolate is considered
    # reconstructed and A/B must not be used.
    # ---------------------------------------------------------

    print("Searching directory E...")

    files_e = find_files(
        directory_e,
        keys,
        fasta_only=True,
    )

    print("Searching directory F...")

    files_f = find_files(
        directory_f,
        prefixes,
        required_substrings=["_bin_"],
        # exclude_suffixes=["_Unbinned.fasta"],
    )

    # ---------------------------------------------------------
    # Search A/B
    #
    # These are still discovered here for reporting and for
    # isolates that are NOT present in E/F.
    #
    # Importantly, they are NOT consulted during selection for
    # an isolate that has an E/F result.
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
        prefixes,
        required_substrings=["_bin_"],
        # exclude_suffixes=["_Unbinned.fasta"],
    )

    # ---------------------------------------------------------
    # Bookkeeping
    # ---------------------------------------------------------

    paired = []
    missing = []

    # Keep these separate because they correspond to different
    # genome metadata files.
    ab_keys = set()
    ef_keys = set()
    no_plasmid_keys = set()

    # ---------------------------------------------------------
    # Select files
    #
    # IMPORTANT:
    #
    # E/F is checked FIRST.
    #
    # If E/F exists for a KEY:
    #
    #     -> use E/F
    #     -> do not check A/B
    #
    # If E/F does not exist:
    #
    #     -> check A/B
    # ---------------------------------------------------------

    for key in keys:
        # =====================================================
        # E/F
        # =====================================================

        e_files = files_e.get(
            key,
            [],
        )

        f_matching = files_f.get(
            key,
            [],
        )

        # -----------------------------------------------------
        # E/F complete pair
        # -----------------------------------------------------

        if e_files and f_matching:

            fasta_file = e_files[0]

            # copy_files(
            #     [fasta_file],
            #     directory_c,
            # )

            # copy_files(
            #     f_matching,
            #     directory_d,
            # )

            ef_keys.add(key)

            paired.append(
                {
                    "KEY": key,
                    "source": "E+F",
                    "fasta_file": fasta_file.name,
                    "number_of_F_files": len(f_matching),
                }
            )

            # -------------------------------------------------
            # CRITICAL:
            #
            # Do NOT inspect A/B for this key.
            #
            # This prevents the original A/B plasmids from
            # being copied over the reconstructed E/F plasmids.
            # -------------------------------------------------

            continue

        # -----------------------------------------------------
        # E chromosome exists but no E/F plasmids
        #
        # This is considered a no-plasmid E/F isolate.
        #
        # We do NOT fall back to A/B here, because the presence
        # of the E chromosome indicates that this isolate was
        # part of the reconstructed set.
        # -----------------------------------------------------

        if e_files and not f_matching:

            no_plasmid_keys.add(key)

            continue

        # =====================================================
        # A/B
        #
        # This section is reached ONLY when there is no E
        # chromosome for this KEY.
        # =====================================================

        a_files = files_a.get(
            key,
            [],
        )

        b_matching = files_b.get(
            key,
            [],
        )

        # -----------------------------------------------------
        # A/B complete pair
        # -----------------------------------------------------

        if a_files and b_matching:

            fasta_file = a_files[0]

            # copy_files(
            #     [fasta_file],
            #     directory_c,
            # )

            # copy_files(
            #     b_matching,
            #     directory_d,
            # )

            ab_keys.add(key)

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
        # No complete pair
        # -----------------------------------------------------

        missing.append(
            {
                "KEY": key,
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
    # Identify isolates from E with no plasmids
    # ---------------------------------------------------------

    no_plasmids = []

    for key in sorted(no_plasmid_keys):

        e_files = files_e.get(
            key,
            [],
        )

        no_plasmids.append(
            {
                "KEY": key,
                "chromosome_file": ";".join(str(path) for path in e_files),
                "plasmid_count": 0,
            }
        )

    # ---------------------------------------------------------
    # Write no-plasmid CSV
    # ---------------------------------------------------------

    no_plasmids_df = pd.DataFrame(
        no_plasmids,
        columns=[
            "KEY",
            "chromosome_file",
            "plasmid_count",
        ],
    )

    no_plasmids_df.to_csv(
        no_plasmids_csv,
        sep=";",
        index=False,
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
    print(f"Total keys:                                 " f"{len(keys)}")
    print(f"Complete A+B pairs:                         " f"{primary_count}")
    print(f"Complete E+F pairs:                         " f"{backup_count}")
    print(f"Isolates from E with no plasmids:           " f"{len(no_plasmid_keys)}")
    print(f"Missing complete pairs:                     " f"{len(missing)}")
    print(f"Missing report:                             " f"{missing_csv}")
    print(f"No-plasmid report:                          " f"{no_plasmids_csv}")

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    overlap_ab_ef = ab_keys & ef_keys

    if overlap_ab_ef:
        raise RuntimeError(
            "A key was selected from both A+B and E+F: " f"{sorted(overlap_ab_ef)}"
        )

    overlap_ab_no_plasmid = ab_keys & no_plasmid_keys

    if overlap_ab_no_plasmid:
        raise RuntimeError(
            "A key was selected as A+B and as no-plasmid: "
            f"{sorted(overlap_ab_no_plasmid)}"
        )

    overlap_ef_no_plasmid = ef_keys & no_plasmid_keys

    if overlap_ef_no_plasmid:
        raise RuntimeError(
            "A key was selected as E+F and as no-plasmid: "
            f"{sorted(overlap_ef_no_plasmid)}"
        )

    # ---------------------------------------------------------
    # Reduce isolate metadata
    #
    # The RIVM metadata is generated in the same way regardless
    # of whether the FASTA came from A/B or E/F.
    # ---------------------------------------------------------
    metadata_keys = set(bn_metadata["KEY"].dropna().astype("string").str.strip())

    found_keys = ab_keys | ef_keys | no_plasmid_keys

    missing_meta = found_keys - metadata_keys

    print(f"Found keys: {len(found_keys)}")
    print(f"Metadata keys: {len(metadata_keys)}")
    print(f"Found keys absent from metadata: {len(missing_meta)}")

    if missing_meta:
        print("Missing metadata keys:")
        print(sorted(missing_meta))

    found_keys = ab_keys | ef_keys | no_plasmid_keys

    bn_metadata = bn_metadata.loc[bn_metadata["KEY"].isin(found_keys)].copy()

    print(f"Metadata rows after filtering: " f"{len(bn_metadata)}")

    print(f"Unique keys after filtering: " f"{bn_metadata['KEY'].nunique()}")

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

    # # ---------------------------------------------------------
    # # Genome metadata
    # #
    # # A/B isolates -> old metadata
    # # E/F isolates -> new metadata
    # #
    # # This prevents a redone isolate from being selected from
    # # the old genome metadata simply because the Parent exists
    # # in both datasets.
    # # ---------------------------------------------------------

    # metadata_old = pd.read_csv(
    #     "../../metadata/RIVM/mrsa_plasmidNL.csv",
    #     sep=";",
    # )

    # metadata_new = pd.read_csv(
    #     "../../PlasmidNL_typing_public/PlasmidNL_report.csv",
    #     sep=";",
    # )

    # # ---------------------------------------------------------
    # # Convert selected keys to integer Parent values
    # # ---------------------------------------------------------

    # ab_parent_keys = set()

    # for key in ab_keys:
    #     try:
    #         ab_parent_keys.add(int(key))
    #     except (ValueError, TypeError):
    #         print(f"Warning: could not convert A/B KEY to integer: " f"{key}")

    # ef_parent_keys = set()

    # for key in ef_keys:
    #     try:
    #         ef_parent_keys.add(int(key))
    #     except (ValueError, TypeError):
    #         print(f"Warning: could not convert E/F KEY to integer: " f"{key}")

    # # ---------------------------------------------------------
    # # Old genome metadata -> A/B only
    # # ---------------------------------------------------------

    # metadata_old = metadata_old.loc[metadata_old["Parent"].isin(ab_parent_keys)].copy()

    # # ---------------------------------------------------------
    # # New genome metadata -> E/F only
    # # ---------------------------------------------------------

    # metadata_new = metadata_new.loc[metadata_new["Parent"].isin(ef_parent_keys)].copy()

    # # ---------------------------------------------------------
    # # Combine genome metadata
    # # ---------------------------------------------------------

    # genome_data = pd.concat(
    #     [
    #         metadata_old,
    #         metadata_new,
    #     ],
    #     ignore_index=True,
    # )

    # genome_data.to_csv(
    #     "data/genome_data.csv",
    #     sep=";",
    #     index=False,
    # )

    # print(f"Genome metadata rows: " f"{len(genome_data)}")

    # print(f"Genome metadata from A/B: " f"{len(metadata_old)}")

    # print(f"Genome metadata from E/F: " f"{len(metadata_new)}")

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\nSelected isolate keys")
    print("---------------------")
    print(f"A+B:                         " f"{len(ab_keys)}")
    print(f"E+F:                         " f"{len(ef_keys)}")
    print(f"E chromosome, no plasmids:   " f"{len(no_plasmid_keys)}")
    print(f"Missing:                     " f"{len(missing)}")
    print(f"Total selected:              " f"{len(found_keys)}")


if __name__ == "__main__":
    qc_set()
