#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR1="/data/bioinfo/NGS_Data/IDS/AMR/MRSA al in iRODS"
SOURCE_DIR2="/data/bioinfo/NGS_Data/IDS/AMR/"
SOURCE_DIR3="/data/bioinfo/NGS_Data/IDS/AMR/MRSArg al in iRODS"
OUTPUT_DIR="fastqs/"

# ============================================================
# USAGE
# ============================================================

usage() {
    echo "Usage:"
    echo "  $0 SAMPLE_ID"
    echo "  $0 --file SAMPLE_LIST.txt"
    exit 1
}

# ============================================================
# READ SAMPLE LIST
# ============================================================

declare -a SAMPLES

if [[ $# -eq 1 ]]; then

    [[ "$1" == "-h" || "$1" == "--help" ]] && usage

    SAMPLES+=("$1")

elif [[ $# -eq 2 && "$1" == "--file" ]]; then

    SAMPLE_FILE="$2"

    [[ ! -f "$SAMPLE_FILE" ]] && {
        echo "ERROR: Sample file does not exist: $SAMPLE_FILE" >&2
        exit 1
    }

    while IFS= read -r sample || [[ -n "$sample" ]]; do

        sample="${sample%$'\r'}"

        [[ -z "$sample" ]] && continue
        [[ "$sample" =~ ^[[:space:]]*# ]] && continue

        SAMPLES+=("$sample")

    done < "$SAMPLE_FILE"

else
    usage
fi

if [[ ${#SAMPLES[@]} -eq 0 ]]; then
    echo "ERROR: No samples supplied." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================
# BUILD RANGE INDEX
#
# For each source, store:
#
#   RANGE_START
#   RANGE_END
#   RANGE_DIR
#
# ============================================================

declare -a S1_START
declare -a S1_END
declare -a S1_DIR

declare -a S2_START
declare -a S2_END
declare -a S2_DIR

build_range_index() {

    local source_dir="$1"
    local start_array="$2"
    local end_array="$3"
    local dir_array="$4"

    eval "$start_array=()"
    eval "$end_array=()"
    eval "$dir_array=()"

    while IFS= read -r -d '' dir; do

        dirname=$(basename "$dir")

        if [[ "$dirname" =~ ^([0-9]+)-([0-9]+)$ ]]; then

            start="${BASH_REMATCH[1]}"
            end="${BASH_REMATCH[2]}"

            if (( start > end )); then
                echo "WARNING: Invalid range: $dir" >&2
                continue
            fi

            eval "$start_array+=(\"$start\")"
            eval "$end_array+=(\"$end\")"
            eval "$dir_array+=(\"$dir\")"

        fi

    done < <(
        find "$source_dir" -mindepth 1 -maxdepth 1 -type d -print0
    )
}

echo "Building range index for SOURCE_DIR1..."
build_range_index "$SOURCE_DIR1" S1_START S1_END S1_DIR

echo "Building range index for SOURCE_DIR2..."
build_range_index "$SOURCE_DIR2" S2_START S2_END S2_DIR

# ============================================================
# FIND RANGE
# ============================================================

find_range() {

    local sample="$1"
    local start_array="$2"
    local end_array="$3"
    local dir_array="$4"

    local result=""
    local matches=0

    eval 'local n=${#'"$start_array"'[@]}'

    for ((i=0; i<n; i++)); do

        eval 'local start=${'"$start_array"'[$i]}'
        eval 'local end=${'"$end_array"'[$i]}'
        eval 'local dir=${'"$dir_array"'[$i]}'

        if (( sample >= start && sample <= end )); then

            matches=$((matches + 1))

            if [[ -z "$result" ]]; then
                result="$dir"
            fi

        fi

    done

    if (( matches > 1 )); then
        echo "WARNING: $sample occurs in $matches overlapping ranges." >&2
        echo "         Using: $result" >&2
    fi

    [[ -n "$result" ]] && printf '%s\n' "$result"
}

# ============================================================
# BUILD FASTQ INDEX
#
# Index format:
#
#   sample_id<TAB>pair_prefix<TAB>R1<TAB>R2
#
# We only keep COMPLETE pairs.
# ============================================================

declare -A R1_INDEX
declare -A R2_INDEX

index_directory() {

    local directory="$1"

    while IFS= read -r -d '' file; do

        filename=$(basename "$file")

        # ----------------------------------------------------
        # R1
        # ----------------------------------------------------

        if [[ "$filename" =~ ^([0-9]+)(.*)_R1(_[0-9]+)?\.(fastq|fq)(\.gz)?$ ]]; then

            sample_id="${BASH_REMATCH[1]}"
            prefix="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"

            R1_INDEX["$sample_id|$prefix"]="$file"

        # ----------------------------------------------------
        # R2
        # ----------------------------------------------------

        elif [[ "$filename" =~ ^([0-9]+)(.*)_R2(_[0-9]+)?\.(fastq|fq)(\.gz)?$ ]]; then

            sample_id="${BASH_REMATCH[1]}"
            prefix="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"

            R2_INDEX["$sample_id|$prefix"]="$file"

        fi

    done < <(
        find "$directory" -type f \( \
            -iname "*.fastq" -o \
            -iname "*.fastq.gz" -o \
            -iname "*.fq" -o \
            -iname "*.fq.gz" \
        \) -print0
    )
}

# ============================================================
# BUILD INDEXES ONLY FOR RELEVANT RANGES
#
# First determine which range directories are actually needed
# by the requested samples.
# ============================================================

declare -A NEEDED_S1_DIRS
declare -A NEEDED_S2_DIRS

for sample in "${SAMPLES[@]}"; do

    [[ "$sample" =~ ^[0-9]+$ ]] || continue

    if dir=$(find_range "$sample" S1_START S1_END S1_DIR); then
        NEEDED_S1_DIRS["$dir"]=1
    fi

    if dir=$(find_range "$sample" S2_START S2_END S2_DIR); then
        NEEDED_S2_DIRS["$dir"]=1
    fi

done

echo
echo "Indexing relevant SOURCE_DIR1 ranges..."

for dir in "${!NEEDED_S1_DIRS[@]}"; do
    echo "  $dir"
    index_directory "$dir"
done

# Save SOURCE_DIR1 index

declare -A R1_INDEX_S1
declare -A R2_INDEX_S1

for key in "${!R1_INDEX[@]}"; do
    R1_INDEX_S1["$key"]="${R1_INDEX[$key]}"
done

for key in "${!R2_INDEX[@]}"; do
    R2_INDEX_S1["$key"]="${R2_INDEX[$key]}"
done

R1_INDEX=()
R2_INDEX=()

echo
echo "Indexing relevant SOURCE_DIR2 ranges..."

for dir in "${!NEEDED_S2_DIRS[@]}"; do
    echo "  $dir"
    index_directory "$dir"
done

declare -A R1_INDEX_S2
declare -A R2_INDEX_S2

for key in "${!R1_INDEX[@]}"; do
    R1_INDEX_S2["$key"]="${R1_INDEX[$key]}"
done

for key in "${!R2_INDEX[@]}"; do
    R2_INDEX_S2["$key"]="${R2_INDEX[$key]}"
done

R1_INDEX=()
R2_INDEX=()

# ============================================================
# SOURCE 3
#
# No range structure, so index it once.
# ============================================================

echo
echo "Indexing fallback SOURCE_DIR3..."

index_directory "$SOURCE_DIR3"

declare -A R1_INDEX_S3
declare -A R2_INDEX_S3

for key in "${!R1_INDEX[@]}"; do
    R1_INDEX_S3["$key"]="${R1_INDEX[$key]}"
done

for key in "${!R2_INDEX[@]}"; do
    R2_INDEX_S3["$key"]="${R2_INDEX[$key]}"
done

# ============================================================
# FIND PAIR IN INDEX
# ============================================================

find_indexed_pair() {

    local sample="$1"
    local r1_array="$2"
    local r2_array="$3"

    eval 'local keys=("${!'"$r1_array"'[@]}")'

    for key in "${keys[@]}"; do

        if [[ "$key" == "$sample|"* ]]; then

            eval 'local r1=${'"$r1_array"'[$key]}'

            eval 'if [[ -v "'"$r2_array"'[$key]" ]]; then
                local r2=${'"$r2_array"'[$key]}
                printf "%s\n%s\n" "$r1" "$r2"
                return 0
            fi'

        fi

    done

    return 1
}

# ============================================================
# PROCESS SAMPLES
# ============================================================

declare -a NOT_FOUND
declare -a FOUND

for sample in "${SAMPLES[@]}"; do

    echo
    echo "============================================================"
    echo "Sample: $sample"
    echo "============================================================"

    if [[ ! "$sample" =~ ^[0-9]+$ ]]; then
        echo "WARNING: Sample ID is not numeric: $sample"
        NOT_FOUND+=("$sample")
        continue
    fi

    found=false

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    if pair=$(find_indexed_pair "$sample" R1_INDEX_S1 R2_INDEX_S1); then

        r1=$(echo "$pair" | sed -n '1p')
        r2=$(echo "$pair" | sed -n '2p')

        source="SOURCE_DIR1"
        found=true

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    elif pair=$(find_indexed_pair "$sample" R1_INDEX_S2 R2_INDEX_S2); then

        r1=$(echo "$pair" | sed -n '1p')
        r2=$(echo "$pair" | sed -n '2p')

        source="SOURCE_DIR2"
        found=true

    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    elif pair=$(find_indexed_pair "$sample" R1_INDEX_S3 R2_INDEX_S3); then

        r1=$(echo "$pair" | sed -n '1p')
        r2=$(echo "$pair" | sed -n '2p')

        source="SOURCE_DIR3"
        found=true

    fi

    # --------------------------------------------------------
    # COPY
    # --------------------------------------------------------

    if [[ "$found" == true ]]; then

        output_dir="$OUTPUT_DIR/$sample"

        mkdir -p "$output_dir"

        echo "Found in: $source"
        echo "  R1: $r1"
        echo "  R2: $r2"

        cp "$r1" "$output_dir/"
        cp "$r2" "$output_dir/"

        FOUND+=("$sample")

    else

        echo "WARNING: Complete R1/R2 pair not found."
        NOT_FOUND+=("$sample")

    fi

done

# ============================================================
# SUMMARY
# ============================================================

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"

echo "Requested : ${#SAMPLES[@]}"
echo "Found     : ${#FOUND[@]}"
echo "Not found : ${#NOT_FOUND[@]}"

if (( ${#NOT_FOUND[@]} > 0 )); then

    echo
    echo "Samples without a complete R1/R2 pair:"

    for sample in "${NOT_FOUND[@]}"; do
        echo "  $sample"
    done

fi

echo
echo "Done."