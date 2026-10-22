#!/usr/bin/env bash
# Rebuild the revision's pinned muon tables without replacing an existing dump.
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="$PWD/examples/data/revision_physics"
OUT=${1:-$SOURCE}
JULIA_BIN=${JULIA_BIN:-julia}
if (($# > 1)) || [[ "$OUT" == --help || "$OUT" == -h ]]; then
    echo 'Usage: bash dev/create_revision_physics.sh [OUTPUT_DIRECTORY]'
    echo 'Uses the committed pinned input files; refuses to replace an existing materials.pumas.'
    exit 0
fi
if [[ -e "$OUT/materials.pumas" ]]; then
    echo "Dump already exists: $OUT/materials.pumas. Select another output directory to regenerate." >&2
    exit 2
fi
(cd "$SOURCE" && sha256sum -c INPUT_SHA256SUMS)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PUMAS_MATERIALS_PATH="$SOURCE/pumas-materials"
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-2}
export JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS:-2}
{
    date -u +%FT%TZ
    "$JULIA_BIN" --version
    echo 'PUMAS materials commit: d00fd31e5365b699b2d077232313f0c36343a837'
    printf 'Command: %q --project=. examples/loader_example.jl --mdf %q --dump %q\n' \
        "$JULIA_BIN" "$SOURCE/materials.xml" "$OUT/materials.pumas"
} > "$OUT/generation.txt"
"$JULIA_BIN" --project=. examples/loader_example.jl \
    --mdf "$SOURCE/materials.xml" --dump "$OUT/materials.pumas" > "$OUT/loader.log" 2>&1
(cd "$OUT" && sha256sum materials.pumas loader.log generation.txt > OUTPUT_SHA256SUMS)
echo "Created $OUT/materials.pumas; provenance and loader output are beside it."
