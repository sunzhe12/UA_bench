#!/usr/bin/env bash
# Download the UA-Bench evaluation set from Hugging Face and extract it.
#
#   bash testsite/scripts/download_water_testset.sh [TARGET_DIR]
#
# TARGET_DIR defaults to ./testset_export. Requires the `hf` CLI.
set -euo pipefail

repo_id="zhesun12/water"
filename="testset_export_v2.tar.gz"
revision="${DATASET_REVISION:-117081964052b6b993c0d7e90711a3dd679181f0}"
target_dir="${1:-${TESTSET_DIR:-testset_export}}"

# Integrity checks for the published archive. The values below match the
# relabelled package testset_export_v2.tar.gz (2600 WAV + embedded _gt
# manifest) at the revision above. Update all three whenever the archive on
# Hugging Face is replaced. Both can be overridden per run with
# EXPECTED_SIZE / EXPECTED_SHA256.
expected_size="${EXPECTED_SIZE:-2237271279}"
expected_sha256="${EXPECTED_SHA256:-8d645c5f2255dfb10008a678590735da0b6558d2b11dd8310219961cd0979e86}"

if [[ -f "$target_dir/sft_test_highquality.jsonl" ]]; then
  echo "Evaluation set already present, nothing to do: $target_dir"
  exit 0
fi
if [[ -e "$target_dir" ]]; then
  echo "Refusing to overwrite an incomplete path: $target_dir" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "The hf CLI is required: pip install huggingface_hub" >&2
  exit 1
fi

work_dir="$(dirname "$target_dir")/.water_download"
mkdir -p "$work_dir"
trap 'rm -rf "$work_dir"' EXIT
archive="$work_dir/$filename"

echo "Dataset: $repo_id (revision $revision)"
echo "Target:  $target_dir"
hf download "$repo_id" "$filename" \
  --repo-type dataset \
  --revision "$revision" \
  --local-dir "$work_dir" \
  --max-workers 1
test -f "$archive"

if [[ -n "$expected_size" ]]; then
  actual_size="$(wc -c < "$archive" | tr -d ' ')"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "Unexpected archive size: $actual_size bytes (expected $expected_size)." >&2
    exit 3
  fi
fi

if [[ -n "$expected_sha256" ]]; then
  actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Unexpected archive SHA-256: $actual_sha256" >&2
    echo "Expected: $expected_sha256" >&2
    exit 3
  fi
else
  echo "Warning: EXPECTED_SHA256 is not set; skipping the integrity check." >&2
fi

if tar -tzf "$archive" | awk 'BEGIN { unsafe = 0 }
  /(^\/|(^|\/)\.\.(\/|$))/ { unsafe = 1 }
  END { exit unsafe ? 0 : 1 }'; then
  echo "Archive contains an unsafe path; refusing extraction." >&2
  exit 3
fi

staging="$work_dir/extract"
mkdir -p "$staging"
tar -xzf "$archive" -C "$staging"
if [[ ! -d "$staging/testset_export" ]]; then
  echo "Archive does not contain a testset_export directory." >&2
  exit 3
fi

mkdir -p "$(dirname "$target_dir")"
mv "$staging/testset_export" "$target_dir"
echo "Water test set ready: $target_dir"
