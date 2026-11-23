#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
BASE_URL="https://zenodo.org/records/2595813/files"

mkdir -p "$DATA_DIR"

files=(
  "annotations.csv"
  "candidates.csv"
  "subset0.zip"
  "subset1.zip"
)

for file in "${files[@]}"; do
  echo "Downloading $file..."
  curl -fL -C - --retry 3 --retry-delay 5 \
    "${BASE_URL}/${file}?download=1" \
    -o "${DATA_DIR}/${file}"
done

echo "Done. Files saved to ${DATA_DIR}/"
