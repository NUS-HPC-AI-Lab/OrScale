#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_and_prepare_imagenet.sh [SRC_DIR] [OUT_DIR]
#
# Examples:
#   bash scripts/download_and_prepare_imagenet.sh
#   bash scripts/download_and_prepare_imagenet.sh /data/imagenet_tars /data/imagenet
#   IMAGENET_SRC=/data/imagenet_tars IMAGENET_OUT=/data/imagenet \
#       bash scripts/download_and_prepare_imagenet.sh
#
# Notes:
#   - You may need to sign in / request access at:
#       https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php
#   - If the train/val wget commands download an HTML login page instead of a
#     tarball, fetch those files from a signed-in browser session instead.

SRC_DIR="${1:-${IMAGENET_SRC:-/path/to/imagenet_tars}}"
OUT_DIR="${2:-${IMAGENET_OUT:-/data/imagenet}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORIG_CWD="$(pwd -P)"

abspath() {
    python -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

if [[ "$SRC_DIR" != /* ]]; then
    SRC_DIR="$(cd "$ORIG_CWD" && abspath "$SRC_DIR")"
fi

if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$(cd "$ORIG_CWD" && abspath "$OUT_DIR")"
fi

mkdir -p "$SRC_DIR"
cd "$SRC_DIR"

download() {
    local url="$1"
    local filename="$2"
    echo "Downloading $filename ..."
    wget -c --no-check-certificate -O "$filename" "$url"
}

download \
    "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" \
    "ILSVRC2012_img_train.tar"

download \
    "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" \
    "ILSVRC2012_img_val.tar"

download \
    "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz" \
    "ILSVRC2012_devkit_t12.tar.gz"

for required in \
    "ILSVRC2012_img_train.tar" \
    "ILSVRC2012_img_val.tar" \
    "ILSVRC2012_devkit_t12.tar.gz"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing or empty download: $required" >&2
        exit 1
    fi
done

python "$REPO_ROOT/scripts/prepare_vision_data.py" \
    --dataset imagenet \
    --src "$SRC_DIR" \
    --out "$OUT_DIR"
