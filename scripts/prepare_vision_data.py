#!/usr/bin/env python3
"""
Prepare vision datasets for OrScale experiments.

Handles two datasets used in the LAMB / Muon-post vision tables:

    - CIFAR-10: fully automatic download + extract via torchvision.
    - ImageNet-1K (ILSVRC2012): extracts the three official tarballs into
      the ``train/<wnid>/*.JPEG`` + ``val/<wnid>/*.JPEG`` layout expected by
      ``orscale/data/vision.py``.  ImageNet cannot be auto-downloaded (it
      requires a signed image-net.org account), so the user must place
      ``ILSVRC2012_img_train.tar``, ``ILSVRC2012_img_val.tar``, and
      ``ILSVRC2012_devkit_t12.tar.gz`` into ``--src`` (or pass each path
      explicitly).

Usage:
    # Fully automatic: downloads CIFAR-10 to data/cifar10/ .
    python scripts/prepare_vision_data.py --dataset cifar10

    # ImageNet: first download the three official ILSVRC2012 tarballs.
    # 1) Sign in (or request access) at:
    #      https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php
    # 2) Either download them in the browser from that page, or use wget:
    #      mkdir -p /path/to/imagenet_tars && cd /path/to/imagenet_tars
    #      wget -c --no-check-certificate \
    #          https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    #      wget -c --no-check-certificate \
    #          https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    #      wget -c --no-check-certificate \
    #          https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
    #    If the image tarball wget commands download an HTML login page instead
    #    of a tarball, use the browser download links after signing in.
    #
    # ImageNet: extract from pre-downloaded tarballs.
    python scripts/prepare_vision_data.py --dataset imagenet \
        --src /path/to/imagenet_tars \
        --out /data/imagenet

Reference layout produced for ImageNet:

    <out>/train/n01440764/*.JPEG
    <out>/train/n01443537/*.JPEG
    ...
    <out>/val/n01440764/ILSVRC2012_val_00000293.JPEG
    ...
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOGGER = logging.getLogger("orscale.prepare_vision_data")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def prepare_cifar10(out_dir: str) -> None:
    """Download CIFAR-10 via torchvision into ``out_dir``."""
    try:
        from torchvision import datasets  # noqa: F401
    except ImportError as err:
        raise SystemExit(
            "torchvision is required to download CIFAR-10. "
            "Run `pip install torchvision`."
        ) from err
    from torchvision.datasets import CIFAR10

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading CIFAR-10 train split to %s ...", out_path)
    CIFAR10(root=str(out_path), train=True, download=True)
    LOGGER.info("Downloading CIFAR-10 test split to %s ...", out_path)
    CIFAR10(root=str(out_path), train=False, download=True)
    LOGGER.info("CIFAR-10 ready at %s", out_path)


# ---------------------------------------------------------------------------
# ImageNet-1K (ILSVRC2012)
# ---------------------------------------------------------------------------

IMAGENET_TRAIN_TAR = "ILSVRC2012_img_train.tar"
IMAGENET_VAL_TAR = "ILSVRC2012_img_val.tar"
IMAGENET_DEVKIT_TAR = "ILSVRC2012_devkit_t12.tar.gz"


def _resolve_tar(src_dir: str | None, explicit: str | None, default_name: str) -> Path:
    """Pick an explicit path, or fall back to ``{src_dir}/{default_name}``."""
    if explicit:
        p = Path(explicit)
    elif src_dir:
        p = Path(src_dir) / default_name
    else:
        raise SystemExit(
            f"Cannot locate {default_name}. Pass --src <dir> or the explicit "
            "flag for this tarball."
        )
    if not p.is_file():
        raise SystemExit(f"Expected tarball not found: {p}")
    return p


def _extract_train_tar(train_tar: Path, train_dir: Path, workers: int) -> None:
    """
    Extract the ImageNet train tar.

    The outer tar contains 1000 inner tars, one per WordNet synset
    (``n01440764.tar``, etc). For each inner tar we:
        1. Create ``train_dir/<wnid>/``.
        2. Extract its JPEGs into that directory.
    """
    LOGGER.info("Extracting %s to %s (per-class tarballs) ...", train_tar, train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    # First, extract the outer tar to a staging directory.
    staging = train_dir / "_inner_tars"
    staging.mkdir(exist_ok=True)

    with tarfile.open(train_tar, "r") as outer:
        members = [m for m in outer.getmembers() if m.name.endswith(".tar")]
        LOGGER.info("Outer tar has %d per-class tarballs", len(members))
        for i, member in enumerate(members, 1):
            if i % 50 == 0 or i == len(members):
                LOGGER.info("Staging inner tarballs: %d/%d", i, len(members))
            outer.extract(member, path=staging)

    # Then, in parallel, extract each inner tar into its own class folder.
    inner_tars = sorted(staging.rglob("*.tar"))
    LOGGER.info("Extracting %d inner tarballs with %d workers ...", len(inner_tars), workers)

    def _extract_one(tar_path: Path) -> tuple[str, int]:
        wnid = tar_path.stem
        cls_dir = train_dir / wnid
        cls_dir.mkdir(exist_ok=True)
        count = 0
        with tarfile.open(tar_path, "r") as inner:
            inner.extractall(cls_dir)
            count = sum(1 for _ in cls_dir.glob("*.JPEG"))
        tar_path.unlink()
        return wnid, count

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, tp): tp for tp in inner_tars}
        for i, fut in enumerate(as_completed(futures), 1):
            wnid, count = fut.result()
            if i % 50 == 0 or i == len(futures):
                LOGGER.info("Extracted %d/%d classes (last: %s, %d imgs)",
                            i, len(futures), wnid, count)

    shutil.rmtree(staging, ignore_errors=True)
    LOGGER.info("Train split ready at %s (%d classes)", train_dir,
                len(list(train_dir.iterdir())))


def _load_val_ground_truth(devkit_tar: Path) -> list[str]:
    """Return the WordNet ID (wnid) for each ImageNet val image, in order.

    The devkit tar contains:
      - ``data/ILSVRC2012_validation_ground_truth.txt``: 50,000 class IDs
        (1..1000) in image-filename order.
      - ``data/meta.mat``: maps class ID -> WordNet ID (wnid).

    We parse both and return ``[wnid_for_image_1, wnid_for_image_2, ...]``.
    """
    import numpy as np
    from scipy.io import loadmat

    LOGGER.info("Reading ground truth labels from %s ...", devkit_tar)
    with tarfile.open(devkit_tar, "r:gz") as tar:
        gt_member = next(m for m in tar.getmembers() if m.name.endswith("validation_ground_truth.txt"))
        gt_text = tar.extractfile(gt_member).read().decode()  # type: ignore[union-attr]
        ids = [int(x) for x in gt_text.strip().splitlines()]

        meta_member = next(m for m in tar.getmembers() if m.name.endswith("meta.mat"))
        with tar.extractfile(meta_member) as f:  # type: ignore[union-attr]
            meta_bytes = f.read()

    # scipy.io.loadmat needs a file-like or path, so dump to a temp buffer.
    import io
    meta = loadmat(io.BytesIO(meta_bytes))
    synsets = meta["synsets"]
    # synsets is a (1000, 1) struct array; each entry is (ILSVRC2012_ID, WNID, ...).
    # Class IDs in the ground truth file are 1-indexed.
    id_to_wnid: dict[int, str] = {}
    for entry in synsets.flatten():
        ilsvrc_id = int(entry["ILSVRC2012_ID"][0][0])
        wnid = str(entry["WNID"][0])
        id_to_wnid[ilsvrc_id] = wnid
    return [id_to_wnid[c] for c in ids]


def _extract_val_tar(val_tar: Path, val_dir: Path, wnids_per_image: list[str]) -> None:
    """Extract the val tar and reorganize into ``val/<wnid>/`` subdirectories."""
    LOGGER.info("Extracting %s to %s (sorting into wnid subdirs) ...", val_tar, val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create class directories
    for wnid in set(wnids_per_image):
        (val_dir / wnid).mkdir(exist_ok=True)

    with tarfile.open(val_tar, "r") as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.name.endswith(".JPEG")),
            key=lambda m: m.name,
        )
        if len(members) != len(wnids_per_image):
            raise SystemExit(
                f"Val tar has {len(members)} JPEGs but ground truth has "
                f"{len(wnids_per_image)} labels. Corrupt downloads?"
            )
        for i, (member, wnid) in enumerate(zip(members, wnids_per_image), 1):
            target = val_dir / wnid / Path(member.name).name
            with tar.extractfile(member) as src, open(target, "wb") as dst:  # type: ignore[union-attr]
                shutil.copyfileobj(src, dst)
            if i % 5000 == 0 or i == len(members):
                LOGGER.info("Val images extracted: %d/%d", i, len(members))
    LOGGER.info("Val split ready at %s (%d classes)", val_dir,
                len(list(val_dir.iterdir())))


def prepare_imagenet(
    out_dir: str,
    src_dir: str | None,
    train_tar_arg: str | None,
    val_tar_arg: str | None,
    devkit_tar_arg: str | None,
    workers: int,
    skip_train: bool,
    skip_val: bool,
) -> None:
    """Extract ImageNet-1K tarballs into ``out_dir/{train,val}/<wnid>/``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_tar = _resolve_tar(src_dir, train_tar_arg, IMAGENET_TRAIN_TAR) if not skip_train else None
    val_tar = _resolve_tar(src_dir, val_tar_arg, IMAGENET_VAL_TAR) if not skip_val else None
    devkit_tar = (
        _resolve_tar(src_dir, devkit_tar_arg, IMAGENET_DEVKIT_TAR)
        if not skip_val else None
    )

    if train_tar is not None:
        _extract_train_tar(train_tar, out_path / "train", workers=workers)
    else:
        LOGGER.info("Skipping train extraction (--skip-train set).")

    if val_tar is not None and devkit_tar is not None:
        try:
            import scipy  # noqa: F401
        except ImportError as err:
            raise SystemExit(
                "scipy is required to parse the ImageNet devkit meta.mat. "
                "Install with `pip install scipy`."
            ) from err
        wnids = _load_val_ground_truth(devkit_tar)
        _extract_val_tar(val_tar, out_path / "val", wnids)
    else:
        LOGGER.info("Skipping val extraction (--skip-val set or devkit missing).")

    LOGGER.info("ImageNet ready at %s", out_path)
    LOGGER.info("Verify with: ls %s/train | wc -l   # should be 1000", out_path)
    LOGGER.info("             ls %s/val   | wc -l   # should be 1000", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Prepare vision datasets (CIFAR-10 / ImageNet) for OrScale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["cifar10", "imagenet"],
                        help="Which dataset to prepare.")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output directory. Defaults to data/<dataset>.")

    parser.add_argument("--src", type=str, default=None,
                        help="(ImageNet only) Directory containing the three "
                             "pre-downloaded ILSVRC2012 tarballs.")
    parser.add_argument("--train-tar", type=str, default=None,
                        help="(ImageNet) Explicit path to ILSVRC2012_img_train.tar.")
    parser.add_argument("--val-tar", type=str, default=None,
                        help="(ImageNet) Explicit path to ILSVRC2012_img_val.tar.")
    parser.add_argument("--devkit-tar", type=str, default=None,
                        help="(ImageNet) Explicit path to ILSVRC2012_devkit_t12.tar.gz.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="(ImageNet) Parallel workers for per-class tar extraction.")
    parser.add_argument("--skip-train", action="store_true",
                        help="(ImageNet) Skip training split extraction.")
    parser.add_argument("--skip-val", action="store_true",
                        help="(ImageNet) Skip val split extraction.")
    args = parser.parse_args()

    default_out = {
        "cifar10": "data/cifar10",
        "imagenet": "data/imagenet",
    }
    out_dir = args.out or default_out[args.dataset]

    if args.dataset == "cifar10":
        prepare_cifar10(out_dir)
    else:
        prepare_imagenet(
            out_dir=out_dir,
            src_dir=args.src,
            train_tar_arg=args.train_tar,
            val_tar_arg=args.val_tar,
            devkit_tar_arg=args.devkit_tar,
            workers=args.workers,
            skip_train=args.skip_train,
            skip_val=args.skip_val,
        )


if __name__ == "__main__":
    sys.exit(main())
