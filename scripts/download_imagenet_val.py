"""Download ImageNet-1k validation parquet shards and extract N random JPEGs.

Source: Tsomaros/Imagenet-1k_validation on HuggingFace (ungated, ~6.24 GB, 50k images,
15 parquet shards under data/validation-*-of-15.parquet).

Output:
- data/imagenet_val/{idx:05d}_{orig_label}.JPEG  — extracted images
- data/imagenet_val_manifest.txt                  — seed + protocol + ordered file list

Designed to be re-runnable: clears existing extracted JPEGs first, but won't re-download
parquet shards already in the HF cache.
"""
import argparse
import io
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("TMP", r"C:\T")
os.environ.setdefault("TEMP", r"C:\T")
os.environ.setdefault("TMPDIR", r"C:\T")

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="Number of JPEGs to extract")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--keep-parquets", action="store_true",
                    help="Don't delete the HF parquet cache after extraction")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "imagenet_val"))
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    out = Path(args.out).resolve()
    print(f"Output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Clear stale extracted JPEGs (but NOT the HF cache)
    for p in out.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()

    print(f"Loading Tsomaros/Imagenet-1k_validation ...")
    # This downloads the parquet shards on first access (cached in ~/.cache/huggingface/)
    ds = load_dataset("Tsomaros/Imagenet-1k_validation", split="validation")
    n_total = len(ds)
    print(f"  Loaded {n_total} validation images")

    if n_total < args.n:
        raise SystemExit(f"Not enough images: need {args.n}, have {n_total}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(n_total), args.n)
    indices.sort()  # ascending order for sequential parquet reads -> cache-friendly

    manifest = REPO_ROOT / "data" / "imagenet_val_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    f = manifest.open("w", encoding="utf-8")
    f.write(f"# Source: Tsomaros/Imagenet-1k_validation (HuggingFace)\n")
    f.write(f"# Split: validation (50000 images)\n")
    f.write(f"# Seed: {args.seed}\n")
    f.write(f"# Count: {args.n}\n")

    print(f"Extracting {args.n} JPEGs ...")
    for written, src_idx in enumerate(indices):
        row = ds[src_idx]
        img = row["image"]  # PIL.Image
        label = row.get("label", -1)
        # Ensure 3-channel RGB JPEG (a few ImageNet val images are grayscale or RGBA)
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = f"{written:05d}_idx{src_idx:05d}_label{label}.JPEG"
        dst = out / fname
        img.save(dst, format="JPEG", quality=95)
        f.write(f"{fname}\tsrc_idx={src_idx}\tlabel={label}\n")
        if (written + 1) % 500 == 0:
            print(f"  {written+1}/{args.n}")

    f.close()
    print(f"\nDone. {args.n} JPEGs at {out}")
    print(f"Manifest: {manifest}")

    if not args.keep_parquets:
        # The HF datasets cache for this dataset can be reclaimed once extraction is done.
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        ds_cache = cache_root / "datasets" / "Tsomaros___imagenet-1k_validation"
        downloads_cache = cache_root / "datasets" / "downloads"
        for d in (ds_cache, downloads_cache):
            if d.exists():
                size_gb = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1024**3
                print(f"Removing HF cache at {d} ({size_gb:.2f} GB)")
                shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    main()
