"""Sample N random images from an ImageNet val source folder into data/imagenet_val/.

Usage:
    python scripts/sample_imagenet_val.py --src <path-to-full-val-set> --n 5000 [--seed 42]

Source folder layout assumptions:
- ILSVRC2012_val/ contains 50000 JPEGs (or has subdirectories from devkit re-organization).
- Or any flat folder of .JPEG/.jpg images.

Writes:
- data/imagenet_val/{filename}.JPEG   — symlinks (Windows-junction-friendly) or copies
- data/imagenet_val_manifest.txt      — line-per-file with the sampling seed at top
"""
import argparse
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def gather_images(src: Path) -> list[Path]:
    exts = {".jpeg", ".jpg", ".png", ".bmp"}
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to source folder containing ImageNet val images")
    ap.add_argument("--n", type=int, default=5000, help="Number of images to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "imagenet_val"),
                    help="Destination folder")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if not src.is_dir():
        raise SystemExit(f"src not a directory: {src}")

    print(f"Scanning {src} ...")
    all_imgs = gather_images(src)
    print(f"  Found {len(all_imgs)} candidate images")
    if len(all_imgs) < args.n:
        raise SystemExit(f"Not enough images: need {args.n}, found {len(all_imgs)}")

    random.seed(args.seed)
    sampled = random.sample(all_imgs, args.n)

    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()

    manifest = REPO_ROOT / "data" / "imagenet_val_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"# Sampled from: {src}\n")
        f.write(f"# Seed: {args.seed}\n")
        f.write(f"# Count: {args.n}\n")
        f.write(f"# Mode: {'copy' if args.copy else 'symlink'}\n")
        for src_path in sampled:
            dst = out / src_path.name
            if args.copy:
                shutil.copy2(src_path, dst)
            else:
                try:
                    dst.symlink_to(src_path)
                except OSError:
                    # Windows without dev-mode/admin: fall back to copy
                    shutil.copy2(src_path, dst)
            f.write(f"{src_path}\n")

    print(f"Wrote {args.n} images to {out}")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    main()
