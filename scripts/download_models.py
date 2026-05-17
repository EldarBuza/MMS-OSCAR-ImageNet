"""Download SD-2.1 (768 v-prediction monolithic safetensors), convert to diffusers
folder layout, and download the OSCAR checkpoint. Output: model_zoo/."""
import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ZOO = REPO_ROOT / "model_zoo"
RAW_DIR = MODEL_ZOO / "_raw"
SD_DIR = MODEL_ZOO / "stable-diffusion-2-1"
MODEL_ZOO.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Use the Windows-friendly TMP override so any extension building / conversion
# work stays under MAX_PATH.
os.environ.setdefault("TMP", r"C:\T")
os.environ.setdefault("TEMP", r"C:\T")
os.environ.setdefault("TMPDIR", r"C:\T")

from huggingface_hub import hf_hub_download

print("=" * 60)
print("[1/3] Downloading SD-2.1 768 v-prediction safetensors + yaml")
print("       from webui/stable-diffusion-2-1 (monolithic format)")
print("=" * 60)
sf_path = hf_hub_download(
    repo_id="webui/stable-diffusion-2-1",
    filename="v2-1_768-ema-pruned.safetensors",
    local_dir=str(RAW_DIR),
)
yaml_path = hf_hub_download(
    repo_id="webui/stable-diffusion-2-1",
    filename="v2-1_768-ema-pruned.yaml",
    local_dir=str(RAW_DIR),
)
print(f"  safetensors: {sf_path}")
print(f"  yaml: {yaml_path}\n")

print("=" * 60)
print("[2/3] Converting to diffusers folder layout (vae/unet/scheduler/...)")
print("=" * 60)
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file(
    sf_path,
    original_config_file=yaml_path,
    prediction_type="v_prediction",
    image_size=768,
    use_safetensors=True,
    load_safety_checker=False,
)
print(f"  Saving to {SD_DIR} ...")
SD_DIR.mkdir(parents=True, exist_ok=True)
pipe.save_pretrained(str(SD_DIR), safe_serialization=True)
del pipe  # free RAM before any further work

# Free the source ~5 GB safetensors — we have the diffusers folder now
print(f"  Removing raw safetensors at {RAW_DIR} to reclaim ~5 GB ...")
shutil.rmtree(RAW_DIR, ignore_errors=True)

print()
print("=" * 60)
print("[3/3] Downloading OSCAR checkpoint (oscar.pkl, 394 MB)")
print("=" * 60)
oscar_path = hf_hub_download(
    repo_id="jinpeig/OSCAR",
    filename="oscar.pkl",
    local_dir=str(MODEL_ZOO),
)
print(f"  oscar.pkl: {oscar_path}")

print("\nDone.")
print(f"\nFinal layout under {MODEL_ZOO}:")
for p in sorted(MODEL_ZOO.iterdir()):
    if p.is_dir():
        children = sorted(c.name for c in p.iterdir())
        print(f"  {p.name}/")
        for c in children[:8]:
            print(f"    {c}")
    else:
        print(f"  {p.name}  ({p.stat().st_size / 1024**2:.1f} MB)")
