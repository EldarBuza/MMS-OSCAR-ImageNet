"""Run main_test.py on the 5000-image ImageNet val subset on the 9070 XT.

Bypasses the broken test.sh / accelerate launch / --main_process_port wiring —
just calls main_test.py directly with the right args. Single-process, ROCm GPU.
"""
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = REPO_ROOT / "oscar_env_amd" / "Scripts" / "python.exe"

env = os.environ.copy()
env.setdefault("TMP", r"C:\T")
env.setdefault("TEMP", r"C:\T")
env.setdefault("TMPDIR", r"C:\T")

cmd = [
    str(PY), str(REPO_ROOT / "main_test.py"),
    "-i", str(REPO_ROOT / "data" / "imagenet_val"),
    "-o", str(REPO_ROOT / "results" / "imagenet_val_5k"),
    "--pretrained_model_name_or_path", str(REPO_ROOT / "model_zoo" / "stable-diffusion-2-1"),
    "--oscar_path", str(REPO_ROOT / "model_zoo" / "oscar.pkl"),
    "--seed", "42",
    "--mixed_precision", "fp32",
]

print("Running:", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, env=env, cwd=str(REPO_ROOT)))
