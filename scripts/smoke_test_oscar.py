"""End-to-end OSCAR smoke test — verifies model loads and runs one forward pass per bpp.

Run AFTER scripts/download_models.py finishes and model_zoo/ is populated.
Uses one synthetic 256x256 image; no real ImageNet val needed yet.
"""
import os
import sys
from pathlib import Path

# Resolve to repo root so relative imports work like main_test.py expects
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# torch.distributed shim must come before any lib that imports it (vector_quantize_pytorch)
from diffusion import _compat_torch_distributed  # noqa: F401

import torch
import argparse
from diffusion.oscar import OSCAR

class Args:
    pretrained_model_name_or_path = "model_zoo/stable-diffusion-2-1"
    oscar_path = "model_zoo/oscar.pkl"
    lora_rank = 16
    hyper_dim = 320

def main():
    args = Args()
    print(f"Loading OSCAR from {args.oscar_path}")
    print(f"Loading SD-2.1 from {args.pretrained_model_name_or_path}")

    model = OSCAR(args).to("cuda")
    model.set_eval()
    sd = torch.load(args.oscar_path, map_location="cuda", weights_only=False)
    model.load_ckpt(sd)
    print("Model loaded.")
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Synthetic 256x256 RGB image in [-1, 1]
    torch.manual_seed(0)
    img = torch.rand(1, 3, 256, 256, device="cuda") * 2 - 1
    print(f"Input: {tuple(img.shape)} on {img.device}")

    print("\nForward pass at all 8 bpps:")
    print(f"  {'idx':>3} {'bpp':>8} {'t':>4} {'out shape':>20} {'peak VRAM (GB)':>15}")
    for idx in range(len(model.bpps)):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out, x_denoised, cos_loss = model(img, idx)
        peak = torch.cuda.max_memory_allocated()/1024**3
        print(f"  {idx:>3} {model.bpps[idx]:>8.4f} {int(model.timesteps[idx]):>4} {str(tuple(out.shape)):>20} {peak:>15.2f}")

    print("\nSMOKE TEST PASSED.")

if __name__ == "__main__":
    main()
