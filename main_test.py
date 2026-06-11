import os
import sys

sys.path.append(os.getcwd())

# Windows ROCm PyTorch is USE_DISTRIBUTED=0; shim torch.distributed before any
# library that imports symbols from it (vector_quantize_pytorch in particular).
from diffusion import _compat_torch_distributed  # noqa: F401

import utils.utils_image as utils
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict

from diffusion.oscar import OSCAR

import pyiqa
from tqdm import tqdm


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Fixed input size for all evaluations on ImageNet. Reasons:
#  - mod-128 trim from the original repo breaks on ImageNet's diverse aspect
#    ratios (60-3264 px range): tiny images zero-out, huge images blow up runtime
#    quadratically, and pyiqa MS-SSIM (5 scales x 11x11 kernel) crashes when any
#    side ends up below 352 px after the trim.
#  - 384x384 is divisible by 128 (clean for the VAE), >=352 (MS-SSIM safe), and
#    consistent with the paper's "center-crop to a fixed square" treatment of
#    CLIC2020 / DIV2K. Latent is 48x48 -> predictable VRAM and runtime.
TARGET_SIZE = 384

def prepare_input(img):
    """Aspect-preserving resize so the SHORT side = TARGET_SIZE, then center-crop
    to TARGET_SIZE x TARGET_SIZE. Handles all ImageNet sizes uniformly."""
    w, h = img.size
    scale = TARGET_SIZE / min(w, h)
    new_w = max(TARGET_SIZE, int(round(w * scale)))
    new_h = max(TARGET_SIZE, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - TARGET_SIZE) // 2
    top  = (new_h - TARGET_SIZE) // 2
    return img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

def _safe_metric(name, fn, *args, **kwargs):
    """Call a pyiqa metric and return its scalar value, or NaN if it raises.
    Per-image metric failures (e.g. MS-SSIM size, ROCm op quirks) must not kill
    the whole 8-bpp loop."""
    try:
        v = fn(*args, **kwargs)
        return v.item() if hasattr(v, "item") else float(v)
    except Exception as e:
        print(f"[metric {name} failed: {type(e).__name__}: {e}]", flush=True)
        return float("nan")

def _nanmean(arr):
    vals = [v for v in arr if v == v]  # NaN != NaN
    return sum(vals) / len(vals) if vals else float("nan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--oscar_path", type=str)
    parser.add_argument("--lora_rank", type=int, default=16)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp32")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    parser.add_argument('--hyper_dim', type=int, default=320, help='output dim of hyper encoder')

    args = parser.parse_args()

    # initialize the model
    model = OSCAR(args).to("cuda")
    model.set_eval()
    sd = torch.load(args.oscar_path)
    model.load_ckpt(sd)

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)

    H_paths = utils.get_image_paths(args.input_image)
    print(f'There are {len(H_paths)} images.')

    device = 'cuda'
    psnr_metric = pyiqa.create_metric('psnr', device="cuda")
    lpips_metric = pyiqa.create_metric('lpips-vgg', device="cuda")
    # DISTS on CUDA fp32 emits NaN on Windows ROCm 7.2.1 (RDNA 4) — VGG feature
    # extractor produces NaN from stage 2 onward. CPU DISTS is correct (verified)
    # and ~82 ms/call, so we run DISTS on CPU and pass .cpu() inputs at call time.
    dists_metric = pyiqa.create_metric('dists', device="cpu")
    musiq_metric = pyiqa.create_metric('musiq', device="cuda")
    clipiqa_metric = pyiqa.create_metric('clipiqa+', device="cuda")
    msssim_metric = pyiqa.create_metric('ms_ssim', device="cuda")
    fid_metric = pyiqa.create_metric('fid', device="cuda")

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    f = open(os.path.join(args.output_dir, 'results.csv'), 'w')
    print('bpp,PSNR,MS-SSIM,DISTS,LPIPS,MUSIQ,CLIPIQA,FID', file=f)
    for level in range(len(model.timesteps)):
        bpp = model.bpps[level]
        os.makedirs(os.path.join(args.output_dir, str(bpp)), exist_ok=True)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['dists'] = []
        test_results['lpips'] = []
        test_results['niqe'] = []
        test_results['musiq'] = []
        test_results['clipiqa'] = []
        test_results['msssim'] = []
        for img in tqdm(H_paths):
            img_name, ext = os.path.splitext(os.path.basename(img))

            img_H = Image.open(img).convert('RGB')
            img_H = prepare_input(img_H)

            # get caption
            lq = tensor_transforms(img_H.copy()).unsqueeze(0).to(device)
            lq = lq * 2 - 1
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=weight_dtype,
                enabled=(weight_dtype != torch.float32),
            ):
                img_E, _, _ = model(lq, level)
            # Autocast may emit fp16 tensors; metrics + saving expect fp32 RGB.
            img_E = img_E.float()

            img_H = np.array(img_H)
            img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
            img_E = np.array(img_E)
            utils.imsave(img_E, os.path.join(args.output_dir, str(bpp), img_name + '.png'))

            img_E, img_H = img_E / 255., img_H / 255.
            img_E = torch.tensor(img_E, device="cuda").permute(2, 0, 1).unsqueeze(0)
            img_H = torch.tensor(img_H, device="cuda").permute(2, 0, 1).unsqueeze(0)
            img_E, img_H = img_E.type(torch.float32), img_H.type(torch.float32)

            # Per-image metrics — each wrapped so a single failure doesn't kill the run.
            # DISTS lives on CPU (ROCm fp32 NaN workaround).
            test_results['psnr'].append(_safe_metric('psnr', psnr_metric, img_E, img_H))
            test_results['lpips'].append(_safe_metric('lpips', lpips_metric, img_E, img_H))
            test_results['dists'].append(_safe_metric('dists', dists_metric, img_E.cpu(), img_H.cpu()))
            test_results['musiq'].append(_safe_metric('musiq', musiq_metric, img_E, img_H))
            test_results['clipiqa'].append(_safe_metric('clipiqa', clipiqa_metric, img_E, img_H))
            test_results['msssim'].append(_safe_metric('msssim', msssim_metric, img_E, img_H))

        avg_fid = fid_metric(os.path.join(args.output_dir, str(bpp)), args.input_image)
        avg_psnr = _nanmean(test_results['psnr'])
        avg_dists = _nanmean(test_results['dists'])
        avg_lpips = _nanmean(test_results['lpips'])
        avg_musiq = _nanmean(test_results['musiq'])
        avg_clipiqa = _nanmean(test_results['clipiqa'])
        avg_mssim = _nanmean(test_results['msssim'])
        print(bpp, 'PSNR:', avg_psnr, 'MS-SSIM:', avg_mssim, 'DISTS:', avg_dists, 'LPIPS:', avg_lpips, 'MUSIQ:', avg_musiq,
              'CLIP-IQA:', avg_clipiqa, "FID:", avg_fid)

        print(bpp, avg_psnr, avg_mssim, avg_dists, avg_lpips, avg_musiq, avg_clipiqa, avg_fid, sep=',', end='\n', file=f)
    f.close()