"""
test_elvis.py — AdaIR testing on LoViF 5-type data (Blur/Haze/Lowlight/Rain/Snow)
Usage:
    python test_elvis.py --elvis_test_dir ../Test --ckpt_path ckpt/adair5d.ckpt
    python test_elvis.py --elvis_test_dir ../Test --ckpt_path ckpt/adair5d.ckpt --tasks Blur Haze Rain
    python test_elvis.py --elvis_test_dir ../Test --ckpt_path ckpt/adair5d.ckpt --output_path ./results
"""

import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

# ========== AdaIR imports ==========
from net.model import AdaIR
from utils.val_utils import AverageMeter


# ====================================================================
#  Y-channel PSNR/SSIM  (LoViF 官方标准：YCbCr Y 通道)
# ====================================================================
def rgb2y(rgb):
    """RGB (0~1) → Y channel of YCbCr (0~1)"""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def compute_psnr_y(a, b):
    """PSNR on Y channel (tensors, 0~1 range)"""
    diff = a - b
    mse = diff.pow(2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim_y(a, b, data_range=1.0):
    """SSIM on Y channel (tensors, 0~1 range), single-image"""
    import torch.nn.functional as F
    # Gaussian kernel
    kernel_size = 11
    sigma = 1.5
    gauss = torch.arange(kernel_size, dtype=torch.float32, device=a.device)
    gauss = torch.exp(-((gauss - kernel_size // 2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.view(1, 1, -1)
    kernel_2d = kernel_1d @ kernel_1d.transpose(1, 2)  # 1×1×k×k

    a = a.unsqueeze(0).unsqueeze(0)  # 1×1×H×W
    b = b.unsqueeze(0).unsqueeze(0)

    mu_a = F.conv2d(a, kernel_2d, padding=kernel_size // 2)
    mu_b = F.conv2d(b, kernel_2d, padding=kernel_size // 2)

    sigma_a_sq = F.conv2d(a ** 2, kernel_2d, padding=kernel_size // 2) - mu_a ** 2
    sigma_b_sq = F.conv2d(b ** 2, kernel_2d, padding=kernel_size // 2) - mu_b ** 2
    sigma_ab = F.conv2d(a * b, kernel_2d, padding=kernel_size // 2) - mu_a * mu_b

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2))
    return ssim_map.mean().item()


# ====================================================================
#  LPIPS (AlexNet) — LoViF 官方标准
# ====================================================================
def get_lpips_fn(device='cuda'):
    """Load LPIPS with AlexNet backbone"""
    try:
        import lpips
    except ImportError:
        print("LPIPS not installed. Install with: pip install lpips")
        raise
    return lpips.LPIPS(net='alex').to(device).eval()


# ====================================================================
#  Dataset: Elvis 5-type paired LQ/GT
# ====================================================================
class ElvisTestDataset(Dataset):
    """Reads paired data from {test_dir}/{type}/LQ/  and  {test_dir}/{type}/GT/"""

    def __init__(self, test_dir, de_type):
        self.lq_dir = os.path.join(test_dir, de_type, 'LQ')
        self.gt_dir = os.path.join(test_dir, de_type, 'GT')

        self.files = sorted([
            f for f in os.listdir(self.lq_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        # Verify GT files exist
        missing = [f for f in self.files if not os.path.exists(os.path.join(self.gt_dir, f))]
        if missing:
            print(f"  ⚠  {len(missing)} images missing GT (will be skipped)")
            self.files = [f for f in self.files if f not in missing]

        self.to_tensor = ToTensor()
        self.de_type = de_type
        print(f"  [{de_type}] {len(self.files)} test samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        lq_path = os.path.join(self.lq_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)

        lq_img = Image.open(lq_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # Crop to multiple of 16 (AdaIR requirement)
        w, h = lq_img.size
        w_16 = (w // 16) * 16
        h_16 = (h // 16) * 16
        lq_img = lq_img.crop((0, 0, w_16, h_16))
        gt_img = gt_img.crop((0, 0, w_16, h_16))

        lq_t = self.to_tensor(lq_img)  # 3×H×W, 0~1
        gt_t = self.to_tensor(gt_img)

        name = os.path.splitext(fname)[0]
        return name, lq_t.unsqueeze(0), gt_t.unsqueeze(0)  # 1×3×H×W


# ====================================================================
#  AdaIR Lightning Model (copy from original test.py)
# ====================================================================
class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        return None  # unused in test

    def configure_optimizers(self):
        return None


# ====================================================================
#  Inference on one degradation type
# ====================================================================
@torch.no_grad()
def evaluate_type(net, dataset, lpips_fn, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()

    for name, lq, gt in tqdm(loader, desc=f"  {dataset.de_type}", leave=False):
        lq, gt = lq.to(device), gt.to(device)

        restored = net(lq)  # 1×3×H×W, 0~1

        # Y-channel PSNR/SSIM
        y_restored = rgb2y(restored[0])  # H×W
        y_gt = rgb2y(gt[0])  # H×W

        psnr_val = compute_psnr_y(y_restored, y_gt)
        ssim_val = compute_ssim_y(y_restored, y_gt)

        # LPIPS
        # Normalize to [-1, 1] as required by LPIPS
        restored_norm = restored * 2 - 1
        gt_norm = gt * 2 - 1
        lpips_val = lpips_fn(restored_norm, gt_norm).item()

        psnr_meter.update(psnr_val, 1)
        ssim_meter.update(ssim_val, 1)
        lpips_meter.update(lpips_val, 1)

    return psnr_meter.avg, ssim_meter.avg, lpips_meter.avg


# ====================================================================
#  Main
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaIR test on LoViF 5-type data')
    parser.add_argument('--elvis_test_dir', type=str, default='../Test',
                        help='Root dir containing Blur/Haze/Lowlight/Rain/Snow subdirs')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/adair5d.ckpt',
                        help='Path to checkpoint file')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow'],
                        help='Degradation types to test (default: all 5)')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Save restored images to this path (optional)')

    opt = parser.parse_args()

    # Validate test dir
    for t in opt.tasks:
        lq_dir = os.path.join(opt.elvis_test_dir, t, 'LQ')
        if not os.path.isdir(lq_dir):
            print(f"✗ {lq_dir} not found!")
            exit(1)

    device = torch.device(f'cuda:{opt.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"📂 Test dir: {opt.elvis_test_dir}")
    print(f"📦 Checkpoint: {opt.ckpt_path}")
    print(f"📋 Tasks: {opt.tasks}")
    print()

    # Load model
    print("🔄 Loading AdaIR model...")
    net = AdaIRModel().load_from_checkpoint(opt.ckpt_path, strict=False).to(device)
    net.eval()
    print("✅ Model loaded\n")

    # Load LPIPS
    print("🔄 Loading LPIPS (AlexNet)...")
    lpips_fn = get_lpips_fn(device)
    print("✅ LPIPS ready\n")

    # Run evaluation
    results = {}
    for task in opt.tasks:
        print(f"━━━ Testing: {task} ━━━")
        dataset = ElvisTestDataset(opt.elvis_test_dir, task)
        if len(dataset) == 0:
            print(f"  ⚠ No samples, skipping.\n")
            continue

        psnr, ssim, lpips_val = evaluate_type(net, dataset, lpips_fn, device)
        score = psnr + 10 * ssim - 5 * lpips_val
        results[task] = (psnr, ssim, lpips_val, score)

        print(f"  ✅ PSNR: {psnr:.4f}  |  SSIM: {ssim:.4f}  |  LPIPS: {lpips_val:.4f}")
        print(f"  🏆 Score: {psnr:.4f} + 10×{ssim:.4f} - 5×{lpips_val:.4f} = {score:.2f}\n")

    # Summary table
    print("=" * 75)
    print(f"{'Task':<16} {'PSNR(Y)':<12} {'SSIM(Y)':<12} {'LPIPS':<12} {'Score':<12}")
    print("-" * 75)
    total = 0.0
    for task in opt.tasks:
        if task in results:
            psnr, ssim, lpips_val, score = results[task]
            print(f"{task:<16} {psnr:<12.4f} {ssim:<12.4f} {lpips_val:<12.4f} {score:<12.2f}")
            total += score
    if results:
        avg = total / len(results)
        print("-" * 75)
        print(f"{'Average':<16} {'':<12} {'':<12} {'':<12} {avg:<12.2f}")
    print("=" * 75)
