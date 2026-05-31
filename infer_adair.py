"""
infer_adair.py — Pure inference with AdaIR on a folder of LQ images (no GT needed)

Usage:
    python infer_adair.py --input_dir /path/to/LQ/images --ckpt_path ckpt/adair5d.ckpt
    python infer_adair.py --input_dir /path/to/LQ/images --ckpt_path ckpt/adair5d.ckpt --output_dir ./restored --cuda 0
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import lightning.pytorch as pl
from net.model import AdaIR


# ====================================================================
#  Dataset: folder of LQ images (any extension, no GT)
# ====================================================================
class LQOnlyDataset(Dataset):
    """Load all images from a flat folder — no GT, just LQ."""

    def __init__(self, input_dir):
        self.input_dir = input_dir
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        self.files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(exts)
        ])
        if not self.files:
            raise FileNotFoundError(f"No image files found in {input_dir}")
        self.to_tensor = ToTensor()
        print(f"[Dataset] {len(self.files)} images loaded from {input_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.input_dir, fname)

        img = Image.open(path).convert('RGB')

        # Crop to multiple of 16 (AdaIR requirement)
        w, h = img.size
        w_16 = (w // 16) * 16
        h_16 = (h // 16) * 16
        if w_16 != w or h_16 != h:
            img = img.crop((0, 0, w_16, h_16))

        tensor = self.to_tensor(img)  # 3×H×W, 0~1
        name = os.path.splitext(fname)[0]
        return name, tensor


# ====================================================================
#  AdaIR Lightning Model
# ====================================================================
class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return None


# ====================================================================
#  Save image (tensor → PNG)
# ====================================================================
def save_image_tensor(tensor, path):
    """Save a 3×H×W tensor (0~1) to PNG."""
    img = tensor.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(img).save(path)


# ====================================================================
#  Inference
# ====================================================================
@torch.no_grad()
def infer(net, dataset, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for name, lq in tqdm(loader, desc="Inferring"):
        lq = lq.to(device)                    # 1×3×H×W

        restored = net(lq)                     # 1×3×H×W, 0~1

        out_path = os.path.join(output_dir, f"{name[0]}.png")
        save_image_tensor(restored[0], out_path)

    print(f"\n✅ Done! {len(dataset)} images saved to {output_dir}")


# ====================================================================
#  Main
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaIR inference on LQ images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing LQ images (flat, no subdirs)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to .ckpt checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./restored',
                        help='Output folder for restored images (default: ./restored)')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device index')

    opt = parser.parse_args()

    # Validate
    if not os.path.isdir(opt.input_dir):
        print(f"✗ Input directory not found: {opt.input_dir}")
        exit(1)
    if not os.path.isfile(opt.ckpt_path):
        print(f"✗ Checkpoint not found: {opt.ckpt_path}")
        exit(1)

    device = torch.device(f'cuda:{opt.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"📂 Input:  {opt.input_dir}")
    print(f"📦 Model:  {opt.ckpt_path}")
    print(f"💾 Output: {opt.output_dir}")
    print()

    # Load model
    print("🔄 Loading AdaIR...")
    net = AdaIRModel().load_from_checkpoint(opt.ckpt_path, strict=False).to(device)
    net.eval()
    print("✅ Model loaded\n")

    # Load dataset
    dataset = LQOnlyDataset(opt.input_dir)

    # Run inference
    infer(net, dataset, opt.output_dir, device)
