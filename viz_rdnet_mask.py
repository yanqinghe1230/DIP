"""
RDNet mask 可视化脚本 —— 从数据集中随机抽取图片，拼接显示原图与预测 mask。

用法:
    python viz_rdnet_mask.py --ckpt checkpoints/rdnet_pretrain/latest_net_G.pth \
        --datasets postcard,objects,wild --num 8

    python viz_rdnet_mask.py --ckpt checkpoints/rdnet_pretrain/latest_net_G.pth \
        --datasets real20,testdata_CEILNET_table2 --num 5 --data_root datasets/processed_data

输出（每张图一个拼接面板）:
    ┌──────────┬──────────┬──────────┐
    │  Blended │   Mask   │ Overlay  │
    └──────────┴──────────┴──────────┘
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.arch.rdnet import LaplacianPyramid, RDNet

# 支持的 dataset → processed_data 子目录
DATASET_DIRS = {
    "postcard": "postcard",
    "objects": "objects",
    "wild": "wild",
    "real20": "real20",
    "sir2_withgt": "sir2_withgt",
    "testdata_CEILNET_table2": "testdata_CEILNET_table2",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="RDNet mask visualization from datasets")
    parser.add_argument("--ckpt", required=True, help="RDNet checkpoint (.pt)")
    parser.add_argument(
        "--datasets", default="postcard,objects,wild",
        help="Comma-separated dataset keys: postcard,objects,wild,real20,sir2_withgt,testdata_CEILNET_table2",
    )
    parser.add_argument("--data_root", default="datasets/processed_data", help="Root of processed datasets")
    parser.add_argument("--num", type=int, default=8, help="Number of samples per dataset")
    parser.add_argument("--out_dir", default="results/rdnet_viz", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--max_size", type=int, default=1024, help="Resize long edge")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser.parse_args()


def load_rdnet(ckpt_path, device):
    """加载 RDNet + LaplacianPyramid。"""
    lap = LaplacianPyramid(channels=3).to(device)
    rdnet = RDNet(3 + lap.out_channels, out_channels=1).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in ("rdnet", "state_dict", "net_rd"):
            if key in ckpt:
                state_dict = ckpt[key]
                break
    rdnet.load_state_dict(state_dict, strict=False)
    rdnet.eval()
    return rdnet, lap


def preprocess(img, max_size):
    """缩放并转为 tensor (1, 3, H, W)，值域 [0,1]。"""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return img, tensor


@torch.no_grad()
def predict(rdnet, lap, tensor, device):
    x = tensor.to(device)
    lap_feat = lap(x)
    mask = rdnet(torch.cat([x, lap_feat], dim=1))
    return mask.squeeze().cpu().numpy()  # (H, W), ∈ [0,1]


def mask_to_heatmap(mask, size):
    """mask → JET 热力图 PIL Image，尺寸对齐 target size。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm

    heatmap = cm.get_cmap("jet")(mask)[:, :, :3]  # RGBA → RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    return Image.fromarray(heatmap).resize(size, Image.BILINEAR)


def hconcat(images, gap=4, bg_color=(40, 40, 40)):
    """水平拼接多张 PIL Image，统一高度，间隔 gap 像素。"""
    h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != h:
            im = im.resize((int(im.width * h / im.height), h), Image.BICUBIC)
        resized.append(im)
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, h), bg_color)
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def add_title(image, title, font_size=18):
    """在图片顶部添加标题条。"""
    from PIL import ImageDraw, ImageFont
    h_bar = font_size + 10
    w = image.width
    canvas = Image.new("RGB", (w, image.height + h_bar), (30, 30, 30))
    canvas.paste(image, (0, h_bar))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tx = (w - bbox[2] + bbox[0]) // 2
    draw.text((tx, 5), title, fill=(220, 220, 220), font=font)
    return canvas


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[i] device: {device}")

    rdnet, lap = load_rdnet(args.ckpt, device)
    print(f"[i] loaded RDNet from {args.ckpt}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]

    for ds_key in dataset_keys:
        if ds_key not in DATASET_DIRS:
            print(f"[!] unknown dataset '{ds_key}', skip")
            continue

        ds_dir = data_root / DATASET_DIRS[ds_key]
        blended_dir = ds_dir / "blended"
        if not blended_dir.is_dir():
            print(f"[!] {blended_dir} not found, skip. Run prepare_test_data.py first.")
            continue

        all_images = sorted(p for p in blended_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if len(all_images) < args.num:
            print(f"[!] {ds_key}: only {len(all_images)} images (< {args.num}), using all")
            chosen = all_images
        else:
            chosen = rng.sample(all_images, args.num)

        ds_out = out_dir / ds_key
        ds_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"[{ds_key}] {len(all_images)} images total, sampled {len(chosen)}")
        print(f"{'='*50}")

        for img_path in chosen:
            img = Image.open(img_path).convert("RGB")
            img_resized, tensor = preprocess(img, args.max_size)
            mask = predict(rdnet, lap, tensor, device)

            # 拼接：原图 | 灰度 mask | 热力图叠加
            mask_gray = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8), mode="L").convert("RGB")
            overlay = Image.blend(img_resized.convert("RGB"),
                                 mask_to_heatmap(mask, img_resized.size), alpha=0.45)

            panel = hconcat([img_resized, mask_gray, overlay])
            panel = add_title(panel, f"{ds_key} / {img_path.name}")

            out_path = ds_out / f"{img_path.stem}_panel.png"
            panel.save(out_path)

            print(f"  {img_path.name}  "
                  f"mean={mask.mean():.3f}  max={mask.max():.3f}  "
                  f">0.5={(mask > 0.5).mean():.1%}")

    print(f"\n[✓] done → {out_dir}/")


if __name__ == "__main__":
    main()
