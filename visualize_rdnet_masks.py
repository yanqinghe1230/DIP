import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data.transforms import to_tensor
from models.arch.rdnet import LaplacianPyramid, RDNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize RDNet masks on SIR2 datasets (wild/objects/postcard)."
    )
    parser.add_argument("--rdnet_path", required=True, help="Path to RDNet checkpoint (.pt).")
    parser.add_argument(
        "--data_root",
        default="datasets/processed_data",
        help="Root directory that contains postcard/objects/wild.",
    )
    parser.add_argument(
        "--datasets",
        default="wild,objects,postcard",
        help="Comma-separated dataset names under data_root.",
    )
    parser.add_argument(
        "--num_per_set",
        type=int,
        default=10,
        help="Number of samples per dataset.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/rdnet_viz",
        help="Output directory for visualization panels.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument(
        "--mask_source",
        choices=["auto", "reflection", "transmission", "none"],
        default="auto",
        help=(
            "How to build M_gt. auto=reflection_layer if exists else from transmission_layer; "
            "reflection=use reflection_layer; transmission=use M-T; none=skip M_gt."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for M_hat (binary mask).",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=None,
        help="Resize so the longest edge does not exceed this value.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--no_labels", action="store_true", help="Disable label row.")
    return parser.parse_args()


def resolve_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except RuntimeError:
        return torch.device("cpu")


def torch_load_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_rdnet(rdnet_path, device):
    lap = LaplacianPyramid(channels=3).to(device)
    rdnet = RDNet(3 + lap.out_channels, out_channels=1).to(device)
    state = torch_load_compat(rdnet_path, device)

    state_dict = None
    if isinstance(state, dict):
        if "rdnet" in state and isinstance(state["rdnet"], dict):
            state_dict = state["rdnet"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]

    if state_dict is None:
        state_dict = state

    try:
        rdnet.load_state_dict(state_dict)
    except RuntimeError:
        if isinstance(state, dict):
            for value in state.values():
                if isinstance(value, dict):
                    try:
                        rdnet.load_state_dict(value)
                        break
                    except RuntimeError:
                        continue
        else:
            raise

    rdnet.eval()
    return rdnet, lap


def list_images(path):
    return sorted(
        [p for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def resize_long_edge(img, max_long_edge):
    if max_long_edge is None:
        return img
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


def load_image(path, max_long_edge):
    img = Image.open(path).convert("RGB")
    return resize_long_edge(img, max_long_edge)


def tensor_to_gray_image(tensor):
    array = tensor.detach().cpu().squeeze().clamp(0, 1).numpy() * 255.0
    array = array.astype(np.uint8)
    if array.ndim == 2:
        img = Image.fromarray(array, mode="L")
    else:
        img = Image.fromarray(array[..., 0], mode="L")
    return img.convert("RGB")


def compute_mask_gt(m_tensor, dataset_dir, name, mask_source, max_long_edge):
    reflection_path = dataset_dir / "reflection_layer" / name
    transmission_path = dataset_dir / "transmission_layer" / name

    if mask_source == "none":
        return None

    if mask_source in ("auto", "reflection") and reflection_path.exists():
        r_img = load_image(reflection_path, max_long_edge)
        r_tensor = to_tensor(r_img)
        return r_tensor.mean(dim=0, keepdim=True).clamp(0, 1)

    if mask_source in ("auto", "transmission") and transmission_path.exists():
        t_img = load_image(transmission_path, max_long_edge)
        t_tensor = to_tensor(t_img)
        r_tensor = (m_tensor - t_tensor).clamp(0, 1)
        return r_tensor.mean(dim=0, keepdim=True).clamp(0, 1)

    return None


def build_grid(images, labels, add_labels=True):
    w, h = images[0].size
    header_h = 22 if add_labels else 0
    canvas = Image.new("RGB", (w * len(images), h + header_h), color=(0, 0, 0))
    if add_labels:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, header_h))
        if add_labels:
            draw.text((i * w + 4, 4), label, fill=(255, 255, 255), font=font)

    return canvas


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rdnet, lap = load_rdnet(args.rdnet_path, device)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    for dataset_name in dataset_names:
        dataset_dir = data_root / dataset_name
        blended_dir = dataset_dir / "blended"
        if not blended_dir.exists():
            raise FileNotFoundError(f"Missing blended dir: {blended_dir}")

        out_dir = out_root / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = list_images(blended_dir)
        if len(images) < args.num_per_set:
            raise ValueError(
                f"Not enough images in {blended_dir} (found {len(images)})."
            )

        chosen = rng.sample(images, args.num_per_set)
        selection_log = out_dir / "selected.txt"
        with selection_log.open("w", encoding="utf-8") as f:
            for img_path in chosen:
                f.write(f"{img_path.name}\n")

        for img_path in chosen:
            m_img = load_image(img_path, args.max_long_edge)
            m_tensor = to_tensor(m_img)

            mask_gt = compute_mask_gt(
                m_tensor, dataset_dir, img_path.name, args.mask_source, args.max_long_edge
            )
            if mask_gt is None and args.mask_source != "none":
                raise FileNotFoundError(
                    f"Cannot build M_gt for {img_path.name} in {dataset_dir}. "
                    "Check reflection_layer/transmission_layer or set --mask_source none."
                )

            with torch.no_grad():
                input_tensor = m_tensor.unsqueeze(0).to(device)
                lap_feat = lap(input_tensor)
                rd_input = torch.cat([input_tensor, lap_feat], dim=1)
                mask_pred = rdnet(rd_input).cpu()

            panels = []
            labels = []
            if mask_gt is not None:
                mask_gt_4d = mask_gt.unsqueeze(0)
                mask_pred_times_gt = mask_pred * mask_gt_4d
                mask_hat = (mask_pred > args.threshold).float()
                mask_hat_times_gt = mask_hat * mask_gt_4d

                panels.append(tensor_to_gray_image(mask_gt_4d))
                labels.append("M_gt")
                panels.append(tensor_to_gray_image(mask_pred_times_gt))
                labels.append("M_pred*M_gt")
                panels.append(tensor_to_gray_image(mask_hat_times_gt))
                labels.append("M_hat*M_gt")

            panels.append(tensor_to_gray_image(mask_pred))
            labels.append("M_pred")

            grid = build_grid(panels, labels, add_labels=not args.no_labels)
            out_path = out_dir / f"{img_path.stem}_viz.png"
            grid.save(out_path)

        print(f"Saved {args.num_per_set} panels for {dataset_name} to {out_dir}")


if __name__ == "__main__":
    main()
