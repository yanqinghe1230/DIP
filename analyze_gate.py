"""
Gate Activation Analysis Tool (P1)

Analyses the learned gate modulation behaviour of ERRNet+RDNet across datasets
of varying reflection complexity. Generates visualisations and statistics for:

  - Input image
  - RDNet mask prediction (M)
  - Gate activation map (spatial modulation strength A)
  - Output image (reflection-removed)
  - Alpha value statistics (per-channel or scalar)

Usage:
    # Analyse a structure_aware gate checkpoint across datasets
    python analyze_gate.py \
        --icnn_path checkpoints/errnet_gate/errnet_latest.pt \
        --use_rdnet --rdnet_guidance gate --gate_type structure_aware \
        --output_dir ./analysis_results/gate_analysis

    # Analyse a simple gate checkpoint
    python analyze_gate.py \
        --icnn_path checkpoints/errnet_rdnet_joint/latest_net_G.pth \
        --use_rdnet --rdnet_guidance gate --gate_type simple \
        --output_dir ./analysis_results/simple_gate_analysis

    # Specify custom datasets and sample count
    python analyze_gate.py \
        --icnn_path <path> --use_rdnet --rdnet_guidance gate \
        --datasets ceilnet_table2,real20,wild \
        --num_samples 10 --output_dir ./analysis_results
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from os.path import join

import torch.backends.cudnn as cudnn

from options.errnet.train_options import TrainOptions
from engine import Engine
import data.reflect_dataset as datasets


# ---------------------------------------------------------------------------
# Dataset registry (mirrors test_errnet.py EVAL_DATASETS)
# ---------------------------------------------------------------------------
EVAL_DATASETS = OrderedDict({
    "ceilnet_table2": {
        "label": "CEILNet (synthetic)",
        "path": "testdata_CEILNET_table2",
        "max_long_edge": None,
    },
    "real20": {
        "label": "Real20",
        "path": "real20",
        "max_long_edge": 512,
    },
    "objects": {
        "label": "Objects",
        "path": "objects",
        "max_long_edge": None,
    },
    "postcard": {
        "label": "Postcard",
        "path": "postcard",
        "max_long_edge": None,
    },
    "wild": {
        "label": "Wild",
        "path": "wild",
        "max_long_edge": None,
    },
})

TEST_DATASETS = OrderedDict({
    "internet": {
        "label": "Internet",
        "path": None,
        "max_long_edge": None,
    },
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor2np(image_tensor):
    """Convert (1, C, H, W) tensor to (H, W, C) numpy uint8 array."""
    img = image_tensor.detach().cpu().float().numpy()
    # Squeeze batch dim → (C, H, W)
    if img.ndim == 4:
        img = img[0]
    img = np.clip(img, 0, 1)
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))   # (1, H, W) → (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)
    return (img * 255).astype(np.uint8)


def gate_activation_heatmap(gate_tensor):
    """Convert a gate activation tensor to a colour-mapped heatmap.

    Parameters
    ----------
    gate_tensor : (1, 1, H, W) or (1, H, W) torch.Tensor

    Returns
    -------
    np.ndarray (H, W, 3) uint8 — JET-coloured heatmap.
    """
    arr = gate_tensor.detach().cpu().float().numpy().squeeze()
    arr = np.clip(arr, 0, 1)
    # Simple colour map: low=blue, mid=green, high=red
    h, w = arr.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    # Red channel: high activation
    heatmap[..., 0] = (arr * 255).astype(np.uint8)
    # Blue channel: low activation (inverse)
    heatmap[..., 2] = ((1 - arr) * 255).astype(np.uint8)
    # Green channel: mid activation
    heatmap[..., 1] = (np.minimum(arr, 1 - arr) * 2 * 255).astype(np.uint8)
    return heatmap


def save_image(np_img, path):
    """Save numpy (H, W, C) uint8 array to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np_img).save(path)


# ---------------------------------------------------------------------------
# Main analysis logic
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Gate activation analysis for ERRNet+RDNet")
    parser.add_argument(
        "--datasets", type=str,
        default="ceilnet_table2,real20,objects,postcard,wild",
        help="comma-separated dataset keys to analyse")
    parser.add_argument(
        "--data_root", default="./datasets/processed_data",
        help="root directory for processed datasets")
    parser.add_argument(
        "--num_samples", type=int, default=8,
        help="number of samples per dataset")
    parser.add_argument(
        "--output_dir", default="./analysis_results/gate_analysis",
        help="directory for saved outputs")
    parser.add_argument(
        "--no_visuals", action="store_true",
        help="skip saving individual sample visualisations")
    return parser


def run_analysis(cli_args, opt):
    """Core analysis routine."""

    cudnn.benchmark = len(opt.gpu_ids) > 0
    opt.no_log = True
    opt.display_id = 0

    engine = Engine(opt)
    model = engine.model

    # Print model configuration
    print("=" * 60)
    print("Gate Analysis Configuration")
    print(f"  Checkpoint:    {opt.icnn_path}")
    print(f"  RDNet:         {'enabled' if model.use_rdnet else 'disabled'}")
    print(f"  RDNet guidance:{model.rdnet_guidance if model.use_rdnet else 'N/A'}")
    print(f"  Gate type:     {getattr(model, 'gate_type', 'N/A')}")
    print(f"  No Laplacian:  {getattr(model, 'rdnet_no_laplacian', False)}")

    # Alpha stats (before any forward pass — shows learned values)
    alpha_stats = model.get_gate_alpha_stats()
    print(f"  Alpha type:    {alpha_stats['type']}")
    if alpha_stats['type'] == 'simple':
        print(f"  Alpha value:   {alpha_stats['alpha']:.4f}")
    elif alpha_stats['type'] == 'structure_aware':
        print(f"  Alpha mean:    {alpha_stats['alpha_mean']:.4f}")
        print(f"  Alpha std:     {alpha_stats['alpha_std']:.4f}")
        print(f"  Alpha range:   [{alpha_stats['alpha_min']:.4f}, "
              f"{alpha_stats['alpha_max']:.4f}]")
    print("=" * 60)

    # Register hooks to capture gate activations during forward
    captured = model.register_gate_hooks()

    # Process each dataset
    dataset_keys = [k.strip() for k in cli_args.datasets.split(",") if k.strip()]
    per_dataset_results = {}

    for ds_key in dataset_keys:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_key}")
        print(f"{'=' * 60}")

        spec = EVAL_DATASETS.get(ds_key)
        if spec is None:
            print(f"  [skip] Unknown dataset key: {ds_key}")
            continue

        dataset = datasets.CEILTestDataset(
            join(cli_args.data_root, spec["path"]),
            max_long_edge=spec.get("max_long_edge"),
        )
        dataloader = datasets.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=opt.nThreads, pin_memory=True,
        )

        samples_gate_stats = []

        for i, data in enumerate(dataloader):
            if i >= cli_args.num_samples:
                break

            model.set_input(data, mode="eval")
            with torch.no_grad():
                model.forward()

            # --- Spatial gate activation statistics ---
            # Global mean is useless (washes out spatial structure).
            # What matters: how concentrated / localised is the modulation?
            if "gate_activation" in captured:
                gate_act = captured["gate_activation"]  # (1, 1, H_g, W_g)
                gate_2d = gate_act.squeeze().float()     # (H_g, W_g)
                gmin = gate_2d.min()
                gmax = gate_2d.max()
                # Normalise to [0, 1] for thresholding
                gate_n = (gate_2d - gmin) / (gmax - gmin + 1e-8)

                spatial_std = gate_2d.std().item()       # higher → more structure
                peak = gmax.item()
                # Sparsity: fraction of pixels above a relative threshold
                sparsity_20 = (gate_n > 0.2).float().mean().item()
                sparsity_50 = (gate_n > 0.5).float().mean().item()
                sparsity_80 = (gate_n > 0.8).float().mean().item()
                # Spatial gradient magnitude — higher → sharper boundaries
                gy = gate_2d[1:, :] - gate_2d[:-1, :]
                gx = gate_2d[:, 1:] - gate_2d[:, :-1]
                grad_mag = (gy.abs().mean() + gx.abs().mean()).item()

                samples_gate_stats.append({
                    "idx": i,
                    "spatial_std": spatial_std,
                    "peak": peak,
                    "sparsity_20": sparsity_20,
                    "sparsity_50": sparsity_50,
                    "sparsity_80": sparsity_80,
                    "grad_mag": grad_mag,
                })

            # --- Per-sample visualisation ---
            if not cli_args.no_visuals:
                ds_out = join(cli_args.output_dir, ds_key)
                os.makedirs(ds_out, exist_ok=True)

                input_np = tensor2np(model.input)
                save_image(input_np, join(ds_out, f"{i:03d}_input.png"))

                output_np = tensor2np(model.output_i)
                save_image(output_np, join(ds_out, f"{i:03d}_output.png"))

                if model.target_t is not None:
                    target_np = tensor2np(model.target_t)
                    save_image(target_np, join(ds_out, f"{i:03d}_target.png"))

                if model.mask_pred is not None:
                    mask_np = tensor2np(model.mask_pred)
                    save_image(mask_np, join(ds_out, f"{i:03d}_mask.png"))

                if "gate_activation" in captured:
                    gate_act_np = captured["gate_activation"]
                    # If per-channel, average to 1 channel for heatmap
                    if gate_act_np.shape[1] > 1:
                        gate_act_np = gate_act_np.mean(dim=1, keepdim=True)
                    heatmap = gate_activation_heatmap(gate_act_np)
                    # Resize heatmap to match input resolution (gate is applied
                    # at bottleneck, which is H/2 × W/2 of the input)
                    h_in, w_in = input_np.shape[:2]
                    if heatmap.shape[:2] != (h_in, w_in):
                        heatmap = np.array(
                            Image.fromarray(heatmap).resize(
                                (w_in, h_in), Image.BILINEAR))
                    save_image(heatmap, join(ds_out, f"{i:03d}_gate_activation.png"))

                # Overlay gate activation on input
                if "gate_activation" in captured:
                    gate_act_np = captured["gate_activation"]
                    if gate_act_np.shape[1] > 1:
                        gate_act_np = gate_act_np.mean(dim=1, keepdim=True)
                    gate_map = gate_act_np.squeeze().cpu().float().numpy()
                    gate_map = (gate_map - gate_map.min()) / (
                        gate_map.max() - gate_map.min() + 1e-8)
                    # gate_map is at bottleneck resolution; build a same-size
                    # heatmap for blending with the input
                    gate_heatmap = gate_activation_heatmap(gate_act_np)
                    h_in, w_in = input_np.shape[:2]
                    if gate_heatmap.shape[:2] != (h_in, w_in):
                        gate_heatmap = np.array(
                            Image.fromarray(gate_heatmap).resize(
                                (w_in, h_in), Image.BILINEAR))
                    overlay = (input_np.astype(np.float32) * 0.5 +
                               gate_heatmap.astype(np.float32) * 0.5)
                    save_image(overlay.clip(0, 255).astype(np.uint8),
                               join(ds_out, f"{i:03d}_overlay.png"))

            print(f"  [{i:3d}/{cli_args.num_samples}] "
                  f"peak={samples_gate_stats[-1]['peak']:.4f} "
                  f"spatial_std={samples_gate_stats[-1]['spatial_std']:.4f} "
                  f"sparsity@50={samples_gate_stats[-1]['sparsity_50']:.3f}"
                  if samples_gate_stats else f"  [{i:3d}/{cli_args.num_samples}]")

        # Dataset summary — aggregate spatial statistics
        if samples_gate_stats:
            agg = {
                "label": spec["label"],
                "peak_avg": np.mean([s["peak"] for s in samples_gate_stats]),
                "spatial_std_avg": np.mean([s["spatial_std"] for s in samples_gate_stats]),
                "grad_mag_avg": np.mean([s["grad_mag"] for s in samples_gate_stats]),
                "sparsity50_avg": np.mean([s["sparsity_50"] for s in samples_gate_stats]),
                "sparsity20_avg": np.mean([s["sparsity_20"] for s in samples_gate_stats]),
                "n_samples": len(samples_gate_stats),
            }
            per_dataset_results[ds_key] = agg
            print(f"  Summary: peak={agg['peak_avg']:.4f}  "
                  f"spatial_std={agg['spatial_std_avg']:.4f}  "
                  f"grad={agg['grad_mag_avg']:.4f}  "
                  f"sparsity@50={agg['sparsity50_avg']:.3f}")

        dataloader.reset()

    # -----------------------------------------------------------------------
    # Cross-dataset summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Cross-Dataset Summary")
    print(f"{'=' * 60}")

    if alpha_stats["type"] == "simple":
        print(f"\n  Global gate alpha:  {alpha_stats['alpha']:.4f} "
              f"(softplus({np.log(np.exp(alpha_stats['alpha']) - 1):.2f}))")
    elif alpha_stats["type"] == "structure_aware":
        print(f"\n  Per-channel alpha distribution (n=256):")
        print(f"    mean={alpha_stats['alpha_mean']:.4f}  "
              f"std={alpha_stats['alpha_std']:.4f}  "
              f"min={alpha_stats['alpha_min']:.4f}  "
              f"max={alpha_stats['alpha_max']:.4f}  "
              f"median={alpha_stats['alpha_median']:.4f}")

    if per_dataset_results:
        header = (f"{'Dataset':<25s} {'peak':>8s} {'spatial_std':>12s} "
                  f"{'grad':>8s} {'spars@50':>9s} {'spars@20':>9s} {'N':>5s}")
        print(f"\n{header}")
        print("-" * len(header))
        for ds_key in dataset_keys:
            res = per_dataset_results.get(ds_key)
            if res is None:
                continue
            print(f"{res['label']:<25s} "
                  f"{res['peak_avg']:8.4f} "
                  f"{res['spatial_std_avg']:12.4f} "
                  f"{res['grad_mag_avg']:8.4f} "
                  f"{res['sparsity50_avg']:9.3f} "
                  f"{res['sparsity20_avg']:9.3f} "
                  f"{res['n_samples']:5d}")

    # Comparison insight: sort by spatial structure (std) — higher = more localised
    if len(per_dataset_results) >= 2:
        keys_sorted = sorted(
            per_dataset_results.keys(),
            key=lambda k: per_dataset_results[k]["spatial_std_avg"],
        )
        print(f"\n  Spatial structure ranking (low → high, higher = more localised):")
        for rank, k in enumerate(keys_sorted, 1):
            print(f"    {rank}. {per_dataset_results[k]['label']}: "
                  f"std={per_dataset_results[k]['spatial_std_avg']:.4f}  "
                  f"peak={per_dataset_results[k]['peak_avg']:.4f}  "
                  f"grad={per_dataset_results[k]['grad_mag_avg']:.4f}")

        if "wild" in per_dataset_results and "ceilnet_table2" in per_dataset_results:
            ratio = (per_dataset_results["wild"]["spatial_std_avg"] /
                     per_dataset_results["ceilnet_table2"]["spatial_std_avg"])
            print(f"\n  Wild / CEILNet spatial-std ratio: {ratio:.2f}x"
                  f"{' ← stronger spatial structure in wild reflections' if ratio > 1.05 else ''}")
            grad_ratio = (per_dataset_results["wild"]["grad_mag_avg"] /
                          per_dataset_results["ceilnet_table2"]["grad_mag_avg"])
            print(f"  Wild / CEILNet gradient ratio:    {grad_ratio:.2f}x"
                  f"{' ← sharper gate boundaries in wild' if grad_ratio > 1.05 else ''}")

    # Save summary to file
    os.makedirs(cli_args.output_dir, exist_ok=True)
    summary_path = join(cli_args.output_dir, "gate_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Gate Activation Analysis Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {opt.icnn_path}\n")
        f.write(f"Gate type:  {alpha_stats['type']}\n\n")
        if alpha_stats["type"] == "simple":
            f.write(f"Alpha (scalar): {alpha_stats['alpha']:.6f}\n\n")
        elif alpha_stats["type"] == "structure_aware":
            f.write(f"Alpha mean: {alpha_stats['alpha_mean']:.6f}\n")
            f.write(f"Alpha std:  {alpha_stats['alpha_std']:.6f}\n")
            f.write(f"Alpha min:  {alpha_stats['alpha_min']:.6f}\n")
            f.write(f"Alpha max:  {alpha_stats['alpha_max']:.6f}\n")
            f.write(f"Alpha med:  {alpha_stats['alpha_median']:.6f}\n\n")
        f.write(f"{'Dataset':<25s} {'peak':>8s} {'spatial_std':>12s} "
                f"{'grad':>8s} {'spars@50':>9s} {'spars@20':>9s} {'N':>5s}\n")
        f.write("-" * 80 + "\n")
        for ds_key in dataset_keys:
            res = per_dataset_results.get(ds_key)
            if res is None:
                continue
            f.write(f"{res['label']:<25s} "
                    f"{res['peak_avg']:8.4f} "
                    f"{res['spatial_std_avg']:12.4f} "
                    f"{res['grad_mag_avg']:8.4f} "
                    f"{res['sparsity50_avg']:9.3f} "
                    f"{res['sparsity20_avg']:9.3f} "
                    f"{res['n_samples']:5d}\n")

    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Visualisations saved to: {cli_args.output_dir}/<dataset>/\n")

    return per_dataset_results, alpha_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    cli_parser = build_parser()
    cli_args, remaining = cli_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    option_parser = TrainOptions()
    option_parser.isTrain = False
    opt = option_parser.parse()
    opt.isTrain = False

    run_analysis(cli_args, opt)


if __name__ == "__main__":
    main()
