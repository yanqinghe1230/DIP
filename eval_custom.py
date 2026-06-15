#!/usr/bin/env python3
"""
Evaluate reflection removal results on a custom dataset with ground truth.

Computes PSNR, SSIM, NCC, LMSE for:
  1. Input (blended)     vs GT
  2. ERRNet output        vs GT
  3. Final model output   vs GT

Key behaviour: all images are resized to the model output resolution before
comparison. GT is downscaled (not upscaling the output), which avoids
introducing blur artifacts from bicubic upsampling.

Usage:
  # Only compare ERRNet baseline
  python eval_custom.py \
      --input_dir   ./datasets/raw_data/my_test_images \
      --gt_dir      ./datasets/raw_data/my_test_gt \
      --errnet_dir  ./results/errnet_baseline

  # Compare both ERRNet and final model
  python eval_custom.py \
      --input_dir   ./datasets/raw_data/my_test_images \
      --gt_dir      ./datasets/raw_data/my_test_gt \
      --errnet_dir  ./results/errnet_baseline \
      --model_dir   ./results/simple_gate \
      --model_name  errnet_sg
"""

import argparse
import os
import sys
from os.path import join, splitext

import numpy as np
from PIL import Image

# Reuse existing metrics from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util.index import quality_assess


def find_image_files(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
    """Return sorted list of image filenames in directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions) and not f.startswith('.')
    )


def load_image(path):
    """Load image as uint8 numpy array (H, W, 3), RGB."""
    return np.array(Image.open(path).convert('RGB'))


def resize_to_match(src, ref_hw):
    """Resize src to (ref_h, ref_w) using high-quality Lanczos filter."""
    if src.shape[:2] == ref_hw:
        return src
    return np.array(Image.fromarray(src).resize(
        (ref_hw[1], ref_hw[0]), Image.LANCZOS))


def _resolve_output_path(result_dir, stem, model_name):
    """Resolve output image path for a given image stem.

    test() saves: {result_dir}/[{save_subdir}/]{stem}/{model_name}.png
    """
    # Pattern 1: result_dir/{stem}/{model_name}.png
    path = join(result_dir, stem, f'{model_name}.png')
    if os.path.isfile(path):
        return path

    # Pattern 2: result_dir/{any_subdir}/{stem}/{model_name}.png
    if os.path.isdir(result_dir):
        for sub in sorted(os.listdir(result_dir)):
            subdir = join(result_dir, sub)
            if os.path.isdir(subdir):
                p = join(subdir, stem, f'{model_name}.png')
                if os.path.isfile(p):
                    return p

    # Pattern 3: result_dir/{stem}.png  (flat output)
    path = join(result_dir, f'{stem}.png')
    if os.path.isfile(path):
        return path

    return None


def _infer_model_name_from_dir(result_dir):
    """Try to find the model name by looking at existing output files."""
    if not os.path.isdir(result_dir):
        return None
    # Look inside subdirectories
    for item in sorted(os.listdir(result_dir)):
        sub = join(result_dir, item)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.lower().endswith('.png') and f != 'm_input.png':
                    return splitext(f)[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate reflection removal on custom dataset with GT')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing input (blended) images')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory containing ground truth images')
    parser.add_argument('--errnet_dir', default=None,
                        help='Result directory for ERRNet baseline')
    parser.add_argument('--model_dir', default=None,
                        help='Result directory for final model')
    parser.add_argument('--model_name', default=None,
                        help='Output filename for final model (auto-detect if not set)')
    parser.add_argument('--errnet_name', default='errnet',
                        help='Output filename for ERRNet baseline (default: errnet)')
    parser.add_argument('--gt_suffix', default=None,
                        help='Suffix to strip from GT stem, e.g. "_gt" for "img_gt.png"')
    parser.add_argument('--no_resize_gt', action='store_true',
                        help='Do NOT resize GT to output resolution (not recommended)')
    args = parser.parse_args()

    # ---- Find input images ----
    input_files = find_image_files(args.input_dir)
    if not input_files:
        print(f'[ERROR] No images found in {args.input_dir}')
        sys.exit(1)
    print(f'Found {len(input_files)} input images in {args.input_dir}')

    # ---- Resolve GT images ----
    gt_dir = args.gt_dir
    gt_files = find_image_files(gt_dir)
    if not gt_files:
        print(f'[ERROR] No GT images found in {gt_dir}')
        sys.exit(1)

    gt_map = {}
    for f in gt_files:
        stem = splitext(f)[0]
        if args.gt_suffix and stem.endswith(args.gt_suffix):
            stem = stem[:-len(args.gt_suffix)]
        gt_map[stem] = join(gt_dir, f)
    print(f'Found {len(gt_files)} GT images in {gt_dir}')

    # ---- Resolve model outputs ----
    def build_output_map(result_dir, model_name, label):
        """Build stem→path mapping for model outputs."""
        out_map = {}
        if not result_dir:
            return out_map
        if not os.path.isdir(result_dir):
            print(f'[WARN] {label} directory not found: {result_dir}')
            return out_map

        # Auto-detect model name
        if model_name is None:
            model_name = _infer_model_name_from_dir(result_dir)
        if model_name is None:
            print(f'[WARN] {label}: cannot determine model name, using "errnet"')
            model_name = 'errnet'

        found = 0
        for fname in input_files:
            stem = splitext(fname)[0]
            path = _resolve_output_path(result_dir, stem, model_name)
            if path:
                out_map[stem] = path
                found += 1

        print(f'{label}: matched {found}/{len(input_files)} outputs '
              f'(name="{model_name}", dir={result_dir})')
        if found == 0:
            print(f'  [WARN] No outputs found. Check --{label.lower()}_name and directory structure.')
            print(f'  Expected: {result_dir}/<stem>/{model_name}.png')
        return out_map, model_name

    errnet_map, errnet_name = build_output_map(args.errnet_dir, args.errnet_name, 'ERRNet')
    model_map, model_name = build_output_map(args.model_dir, args.model_name, 'Model')

    # ---- Build category list ----
    categories = ['Input']
    if errnet_map:
        categories.append('ERRNet')
    if model_map:
        categories.append('Model')

    # ---- Determine reference resolution ----
    # Use the first available model output to determine the evaluation resolution.
    # GT (and optionally input) will be downscaled to this resolution.
    ref_hw = None
    for out_map in [model_map, errnet_map]:
        if out_map:
            first_path = next(iter(out_map.values()))
            ref_img = load_image(first_path)
            ref_hw = ref_img.shape[:2]
            break

    if ref_hw is None:
        # No model outputs: use input resolution as reference
        first_input = load_image(join(args.input_dir, input_files[0]))
        ref_hw = first_input.shape[:2]
        print(f'\nNo model outputs found. Using input resolution as reference: '
              f'{ref_hw[1]}x{ref_hw[0]}')
    else:
        print(f'\nEvaluation resolution (from model output): {ref_hw[1]}x{ref_hw[0]}')

    # ---- Compute metrics ----
    all_results = {cat: [] for cat in categories}
    resolutions = {}  # for debug

    print(f'\n{"="*85}')
    print(f'{"Image":<25} {"Type":<8} {"PSNR":>8} {"SSIM":>8} {"NCC":>8} {"LMSE":>10}')
    print(f'{"-"*85}')

    matched = 0
    skipped = {cat: 0 for cat in categories}

    for fname in input_files:
        stem = splitext(fname)[0]
        input_path = join(args.input_dir, fname)

        # Find GT
        gt_path = gt_map.get(stem)
        if gt_path is None:
            print(f'  [SKIP] {stem}: GT not found (stem="{stem}")')
            continue

        matched += 1
        gt_full = load_image(gt_path)

        # Resize GT to reference resolution
        if args.no_resize_gt:
            gt = gt_full
            eval_hw = gt_full.shape[:2]
        else:
            gt = resize_to_match(gt_full, ref_hw)
            eval_hw = ref_hw

        # ---- Evaluate each category ----
        for cat in categories:
            if cat == 'Input':
                est_path = input_path
            elif cat == 'ERRNet':
                est_path = errnet_map.get(stem)
            elif cat == 'Model':
                est_path = model_map.get(stem)
            else:
                continue

            if est_path is None:
                skipped[cat] += 1
                continue

            try:
                est = load_image(est_path)
                resolutions.setdefault(cat, []).append(est.shape[:2])

                # Resize estimate to evaluation resolution if needed
                if est.shape[:2] != eval_hw:
                    est = resize_to_match(est, eval_hw)

                metrics = quality_assess(est, gt)
                all_results[cat].append(metrics)
                print(f'{stem:<25} {cat:<8} {metrics["PSNR"]:>8.2f} {metrics["SSIM"]:>8.4f} '
                      f'{metrics["NCC"]:>8.4f} {metrics["LMSE"]:>10.6f}')
            except Exception as e:
                skipped[cat] += 1
                print(f'{stem:<25} {cat:<8} {"--":>8} {"--":>8} {"--":>8} {"--":>10} '
                      f'(error: {e})')

        print()

    # ---- Summary ----
    print(f'{"="*85}')
    print(f'\nSummary ({matched} images evaluated)')
    for cat in categories:
        if skipped[cat]:
            print(f'  Skipped {cat}: {skipped[cat]}')
    print()

    header = f'{"Method":<12} {"PSNR":>8} {"SSIM":>8} {"NCC":>8} {"LMSE":>10}'
    print(header)
    print('-' * len(header))

    for cat in categories:
        results = all_results[cat]
        if results:
            avg = {k: np.mean([r[k] for r in results]) for k in ('PSNR', 'SSIM', 'NCC', 'LMSE')}
            print(f'{cat:<12} {avg["PSNR"]:>8.2f} {avg["SSIM"]:>8.4f} '
                  f'{avg["NCC"]:>8.4f} {avg["LMSE"]:>10.6f}')
        else:
            print(f'{cat:<12} {"--":>8} {"--":>8} {"--":>8} {"--":>10}')

    # ---- Resolution diagnostics ----
    print(f'\n{"="*85}')
    print('Resolution diagnostics:')
    print(f'  GT native:            {gt_full.shape[1]}x{gt_full.shape[0]}')
    print(f'  Evaluation resolution: {eval_hw[1]}x{eval_hw[0]}')
    for cat in categories:
        if cat in resolutions and resolutions[cat]:
            rs = resolutions[cat]
            h_set = sorted(set(r[0] for r in rs))
            w_set = sorted(set(r[1] for r in rs))
            print(f'  {cat} output:       {w_set} x {h_set}')

    # ---- Per-image breakdown ----
    if matched > 1:
        print(f'\n{"="*85}')
        print('Per-image breakdown:')
        for metric in ('PSNR', 'SSIM', 'NCC', 'LMSE'):
            print(f'\n  --- {metric} ---')
            print(f'  {"Image":<25}', end='')
            for cat in categories:
                if all_results[cat]:
                    print(f' {cat:>10}', end='')
            print()

            for i, fname in enumerate(input_files):
                stem = splitext(fname)[0]
                print(f'  {stem:<25}', end='')
                for cat in categories:
                    if i < len(all_results[cat]):
                        val = all_results[cat][i][metric]
                        fmt = f'{val:>10.2f}' if metric == 'PSNR' else f'{val:>10.4f}'
                        print(f' {fmt}', end='')
                    else:
                        print(f' {"--":>10}', end='')
                print()


if __name__ == '__main__':
    main()
