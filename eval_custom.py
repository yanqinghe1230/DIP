#!/usr/bin/env python3
"""
Evaluate reflection removal results on a custom dataset with ground truth.

Computes PSNR, SSIM, NCC, LMSE for:
  1. Input (blended)     vs GT
  2. ERRNet output        vs GT
  3. Final model output   vs GT

Usage:
  python eval_custom.py \
      --input_dir   ./datasets/raw_data/my_test_images \
      --gt_dir      ./datasets/raw_data/my_test_gt \
      --errnet_dir  ./results/errnet_baseline \
      --model_dir   ./results/errnet_simple_gate \
      --model_name  errnet
"""

import argparse
import os
import sys
from os.path import join, basename, splitext

import numpy as np
from PIL import Image

# Reuse existing metrics from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util.index import quality_assess


def find_image_files(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
    """Return sorted list of image filenames (with extension) in directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    files = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions) and not f.startswith('.')
    )
    return files


def load_image(path):
    """Load image as uint8 numpy array (H, W, 3), RGB."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def _resolve_output_path(result_dir, fname, model_name):
    """Resolve output image path given the test() output convention.

    test() saves: {result_dir}/{save_subdir}/{stem}/{model_name}.png
    But result_dir may already include the save_subdir, so try multiple patterns.
    """
    stem = splitext(fname)[0]

    # Pattern 1: result_dir is the save_subdir itself
    #   results/custom/img_stem/errnet.png
    path = join(result_dir, stem, f'{model_name}.png')
    if os.path.exists(path):
        return path

    # Pattern 2: result_dir/{any_subdir}/{stem}/{model_name}.png
    #   Used with --save_subdir override
    for sub in sorted(os.listdir(result_dir)):
        p = join(result_dir, sub, stem, f'{model_name}.png')
        if os.path.exists(p):
            return p
        # Also try without per-image subfolder (flat output)
        p2 = join(result_dir, sub, f'{fname}')
        if os.path.exists(p2):
            return p2

    # Pattern 3: flat directory, filename matches
    for ext in ('.png', '.jpg', '.jpeg'):
        p = join(result_dir, f'{stem}{ext}')
        if os.path.exists(p):
            return p

    # Pattern 4: result_dir/{stem}/{model_name}.png directly
    path = join(result_dir, stem, f'{model_name}.png')
    if os.path.exists(path):
        return path

    return None


def resolve_outputs(result_dir, image_files, model_name):
    """Match each input filename to its corresponding output image path."""
    resolved = []
    for fname in image_files:
        path = _resolve_output_path(result_dir, fname, model_name)
        resolved.append(path)
    return resolved


def compute_metrics(estimate_path, gt_path):
    """Compute PSNR, SSIM, NCC, LMSE. Returns dict or None on failure."""
    try:
        est = load_image(estimate_path)
        gt = load_image(gt_path)

        # Ensure same size — if mismatch, resize estimate to match GT
        if est.shape[:2] != gt.shape[:2]:
            est = np.array(Image.fromarray(est).resize(
                (gt.shape[1], gt.shape[0]), Image.BICUBIC))

        metrics = quality_assess(est, gt)
        return metrics
    except Exception as e:
        print(f'  [WARN] {estimate_path}: {e}')
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate reflection removal on custom dataset with GT')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing input (blended) images')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory containing ground truth images')
    parser.add_argument('--errnet_dir', default=None,
                        help='Result directory for ERRNet baseline output')
    parser.add_argument('--model_dir', default=None,
                        help='Result directory for final model output')
    parser.add_argument('--model_name', default='errnet',
                        help='Output filename prefix for model (default: errnet)')
    parser.add_argument('--errnet_name', default='errnet',
                        help='Output filename prefix for ERRNet baseline (default: errnet)')
    parser.add_argument('--gt_ext', default=None,
                        help='Force GT file extension (e.g. .png). If not set, match by stem.')
    parser.add_argument('--gt_suffix', default=None,
                        help='Suffix appended to stem to form GT filename (e.g. "_gt" for img_gt.png)')
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

    # Build GT lookup: stem → path
    gt_map = {}
    for f in gt_files:
        stem = splitext(f)[0]
        # Strip suffix if configured
        if args.gt_suffix and stem.endswith(args.gt_suffix):
            stem = stem[:-len(args.gt_suffix)]
        gt_map[stem] = join(gt_dir, f)

    print(f'Found {len(gt_files)} GT images in {gt_dir}')

    # ---- Resolve model outputs ----
    errnet_map = {}
    if args.errnet_dir:
        errnet_outputs = resolve_outputs(args.errnet_dir, input_files, args.errnet_name)
        for fname, path in zip(input_files, errnet_outputs):
            if path:
                errnet_map[splitext(fname)[0]] = path

    model_map = {}
    if args.model_dir:
        model_outputs = resolve_outputs(args.model_dir, input_files, args.model_name)
        for fname, path in zip(input_files, model_outputs):
            if path:
                model_map[splitext(fname)[0]] = path

    # ---- Compute metrics ----
    categories = ['Input']
    if errnet_map:
        categories.append('ERRNet')
    if model_map:
        categories.append('Model')

    all_results = {cat: [] for cat in categories}

    print(f'\n{"="*80}')
    print(f'{"Image":<30} {"Type":<10} {"PSNR":>8} {"SSIM":>8} {"NCC":>8} {"LMSE":>10}')
    print(f'{"-"*80}')

    matched = 0
    skipped_input = 0
    skipped_errnet = 0
    skipped_model = 0

    for fname in input_files:
        stem = splitext(fname)[0]
        input_path = join(args.input_dir, fname)

        # Find GT
        gt_path = None
        if stem in gt_map:
            gt_path = gt_map[stem]
        elif args.gt_ext:
            gt_path = join(gt_dir, stem + args.gt_ext)
        if gt_path is None or not os.path.exists(gt_path):
            print(f'  [SKIP] {stem}: GT not found (looked for stem="{stem}")')
            continue

        matched += 1

        # Input vs GT
        m = compute_metrics(input_path, gt_path)
        if m:
            all_results['Input'].append(m)
            print(f'{stem:<30} {"Input":<10} {m["PSNR"]:>8.2f} {m["SSIM"]:>8.4f} '
                  f'{m["NCC"]:>8.4f} {m["LMSE"]:>10.6f}')
        else:
            skipped_input += 1

        # ERRNet vs GT
        if 'ERRNet' in categories:
            if stem in errnet_map:
                m = compute_metrics(errnet_map[stem], gt_path)
                if m:
                    all_results['ERRNet'].append(m)
                    print(f'{stem:<30} {"ERRNet":<10} {m["PSNR"]:>8.2f} {m["SSIM"]:>8.4f} '
                          f'{m["NCC"]:>8.4f} {m["LMSE"]:>10.6f}')
                else:
                    skipped_errnet += 1
            else:
                skipped_errnet += 1
                print(f'{stem:<30} {"ERRNet":<10} {"--":>8} {"--":>8} {"--":>8} {"--":>10} '
                      f'(output not found)')

        # Model vs GT
        if 'Model' in categories:
            if stem in model_map:
                m = compute_metrics(model_map[stem], gt_path)
                if m:
                    all_results['Model'].append(m)
                    print(f'{stem:<30} {"Model":<10} {m["PSNR"]:>8.2f} {m["SSIM"]:>8.4f} '
                          f'{m["NCC"]:>8.4f} {m["LMSE"]:>10.6f}')
                else:
                    skipped_model += 1
            else:
                skipped_model += 1
                print(f'{stem:<30} {"Model":<10} {"--":>8} {"--":>8} {"--":>8} {"--":>10} '
                      f'(output not found)')

        print()

    # ---- Summary ----
    print(f'{"="*80}')
    print(f'\nSummary ({matched} images evaluated)')
    if skipped_input:
        print(f'  Skipped input:     {skipped_input}')
    if skipped_errnet:
        print(f'  Skipped ERRNet:    {skipped_errnet}')
    if skipped_model:
        print(f'  Skipped Model:     {skipped_model}')
    print()

    header = f'{"Method":<12} {"PSNR":>8} {"SSIM":>8} {"NCC":>8} {"LMSE":>10}'
    print(header)
    print('-' * len(header))

    for cat in categories:
        results = all_results[cat]
        if results:
            avg_psnr = np.mean([r['PSNR'] for r in results])
            avg_ssim = np.mean([r['SSIM'] for r in results])
            avg_ncc = np.mean([r['NCC'] for r in results])
            avg_lmse = np.mean([r['LMSE'] for r in results])
            print(f'{cat:<12} {avg_psnr:>8.2f} {avg_ssim:>8.4f} '
                  f'{avg_ncc:>8.4f} {avg_lmse:>10.6f}')
        else:
            print(f'{cat:<12} {"--":>8} {"--":>8} {"--":>8} {"--":>10}')

    # Per-image breakdown table
    if matched > 1:
        print(f'\n{"="*80}')
        print('Per-image breakdown:')
        col_widths = {'PSNR': 8, 'SSIM': 8, 'NCC': 8, 'LMSE': 10}
        for metric in ('PSNR', 'SSIM', 'NCC', 'LMSE'):
            print(f'\n  {metric}:')
            print(f'  {"Image":<30}', end='')
            for cat in categories:
                if all_results[cat]:
                    print(f' {cat:>10}', end='')
            print()

            # Collect per-image values across categories
            img_metrics = {}
            for i, fname in enumerate(input_files):
                stem = splitext(fname)[0]
                img_metrics[stem] = {}
                for cat in categories:
                    if i < len(all_results[cat]):
                        img_metrics[stem][cat] = all_results[cat][i][metric]
                    else:
                        img_metrics[stem][cat] = None

            for stem, vals in sorted(img_metrics.items()):
                if all(v is not None for v in vals.values()):
                    print(f'  {stem:<30}', end='')
                    for cat in categories:
                        print(f' {vals[cat]:>10.4f}' if metric != 'PSNR' else f' {vals[cat]:>10.2f}', end='')
                    print()


if __name__ == '__main__':
    main()
