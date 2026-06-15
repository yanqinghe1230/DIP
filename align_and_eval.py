#!/usr/bin/env python3
"""
Check and optionally align GT to input images using ECC (Enhanced Correlation
Coefficient) — a featureless image registration method. This helps diagnose
whether poor metrics are caused by camera misalignment.

Usage:
  # Dry-run: just print per-image alignment PSNR improvement
  python align_and_eval.py \
      --input_dir ./datasets/raw_data/my_test_images \
      --gt_dir    ./datasets/raw_data/my_test_gt

  # Apply alignment and save aligned GT
  python align_and_eval.py \
      --input_dir ./datasets/raw_data/my_test_images \
      --gt_dir    ./datasets/raw_data/my_test_gt \
      --aligned_gt_dir ./datasets/raw_data/my_test_gt_aligned \
      --apply
"""

import argparse
import os
import sys
from os.path import join, splitext, basename

import numpy as np

try:
    import cv2
except ImportError:
    print('[ERROR] OpenCV not found. Install with: pip install opencv-python-headless')
    sys.exit(1)

from PIL import Image


def load_gray(path):
    return np.array(Image.open(path).convert('L'))


def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def ecc_align(src_gray, dst_gray, max_iterations=500, epsilon=1e-6):
    """Align src to dst using ECC. Returns warp_matrix (2×3) or None on failure."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
    motion = cv2.MOTION_EUCLIDEAN  # rotation + translation (preserves scale)
    try:
        _, warp = cv2.findTransformECC(
            src_gray, dst_gray, warp, motion, criteria, inputMask=None, gaussFiltSize=5)
        return warp
    except cv2.error:
        # Try simpler translation-only
        motion = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                src_gray, dst_gray, warp, motion, criteria, inputMask=None, gaussFiltSize=5)
            return warp
        except cv2.error:
            return None


def warp_rgb(img, warp, size_hw):
    return cv2.warpAffine(img, warp, (size_hw[1], size_hw[0]),
                          flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def compute_psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def find_image_files(directory):
    ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if f.lower().endswith(ext) and not f.startswith('.'))


def main():
    parser = argparse.ArgumentParser(description='Check GT-Input alignment')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--aligned_gt_dir', default=None,
                        help='Output directory for aligned GT images')
    parser.add_argument('--apply', action='store_true',
                        help='Actually save aligned GT images')
    parser.add_argument('--gt_suffix', default=None)
    parser.add_argument('--max_long_edge', type=int, default=None,
                        help='Resize images before alignment '
                             '(match model inference resolution)')
    args = parser.parse_args()

    input_files = find_image_files(args.input_dir)
    gt_files = find_image_files(args.gt_dir)

    gt_map = {}
    for f in gt_files:
        stem = splitext(f)[0]
        if args.gt_suffix and stem.endswith(args.gt_suffix):
            stem = stem[:-len(args.gt_suffix)]
        gt_map[stem] = join(args.gt_dir, f)

    if args.apply and args.aligned_gt_dir:
        os.makedirs(args.aligned_gt_dir, exist_ok=True)

    print(f'{"Image":<25} {"PSNR_before":>12} {"PSNR_after":>12} '
          f'{"ΔPSNR":>8} {"Shift(px)":>10} {"Aligned?":>8}')
    print('-' * 80)

    deltas = []
    for fname in input_files:
        stem = splitext(fname)[0]
        input_path = join(args.input_dir, fname)
        gt_path = gt_map.get(stem)
        if gt_path is None:
            print(f'{stem:<25} {"GT not found":>12}')
            continue

        input_img = load_rgb(input_path)
        gt_img = load_rgb(gt_path)

        # Resize to common size for alignment
        if args.max_long_edge:
            h, w = input_img.shape[:2]
            scale = args.max_long_edge / max(h, w)
            if scale < 1.0:
                new_h, new_w = int(round(h * scale)), int(round(w * scale))
                input_img = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Ensure same size
        if input_img.shape[:2] != gt_img.shape[:2]:
            gt_img = cv2.resize(gt_img, (input_img.shape[1], input_img.shape[0]),
                                interpolation=cv2.INTER_LANCZOS4)

        input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)

        # Before PSNR
        psnr_before = compute_psnr(input_img, gt_img)

        # ECC alignment
        warp = ecc_align(gt_gray, input_gray)
        if warp is not None:
            gt_aligned = warp_rgb(gt_img, warp, input_img.shape[:2])
            psnr_after = compute_psnr(input_img, gt_aligned)

            # Compute shift magnitude
            dx = warp[0, 2]
            dy = warp[1, 2]
            shift = np.sqrt(dx**2 + dy**2)

            delta = psnr_after - psnr_before
            deltas.append(delta)

            aligned_str = 'YES'
            if args.apply and args.aligned_gt_dir:
                out_path = join(args.aligned_gt_dir, basename(gt_path))
                Image.fromarray(gt_aligned).save(out_path)
        else:
            psnr_after = psnr_before
            shift = 0
            delta = 0
            aligned_str = 'FAIL'

        print(f'{stem:<25} {psnr_before:>12.2f} {psnr_after:>12.2f} '
              f'{delta:>+8.2f} {shift:>10.2f} {aligned_str:>8}')

    if deltas:
        print(f'\nAverage PSNR improvement from alignment: {np.mean(deltas):.2f} dB')
        if np.mean(deltas) > 3:
            print('→ SIGNIFICANT: your GT images are misaligned. '
                  'Pixel-wise metrics are unreliable.')
            print('  Use --apply to save aligned GT images, then re-run eval_custom.py '
                  'with --gt_dir pointing to the aligned directory.')
        elif np.mean(deltas) > 0.5:
            print('→ MODERATE: minor misalignment, may explain 1-3 dB of metric gap.')
        else:
            print('→ NEGLIGIBLE: images are well-aligned. '
                  'Poor metrics indicate model quality issue.')


if __name__ == '__main__':
    main()
