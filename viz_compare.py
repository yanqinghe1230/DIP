#!/usr/bin/env python3
"""
Qualitative comparison: concatenate input, ERRNet, model, GT side-by-side,
then stack all image rows into one big figure.

Output columns:  Input  |  ERRNet  |  Model  |  GT

Usage:
  python viz_compare.py \
      --input_dir   ./datasets/raw_data/my_test_images \
      --gt_dir      ./datasets/raw_data/my_test_gt \
      --errnet_dir  ./results/errnet_baseline \
      --model_dir   ./results/simple_gate \
      --model_name  errnet_sg \
      --output       comparison.png
"""

import argparse
import os
import sys
from os.path import join, splitext

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_image_files(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions) and not f.startswith('.')
    )


def load_rgb(path):
    return Image.open(path).convert('RGB')


def _resolve_output_path(result_dir, stem, model_name):
    """Find model output for a given image stem."""
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
    return None


def _infer_model_name(result_dir):
    """Auto-detect the model name from output files."""
    if not os.path.isdir(result_dir):
        return None
    for item in sorted(os.listdir(result_dir)):
        sub = join(result_dir, item)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.lower().endswith('.png') and f != 'm_input.png':
                    return splitext(f)[0]
    return None


def _get_font(size=24):
    """Try to get a reasonable font, fall back to default."""
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _add_label(canvas, text, x, y, h):
    """Draw a semi-transparent label bar at the top of a column."""
    draw = ImageDraw.Draw(canvas)
    font = _get_font(min(24, h // 20))
    # Background bar
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        # Fallback for older Pillow
        tw, th = draw.textsize(text, font=font)
        bbox = (0, 0, tw, th)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 6
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=(0, 0, 0, 160))
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)


def main():
    parser = argparse.ArgumentParser(
        description='Qualitative comparison: concat Input|ERRNet|Model|GT into a big figure')
    parser.add_argument('--input_dir', required=True,
                        help='Input (blended) images directory')
    parser.add_argument('--gt_dir', required=True,
                        help='Ground truth images directory')
    parser.add_argument('--errnet_dir', default=None,
                        help='ERRNet result directory')
    parser.add_argument('--model_dir', default=None,
                        help='Final model result directory')
    parser.add_argument('--errnet_name', default='errnet',
                        help='ERRNet output filename (default: errnet)')
    parser.add_argument('--model_name', default=None,
                        help='Model output filename (auto-detect if not set)')
    parser.add_argument('--gt_suffix', default=None,
                        help='Suffix to strip from GT filename stem')
    parser.add_argument('--output', '-o', default='comparison.png',
                        help='Output image path (default: comparison.png)')
    parser.add_argument('--target_height', type=int, default=320,
                        help='Row height in pixels (default: 320)')
    parser.add_argument('--gap', type=int, default=4,
                        help='Gap between images in pixels (default: 4)')
    parser.add_argument('--no_labels', action='store_true',
                        help='Hide column labels')
    args = parser.parse_args()

    # ---- Discover images ----
    input_files = find_image_files(args.input_dir)
    if not input_files:
        print(f'[ERROR] No images in {args.input_dir}')
        sys.exit(1)

    gt_files = find_image_files(args.gt_dir)
    gt_map = {}
    for f in gt_files:
        stem = splitext(f)[0]
        if args.gt_suffix and stem.endswith(args.gt_suffix):
            stem = stem[:-len(args.gt_suffix)]
        gt_map[stem] = join(args.gt_dir, f)

    # ---- Determine columns ----
    columns = []
    col_labels = []

    # Input (always)
    columns.append({'label': 'Input', 'files': {}})
    col_labels.append('Input')

    # ERRNet
    if args.errnet_dir:
        if args.errnet_name is None:
            args.errnet_name = _infer_model_name(args.errnet_dir) or 'errnet'
        columns.append({'label': 'ERRNet', 'files': {}, 'dir': args.errnet_dir,
                        'name': args.errnet_name})
        col_labels.append('ERRNet')

    # Model
    if args.model_dir:
        if args.model_name is None:
            args.model_name = _infer_model_name(args.model_dir) or 'model'
        columns.append({'label': 'Model', 'files': {}, 'dir': args.model_dir,
                        'name': args.model_name})
        col_labels.append('Model')

    # GT (always last)
    columns.append({'label': 'GT', 'files': {}})
    col_labels.append('GT')

    # ---- Resolve paths ----
    for fname in input_files:
        stem = splitext(fname)[0]

        # Input
        columns[0]['files'][stem] = join(args.input_dir, fname)

        # ERRNet
        if 'ERRNet' in col_labels:
            idx = col_labels.index('ERRNet')
            path = _resolve_output_path(args.errnet_dir, stem, args.errnet_name)
            columns[idx]['files'][stem] = path

        # Model
        if 'Model' in col_labels:
            idx = col_labels.index('Model')
            path = _resolve_output_path(args.model_dir, stem, args.model_name)
            columns[idx]['files'][stem] = path

        # GT
        idx = col_labels.index('GT')
        columns[idx]['files'][stem] = gt_map.get(stem)

    # ---- Build the big figure ----
    n_cols = len(columns)
    n_rows = len(input_files)
    gap = args.gap
    target_h = args.target_height

    print(f'Layout: {n_rows} rows × {n_cols} cols')
    print(f'Columns: {" | ".join(col_labels)}')

    # First pass: collect image dimensions and compute column widths
    col_widths = []
    for ci, col in enumerate(columns):
        max_w = 0
        for fname in input_files:
            stem = splitext(fname)[0]
            path = col['files'].get(stem)
            if path and os.path.isfile(path):
                img = Image.open(path)
                w, h = img.size
                scaled_w = int(w * target_h / h)
                max_w = max(max_w, scaled_w)
        col_widths.append(max(target_h // 3, max_w))  # minimum width

    total_w = sum(col_widths) + gap * (n_cols - 1)
    total_h = target_h * n_rows + gap * (n_rows - 1)

    print(f'Output size: {total_w} × {total_h}')

    canvas = Image.new('RGB', (total_w, total_h), color=(40, 40, 40))

    for ri, fname in enumerate(input_files):
        stem = splitext(fname)[0]
        y0 = ri * (target_h + gap)
        x0 = 0

        for ci, col in enumerate(columns):
            path = col['files'].get(stem)
            x0 = sum(col_widths[:ci]) + gap * ci

            if path and os.path.isfile(path):
                img = load_rgb(path)
                # Resize to fit column
                w, h = img.size
                new_w = int(w * target_h / h)
                if new_w < 1:
                    new_w = 1
                img = img.resize((new_w, target_h), Image.LANCZOS)
                # Center horizontally in column
                x_offset = x0 + (col_widths[ci] - new_w) // 2
                canvas.paste(img, (x_offset, y0))
            else:
                # Placeholder for missing
                placeholder = Image.new('RGB', (col_widths[ci], target_h), color=(60, 60, 60))
                draw = ImageDraw.Draw(placeholder)
                draw.text((10, target_h // 2 - 10), 'N/A', fill=(150, 150, 150))
                canvas.paste(placeholder, (x0, y0))

    # ---- Add labels ----
    if not args.no_labels:
        label_h = 36
        # Extend canvas vertically for header
        header = Image.new('RGB', (total_w, total_h + label_h + gap), color=(30, 30, 30))
        header.paste(canvas, (0, label_h + gap))
        canvas = header

        draw = ImageDraw.Draw(canvas)
        font = _get_font(18)
        for ci, label in enumerate(col_labels):
            x0 = sum(col_widths[:ci]) + gap * ci
            # Center label text in column
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw, _ = draw.textsize(label, font=font)
            tx = x0 + (col_widths[ci] - tw) // 2
            draw.text((tx, gap), label, fill=(255, 255, 255), font=font)

        # Row labels (image names) on the left
        for ri, fname in enumerate(input_files):
            stem = splitext(fname)[0]
            y0 = label_h + gap + ri * (target_h + gap) + target_h // 2 - 10
            draw.text((4, y0), stem, fill=(200, 200, 200), font=_get_font(14))

    canvas.save(args.output)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
