"""
模型参数量与 FLOPs 统计脚本。

用法:
    # 自动构建所有 variant 并统计（无需 pt 文件）
    python count_params.py

    # 传入 checkpoint 文件，自动推断配置并统计
    python count_params.py --ckpt_ernnet path/to/errnet.pt
    python count_params.py --ckpt_concat path/to/concat.pt --ckpt_simple path/to/simple.pt

    # 指定输入尺寸
    python count_params.py --img_size 224
"""

import argparse
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from models.arch.default import DRNet
from models.arch.rdnet import LaplacianPyramid, RDNet
from models import arch


def parse_args():
    parser = argparse.ArgumentParser(description="Count model parameters and FLOPs")
    parser.add_argument("--ckpt_base", type=str, default=None, help="ERRNet baseline checkpoint (.pt)")
    parser.add_argument("--ckpt_concat", type=str, default=None)
    parser.add_argument("--ckpt_simple", type=str, default=None)
    parser.add_argument("--ckpt_per_channel", type=str, default=None)
    parser.add_argument("--ckpt_sa", type=str, default=None)
    parser.add_argument("--ckpt_rdnet", type=str, default=None, help="RDNet-only checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size for FLOPs")
    parser.add_argument("--hyper", action="store_true", default=True, help="Use VGG hypercolumn")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(module, trainable_only=False):
    """Return number of parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_params(n):
    """Format parameter count."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_ernnet_base(hyper=True):
    """Build ERRNet without RDNet (baseline)."""
    in_ch = 3
    if hyper:
        in_ch += 1472  # VGG hypercolumn
    net = arch.errnet(in_ch, 3, gate_type='simple')  # gate_type doesn't matter without RDNet input
    return net


def build_model(use_rdnet=False, rdnet_guidance='concat', gate_type='simple',
                hyper=True, rdnet_in_lap=True):
    """
    Build the complete model matching ERRNetModel.initialize() logic.

    Returns (net_i, rdnet, lap) where rdnet/lap may be None.
    """
    in_ch = 3
    rdnet = None
    lap = None

    lap_channels = 12  # 3 channels × 4 scales

    if use_rdnet:
        need_lap = rdnet_in_lap or (rdnet_guidance == 'gate' and gate_type == 'structure_aware')
        if need_lap:
            lap = LaplacianPyramid(channels=3)

        rd_in = 3 + lap_channels if rdnet_in_lap else 3
        rdnet = RDNet(rd_in, out_channels=1)

        if rdnet_guidance == 'concat':
            in_ch += 1

    if hyper:
        in_ch += 1472

    net_kwargs = {}
    if gate_type == 'structure_aware':
        net_kwargs['gate_type'] = 'structure_aware'
        net_kwargs['lap_channels'] = lap_channels

    net = arch.errnet(in_ch, 3, **net_kwargs)

    return net, rdnet, lap


def load_checkpoint(path, device='cpu'):
    """Load a checkpoint file and return state dicts."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    result = {}

    if isinstance(ckpt, dict):
        if 'icnn' in ckpt:
            result['icnn'] = ckpt['icnn']
        if 'rdnet' in ckpt:
            result['rdnet'] = ckpt['rdnet']
        if 'netD' in ckpt:
            result['netD'] = ckpt['netD']
        if 'epoch' in ckpt:
            result['epoch'] = ckpt['epoch']
        if not result:
            # TorchScript or raw state_dict
            result['icnn'] = ckpt
    else:
        result['icnn'] = ckpt

    return result


def infer_config_from_state(state_dict):
    """Infer model configuration from state_dict keys."""
    config = {'use_rdnet': False, 'rdnet_guidance': 'concat', 'gate_type': 'simple',
              'hyper': False, 'rdnet_in_lap': True}

    icnn = state_dict.get('icnn', state_dict)
    # Check for RDNet
    has_rdnet = 'rdnet' in state_dict
    config['use_rdnet'] = has_rdnet

    # Check for gate type
    for key in icnn.keys():
        if 'structure_gate' in key:
            config['gate_type'] = 'structure_aware'
            if has_rdnet:
                config['rdnet_guidance'] = 'gate'
            break
        elif 'gate_conv' in key:
            # Has gate_conv but not structure_gate → simple or per_channel
            pass  # keep default simple, or check alpha shape later

    # Check for per_channel: if gate_alpha is a vector
    if 'gate_alpha' in icnn and config['gate_type'] != 'structure_aware':
        alpha = icnn['gate_alpha']
        if alpha.numel() > 1:
            config['gate_type'] = 'per_channel'

    # Infer guidance mode: if gate_conv or structure_gate exists, it's gate mode
    if has_rdnet:
        if config['gate_type'] == 'structure_aware':
            config['rdnet_guidance'] = 'gate'
        elif 'gate_conv' in icnn:
            config['rdnet_guidance'] = 'gate'
        else:
            # Check input channels of first conv to see if concat is used
            first_conv_key = 'conv1.conv2d.weight'
            if first_conv_key in icnn:
                in_ch = icnn[first_conv_key].shape[1]
                if in_ch == 4 or in_ch == 1476:  # RGB+M or RGB+M+hyper
                    config['rdnet_guidance'] = 'concat'
                elif in_ch == 3 or in_ch == 1475:
                    config['rdnet_guidance'] = 'gate'

    # Check hyper
    if 'conv1.conv2d.weight' in icnn:
        in_ch = icnn['conv1.conv2d.weight'].shape[1]
        if in_ch >= 1472:
            config['hyper'] = True

    # Check rdnet_in_lap: look at RDNet first conv input channels
    if 'rdnet' in state_dict:
        rd_state = state_dict['rdnet']
        for k in rd_state.keys():
            if 'head.0.weight' in k:
                rd_in = rd_state[k].shape[1]
                config['rdnet_in_lap'] = (rd_in == 15)
                break

    return config


# ---------------------------------------------------------------------------
# FLOPs estimation (simple manual count)
# ---------------------------------------------------------------------------

def count_conv2d_flops(module, input_shape):
    """Estimate FLOPs for Conv2d. Returns (multiply_adds, output_shape)."""
    if not isinstance(module, nn.Conv2d):
        return 0, None

    _, in_c, h, w = input_shape
    out_c = module.out_channels
    k_h, k_w = module.kernel_size
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
    groups = module.groups

    out_h = (h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1

    # Multiply-adds per output element
    madd_per_elem = (in_c // groups) * k_h * k_w
    total_madd = madd_per_elem * out_c * out_h * out_w

    # Each multiply-add = 2 FLOPs
    return total_madd * 2, (1, out_c, out_h, out_w)


def estimate_model_gflops(net, rdnet, lap, img_size=224, hyper=True, use_rdnet=False,
                          rdnet_guidance='concat', gate_type='simple'):
    """
    Estimate GFLOPs for the full forward pass with a dummy input.
    Uses a rough layer-by-layer count.
    """
    H = W = img_size
    total = 0.0

    # Laplacian Pyramid FLOPs
    if lap is not None:
        # 4 scales: 1.0, 0.5, 0.25, 0.125
        for scale in [1.0, 0.5, 0.25, 0.125]:
            h_s = int(H * scale)
            w_s = int(W * scale)
            # Conv2d with groups=3, kernel 3x3
            madd = 3 * 3 * 3 * h_s * w_s  # in_ch//groups=1 * k^2 * out_ch * out_h * out_w
            total += madd * 2  # multiply-add to FLOPs
            if scale != 1.0:
                # Upsample bilinear
                total += 3 * H * W * 4  # rough estimate for bilinear upsampling

    # RDNet FLOPs
    if rdnet is not None:
        rd_h, rd_w = H, W
        # Head: Conv 15→32 k3
        rd_in_ch = 3 + (12 if lap is not None else 0)
        total += 2 * rd_in_ch * 3 * 3 * 32 * rd_h * rd_w
        # 3× ResBlock: each has 2 convs 32→32 k3
        for _ in range(3):
            total += 2 * 32 * 3 * 3 * 32 * rd_h * rd_w
            total += 2 * 32 * 3 * 3 * 32 * rd_h * rd_w
        # Tail: Conv 32→16 k3
        total += 2 * 32 * 3 * 3 * 16 * rd_h * rd_w
        # Tail: Conv 16→1 k3
        total += 2 * 16 * 3 * 3 * 1 * rd_h * rd_w

    # ERRNet FLOPs (DRNet)
    in_ch = 3
    if use_rdnet and rdnet_guidance == 'concat':
        in_ch += 1
    if hyper:
        in_ch += 1472

    # Encoder
    # conv1: in_ch → 256, k1 (bottom_kernel_size=1)
    total += 2 * in_ch * 1 * 1 * 256 * H * W
    # conv2: 256 → 256, k3
    total += 2 * 256 * 3 * 3 * 256 * H * W
    # conv3: 256 → 256, k3, stride=2 → H/2 × W/2
    h2, w2 = H // 2, W // 2
    total += 2 * 256 * 3 * 3 * 256 * h2 * w2

    # 13 residual blocks at H/2 × W/2, each with conv1(k3)+conv2(k3), SE optional
    for _ in range(13):
        total += 2 * 256 * 3 * 3 * 256 * h2 * w2  # conv1
        total += 2 * 256 * 3 * 3 * 256 * h2 * w2  # conv2
        # SE: avg_pool + fc(256→32→256), negligible

    # Gate FLOPs
    if use_rdnet and rdnet_guidance == 'gate':
        if gate_type == 'structure_aware':
            # lap_proj: 12→256 k1 at H/2 × W/2
            total += 2 * 12 * 1 * 1 * 256 * h2 * w2
            # gate: 257→64 k1
            total += 2 * 257 * 1 * 1 * 64 * h2 * w2
            # gate: 64→256 k1
            total += 2 * 64 * 1 * 1 * 256 * h2 * w2
        else:
            # gate_conv: 1→256 k1
            total += 2 * 1 * 1 * 1 * 256 * h2 * w2

    # Decoder
    # deconv1: TransposedConv 256→256 k4 stride2
    total += 2 * 256 * 4 * 4 * 256 * H * W
    # deconv2: Conv 256→256 k3
    total += 2 * 256 * 3 * 3 * 256 * H * W
    # PyramidPooling (if pyramid=True in errnet)
    scales = [4, 8, 16, 32]
    for s in scales:
        total += 256 * h2 * w2  # avg_pool approx
        total += 2 * 256 * 1 * 1 * 64 * 1 * 1  # conv
        total += 64 * h2 * w2 * 4  # upsample approx
    # bottleneck: 512→256 k1
    total += 2 * (256 + len(scales) * 64) * 1 * 1 * 256 * h2 * w2
    # deconv3: Conv 256→3 k1
    total += 2 * 256 * 1 * 1 * 3 * H * W

    # VGG hypercolumn (frozen, inference only)
    if hyper:
        # VGG19 ≈ 19.6 GFLOPs for 224×224, but we only use first 30 layers
        # Rough estimate: ~15 GFLOPs
        total += 15.0 * 1e9

    return total / 1e9  # GFLOPs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    variants = OrderedDict()

    # --- Build all variants ---
    # 1. ERRNet baseline (no RDNet)
    net, _, _ = build_model(use_rdnet=False, hyper=args.hyper)
    variants['ERRNet'] = {'net': net, 'rdnet': None, 'lap': None, 'config': 'baseline'}

    # 2. ERRNet + M concat
    net_c, rdnet_c, lap_c = build_model(use_rdnet=True, rdnet_guidance='concat',
                                         hyper=args.hyper)
    variants['ERRNet + M concat'] = {'net': net_c, 'rdnet': rdnet_c, 'lap': lap_c,
                                      'config': 'concat'}

    # 3. ERRNet + M gate (simple)
    net_gs, rdnet_gs, lap_gs = build_model(use_rdnet=True, rdnet_guidance='gate',
                                            gate_type='simple', hyper=args.hyper)
    variants['ERRNet + M gate (simple)'] = {'net': net_gs, 'rdnet': rdnet_gs, 'lap': lap_gs,
                                              'config': 'gate_simple'}

    # 4. ERRNet + M gate (per_channel)
    net_gp, rdnet_gp, lap_gp = build_model(use_rdnet=True, rdnet_guidance='gate',
                                            gate_type='per_channel', hyper=args.hyper)
    variants['ERRNet + M gate (p-ch)'] = {'net': net_gp, 'rdnet': rdnet_gp, 'lap': lap_gp,
                                           'config': 'gate_per_channel'}

    # 5. ERRNet + M gate (structure_aware)
    net_gsa, rdnet_gsa, lap_gsa = build_model(use_rdnet=True, rdnet_guidance='gate',
                                               gate_type='structure_aware', hyper=args.hyper)
    variants['ERRNet + M gate (SA)'] = {'net': net_gsa, 'rdnet': rdnet_gsa, 'lap': lap_gsa,
                                         'config': 'gate_structure_aware'}

    # --- Load checkpoints if provided ---
    ckpt_map = {
        'baseline': args.ckpt_base,
        'concat': args.ckpt_concat,
        'gate_simple': args.ckpt_simple,
        'gate_per_channel': args.ckpt_per_channel,
        'gate_structure_aware': args.ckpt_sa,
    }

    for name, var in variants.items():
        config = var['config']
        ckpt_path = ckpt_map.get(config)
        if ckpt_path:
            print(f"[i] Loading {ckpt_path} for {name} ...")
            state = load_checkpoint(ckpt_path, device)
            icnn_state = state.get('icnn', state)
            try:
                var['net'].load_state_dict(icnn_state, strict=False)
            except Exception as e:
                print(f"[!] Failed to load net_i for {name}: {e}")
            if var['rdnet'] is not None and 'rdnet' in state:
                try:
                    var['rdnet'].load_state_dict(state['rdnet'], strict=False)
                except Exception as e:
                    print(f"[!] Failed to load rdnet for {name}: {e}")

    # --- Count parameters ---
    # First, count baseline params to compute "Extra Params"
    base_net_params = count_params(variants['ERRNet']['net'])

    # RDNet + LaplacianPyramid params (shared across RDNet variants)
    _, rdnet_ref, lap_ref = variants['ERRNet + M concat']['net'], variants['ERRNet + M concat']['rdnet'], variants['ERRNet + M concat']['lap']
    rdnet_params = count_params(rdnet_ref)
    lap_params = count_params(lap_ref)

    print(f"\n{'='*80}")
    print(f"Input size: {args.img_size}×{args.img_size}  |  Hyper: {args.hyper}")
    print(f"Base ERRNet params: {format_params(base_net_params)}")
    print(f"RDNet params: {format_params(rdnet_params)}  |  LaplacianPyramid: {format_params(lap_params)} (non-learnable)")
    print(f"{'='*80}")

    # --- Print table ---
    header = f"{'Method':<28s} {'Params':>10s} {'Extra':>10s} {'GFLOPs':>10s}"
    print(f"\n{header}")
    print("-" * len(header))

    results = []
    base_gflops = None

    for name, var in variants.items():
        net = var['net']
        rd = var['rdnet']
        lp = var['lap']

        total_p = count_params(net)
        if rd is not None:
            total_p += count_params(rd)
        if lp is not None:
            total_p += count_params(lp)  # LaplacianPyramid has no params (only buffer)

        extra_p = total_p - base_net_params

        gflops = estimate_model_gflops(
            net, rd, lp, img_size=args.img_size, hyper=args.hyper,
            use_rdnet=(rd is not None),
            rdnet_guidance=('concat' if 'concat' in name.lower() else 'gate'),
            gate_type=('structure_aware' if 'SA' in name else
                       'per_channel' if 'p-ch' in name else 'simple')
        )

        if base_gflops is None and 'ERRNet' in name and 'M' not in name:
            base_gflops = gflops

        print(f"{name:<28s} {format_params(total_p):>10s} {format_params(extra_p):>10s} {gflops:>9.1f}")
        results.append((name, total_p, extra_p, gflops))

    # --- Breakdown ---
    print(f"\n{'-'*60}")
    print("Parameter breakdown (ERRNet + M gate, simple):")
    _, rdnet_s, lap_s = variants['ERRNet + M gate (simple)']['net'], variants['ERRNet + M gate (simple)']['rdnet'], variants['ERRNet + M gate (simple)']['lap']
    net_s = variants['ERRNet + M gate (simple)']['net']

    # Count gate params specifically
    gate_params = 0
    for name_p, param in net_s.named_parameters():
        if 'gate_conv' in name_p or 'gate_alpha' in name_p:
            gate_params += param.numel()
    for name_p, param in rdnet_s.named_parameters():
        pass  # RDNet is counted separately

    print(f"  ERRNet backbone:  {format_params(count_params(net_s) - gate_params)}")
    print(f"  Gate params:      {format_params(gate_params)}")
    print(f"  RDNet:            {format_params(count_params(rdnet_s))}")
    print(f"  Total:            {format_params(count_params(net_s) + count_params(rdnet_s))}")

    # Also show per-channel and SA gate param breakdowns
    net_gp_ref = variants['ERRNet + M gate (p-ch)']['net']
    gate_pc_params = sum(p.numel() for n, p in net_gp_ref.named_parameters() if 'gate_conv' in n or 'gate_alpha' in n)
    print(f"\n  Gate (per_channel): {format_params(gate_pc_params)}")

    net_sa_ref = variants['ERRNet + M gate (SA)']['net']
    gate_sa_params = sum(p.numel() for n, p in net_sa_ref.named_parameters() if 'structure_gate' in n or 'gate_conv' in n or 'gate_alpha' in n)
    print(f"  Gate (structure_aware): {format_params(gate_sa_params)}")

    print(f"\n[i] Note: FLOPs are rough estimates using layer-by-layer counting.")
    print(f"[i] For exact FLOPs, use thop: pip install thop")


if __name__ == '__main__':
    main()
