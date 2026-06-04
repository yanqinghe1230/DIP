import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import models.networks as networks
from models.arch.rdnet import LaplacianPyramid, RDNet
from .base_model import BaseModel


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy


def _flag_enabled(data, key, default=False):
    value = data.get(key, default)
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


class RDNetModel(BaseModel):
    def name(self):
        return "rdnet"

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print("--------------------- Model ---------------------")
        print("##################### RDNet #####################")
        networks.print_network(self.net_rd)

    def _eval(self):
        self.net_rd.eval()

    def _train(self):
        self.net_rd.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device("cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")

        self.lap_pyramid = LaplacianPyramid(channels=3).to(self.device)
        in_channels = 3 + self.lap_pyramid.out_channels
        self.net_rd = RDNet(in_channels, out_channels=1).to(self.device)
        networks.init_weights(self.net_rd, init_type=opt.init_type)

        # BCE loss for binary gradient-comparison mask
        self.mask_loss = nn.BCELoss()
        self.lambda_tv = getattr(opt, 'lambda_tv', 0.1)
        self.loss_mask = None
        self.loss_tv = None
        self.mask_gt = None
        self.mask_pred = None

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.net_rd.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd
            )
            self._init_optimizer([self.optimizer_G])

        if opt.no_verbose is False:
            self.print_network()

    def _compute_gradient_mask(self, M, B):
        """Compute sparse binary mask: grad(M) > grad(B).

        Identifies regions where the blended image M has stronger gradients
        than the clean transmission B. Extra gradients are attributed to
        reflections, so the network learns to detect reflection edges rather
        than reconstruct reflection intensity.
        """
        # max across RGB channels for single-channel gradient comparison
        maxM = M.max(dim=1, keepdim=True)[0]
        maxB = B.max(dim=1, keepdim=True)[0]

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=M.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=M.device).view(1, 1, 3, 3)

        gradM_x = F.conv2d(maxM, sobel_x, padding=1)
        gradM_y = F.conv2d(maxM, sobel_y, padding=1)
        gradM = torch.sqrt(gradM_x ** 2 + gradM_y ** 2 + 1e-8)

        gradB_x = F.conv2d(maxB, sobel_x, padding=1)
        gradB_y = F.conv2d(maxB, sobel_y, padding=1)
        gradB = torch.sqrt(gradB_x ** 2 + gradB_y ** 2 + 1e-8)

        mask = (gradM > gradB).float()
        return mask

    def _tv_loss(self, mask):
        """Total Variation loss for spatial smoothness."""
        diff_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
        diff_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
        return diff_h + diff_w

    def set_input(self, data, mode="train"):
        target_r = None
        target_t = None
        data_name = None
        mode = mode.lower()
        if mode == "train":
            input, target_r = data["input"], data["target_r"]
            target_t = data.get("target_t", None)
        elif mode == "eval":
            input, target_r, data_name = data["input"], data["target_r"], data["fn"]
            target_t = data.get("target_t", None)
        elif mode == "test":
            input, data_name = data["input"], data["fn"]
        else:
            raise NotImplementedError("Mode [%s] is not implemented" % mode)

        if len(self.gpu_ids) > 0:
            input = input.to(device=self.gpu_ids[0])
            if target_r is not None and isinstance(target_r, torch.Tensor):
                target_r = target_r.to(device=self.gpu_ids[0])
            if target_t is not None and isinstance(target_t, torch.Tensor):
                target_t = target_t.to(device=self.gpu_ids[0])

        self.input = input
        self.target_t = target_t
        self.data_name = data_name
        self.issyn = not _flag_enabled(data, "real", default=False)

        self.mask_gt = None
        if self.issyn and target_t is not None and isinstance(target_t, torch.Tensor):
            # gradient comparison mask: 1 where blended M has stronger edges
            # than clean transmission B, indicating reflection presence
            self.mask_gt = self._compute_gradient_mask(self.input, target_t)

    def forward(self):
        input_device = self.input.device
        if next(self.net_rd.parameters()).device != input_device:
            self.net_rd = self.net_rd.to(input_device)
        if self.lap_pyramid.kernel.device != input_device:
            self.lap_pyramid = self.lap_pyramid.to(input_device)
        lap = self.lap_pyramid(self.input)
        rd_input = torch.cat([self.input, lap], dim=1)
        self.mask_pred = self.net_rd(rd_input)
        return self.mask_pred

    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.mask_gt is None:
            return

        self.optimizer_G.zero_grad()
        self.loss_mask = self.mask_loss(self.mask_pred, self.mask_gt)
        self.loss_tv = self._tv_loss(self.mask_pred) if self.lambda_tv > 0 else 0.0
        loss_total = self.loss_mask * self.opt.lambda_mask + self.loss_tv * self.lambda_tv
        loss_total.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_mask is not None:
            ret_errors["MaskBCE"] = self.loss_mask.item()
        if self.loss_tv is not None and self.loss_tv != 0.0:
            ret_errors["MaskTV"] = float(self.loss_tv)
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals["input"] = tensor2im(self.input).astype(np.uint8)
        if self.mask_pred is not None:
            ret_visuals["mask_pred"] = tensor2im(self.mask_pred).astype(np.uint8)
        if self.mask_gt is not None:
            ret_visuals["mask_gt"] = tensor2im(self.mask_gt).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = {
            "rdnet": self.net_rd.state_dict(),
            "opt_g": self.optimizer_G.state_dict() if self.isTrain else None,
            "epoch": self.epoch,
            "iterations": self.iterations,
        }
        return state_dict
