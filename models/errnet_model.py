import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch
from models.arch.rdnet import LaplacianPyramid, RDNet

from .base_model import BaseModel
from PIL import Image
from os.path import join


def _torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


def _flag_enabled(data, key, default=False):
    value = data.get(key, default)
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class ERRNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])                
        
        self.input = input
        
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.data_name = data_name

        self.issyn = not _flag_enabled(data, 'real', default=False)
        self.aligned = not _flag_enabled(data, 'unaligned', default=False)

        if getattr(self, 'use_rdnet', False):
            self.mask_gt = None
            if self.issyn and target_r is not None:
                self.mask_gt = target_r.mean(dim=1, keepdim=True).clamp(0, 1)
        
        if target_t is not None:            
            self.target_edge = self.edge_map(self.target_t)         
            
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_i = tensor2im(self.output_i)
            target = tensor2im(self.target_t)

            if self.aligned:
                h = min(output_i.shape[0], target.shape[0])
                w = min(output_i.shape[1], target.shape[1])
                res = index.quality_assess(output_i[:h, :w], target[:h, :w])
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 'transmission_layer', str(self._count)+'.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'blended', str(self._count)+'.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return 
        
        with torch.no_grad():
            output_i = self.forward()
            output_i = tensor2im(output_i)
                # if os.path.exists(join(savedir, name,'t_output.png')):
                #     i = 2
                #     while True:
                #         if not os.path.exists(join(savedir, name,'t_output_{}.png'.format(i))):
                #             Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output_{}.png'.format(i)))
                #             break
                #         i += 1
                # else:
                #     Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output.png'))
            if self.data_name is not None and savedir is not None:                
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class ERRNetModel(ERRNetBase):
    def name(self):
        return 'errnet'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.use_rdnet:
            print('##################### RDNet #####################')
            networks.print_network(self.rdnet)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()
        if self.use_rdnet:
            self.rdnet.eval()

    def _train(self):
        self.net_i.train()
        if self.use_rdnet:
            if self.rdnet_trainable:
                self.rdnet.train()
            else:
                self.rdnet.eval()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device("cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")

        self.use_rdnet = getattr(opt, 'use_rdnet', False)
        self.rdnet_guidance = getattr(opt, 'rdnet_guidance', 'concat')
        self.rdnet = None
        self.lap_pyramid = None
        self.rdnet_trainable = False
        self.mask_loss = None
        self.mask_pred = None
        self.mask_gt = None

        in_channels = 3
        self.vgg = None

        if self.use_rdnet:
            if self.rdnet_guidance not in ('concat', 'gate'):
                raise ValueError('Unsupported rdnet_guidance: %s' % self.rdnet_guidance)

            self.rdnet_no_laplacian = getattr(opt, 'rdnet_no_laplacian', False)
            self.gate_type = getattr(opt, 'gate_type', 'simple')

            # LaplacianPyramid is needed when:
            # (a) RDNet uses Laplacian features as input (!rdnet_no_laplacian), OR
            # (b) structure_aware gate requires Laplacian as conditioning signal
            need_lap = (not self.rdnet_no_laplacian) or \
                       (self.rdnet_guidance == 'gate' and self.gate_type == 'structure_aware')

            if need_lap:
                self.lap_pyramid = LaplacianPyramid(channels=3).to(self.device)
            else:
                self.lap_pyramid = None

            if not self.rdnet_no_laplacian:
                rd_in_channels = 3 + self.lap_pyramid.out_channels
            else:
                rd_in_channels = 3  # RGB only ablation

            self.rdnet = RDNet(rd_in_channels, out_channels=1).to(self.device)
            networks.init_weights(self.rdnet, init_type=opt.init_type)
            if opt.rdnet_path:
                rd_state = _torch_load_compat(opt.rdnet_path, map_location=self.device)
                if isinstance(rd_state, dict) and 'rdnet' in rd_state:
                    rd_state = rd_state['rdnet']
                try:
                    self.rdnet.load_state_dict(rd_state)
                except RuntimeError:
                    print('[i] RDNet weight shape mismatch (likely different input channels '
                          'due to --rdnet_no_laplacian). Loading with strict=False.')
                    missing, unexpected = self.rdnet.load_state_dict(rd_state, strict=False)
                    if missing:
                        print(f'    New params (random init): {missing}')
                    if unexpected:
                        print(f'    Old params (discarded):   {unexpected}')
            self.rdnet_trainable = self.isTrain and not opt.rdnet_freeze
            if not self.rdnet_trainable:
                for param in self.rdnet.parameters():
                    param.requires_grad = False
            if self.rdnet_guidance == 'concat':
                in_channels += 1
        
        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        
        self.gate_type = getattr(opt, 'gate_type', 'simple')
        net_kwargs = {}
        if self.gate_type == 'structure_aware':
            if not self.use_rdnet:
                raise ValueError('gate_type=structure_aware requires use_rdnet=True')
            if self.rdnet_guidance != 'gate':
                raise ValueError('gate_type=structure_aware requires rdnet_guidance=gate')
            net_kwargs['gate_type'] = 'structure_aware'
            net_kwargs['lap_channels'] = self.lap_pyramid.out_channels

        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3, **net_kwargs).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            if self.use_rdnet:
                self.mask_loss = nn.L1Loss()
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            g_params = list(self.net_i.parameters())
            if self.use_rdnet and self.rdnet_trainable:
                g_params += list(self.rdnet.parameters())
            self.optimizer_G = torch.optim.Adam(g_params,
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_mask = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
        
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX

        if self.use_rdnet and self.mask_gt is not None and self.opt.lambda_mask > 0:
            self.loss_mask = self.mask_loss(self.mask_pred, self.mask_gt)
            self.loss_G += self.loss_mask * self.opt.lambda_mask
        
        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input
        gate_i = None
        lap_features = None

        if self.use_rdnet:
            input_device = self.input.device
            if next(self.rdnet.parameters()).device != input_device:
                self.rdnet = self.rdnet.to(input_device)

            # Compute Laplacian features if available (for RDNet and/or gate)
            if self.lap_pyramid is not None:
                if self.lap_pyramid.kernel.device != input_device:
                    self.lap_pyramid = self.lap_pyramid.to(input_device)
                lap_features = self.lap_pyramid(self.input)

            # Build RDNet input: with or without Laplacian
            if self.rdnet_no_laplacian:
                rd_input = self.input  # ablation: RGB only
            else:
                rd_input = torch.cat([self.input, lap_features], dim=1)

            self.mask_pred = self.rdnet(rd_input)
            if self.rdnet_guidance == 'concat':
                input_i = torch.cat([input_i, self.mask_pred], dim=1)
            else:
                gate_i = self.mask_pred
        else:
            self.mask_pred = None

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        if gate_i is not None:
            if not getattr(self.net_i, 'supports_gate', False):
                raise NotImplementedError('net_i does not support gated guidance')
            if self.gate_type == 'structure_aware':
                output_i = self.net_i(input_i, gate=gate_i, lap_features=lap_features)
            else:
                output_i = self.net_i(input_i, gate=gate_i)
        else:
            output_i = self.net_i(input_i)

        self.output_i = output_i

        return output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        if self.loss_mask is not None:
            ret_errors['MaskL1'] = self.loss_mask.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        if self.mask_pred is not None:
            ret_visuals['mask_pred'] = tensor2im(self.mask_pred).astype(np.uint8)
        if self.mask_gt is not None:
            ret_visuals['mask_gt'] = tensor2im(self.mask_gt).astype(np.uint8)

        return ret_visuals

    # ------------------------------------------------------------------
    # Alpha / Gate analysis helpers (P1: gate modulation analysis)
    # ------------------------------------------------------------------

    def get_gate_alpha_stats(self):
        """Return per-channel or scalar alpha statistics from the gate module.

        Returns
        -------
        dict with keys:
            type : 'simple' | 'structure_aware' | None
            For 'simple':   alpha (float)
            For 'structure_aware': alpha_mean, alpha_std, alpha_min, alpha_max,
                                   alpha (np.ndarray, shape [n_feats])
            None if no gate is active.
        """
        net = self.net_i
        gate_type = getattr(self, 'gate_type', 'simple')

        if gate_type == 'simple':
            if hasattr(net, 'gate_alpha') and net.gate_alpha is not None:
                alpha_val = F.softplus(net.gate_alpha).item()
                return {'type': 'simple', 'alpha': alpha_val}
        elif gate_type == 'structure_aware':
            if hasattr(net, 'structure_gate') and net.structure_gate is not None:
                alpha = F.softplus(net.structure_gate.alpha).detach().cpu()
                return {
                    'type': 'structure_aware',
                    'alpha_mean': alpha.mean().item(),
                    'alpha_std': alpha.std().item(),
                    'alpha_min': alpha.min().item(),
                    'alpha_max': alpha.max().item(),
                    'alpha_median': alpha.median().item(),
                    'alpha': alpha.numpy(),
                }
        return {'type': None}

    def register_gate_hooks(self):
        """Register forward hooks to capture gate activation maps.

        Returns
        -------
        dict[str, torch.Tensor]  — mutable dict that will be populated each forward pass.
            'gate_activation' : (1, H, W) spatial map of gate modulation
            'gate_alpha'      : scalar or per-channel softplus(alpha)
        """
        captured = {}

        def _hook_simple(module, input, output):
            # output = feats * (1 + alpha * gate)
            # Capture the spatial gate (1, H, W) averaged over channels
            gate = module._last_gate  # stored in _apply_gate
            captured['gate_activation'] = gate.detach().cpu()
            alpha = F.softplus(module.gate_alpha).detach().cpu()
            captured['gate_alpha'] = alpha

        def _hook_structure_aware(module, input, output):
            # input = (feats, lap_features, mask)
            # We want the A map (before alpha modulation) and alphas
            feats, lap_features, mask = input
            L_f = module.lap_proj(lap_features)
            A = module.gate(torch.cat([L_f, mask], dim=1))  # (B, C, H, W)
            alpha = F.softplus(module.alpha).detach().cpu()   # (C,)
            # Mean over channels for visualisation
            captured['gate_activation'] = A.mean(dim=1, keepdim=True).detach().cpu()
            captured['gate_activation_full'] = A.detach().cpu()  # full per-channel
            captured['gate_alpha'] = alpha

        net = self.net_i
        gate_type = getattr(self, 'gate_type', 'simple')

        if gate_type == 'structure_aware' and hasattr(net, 'structure_gate') and net.structure_gate is not None:
            net.structure_gate.register_forward_hook(_hook_structure_aware)
        elif hasattr(net, 'gate_conv') and net.gate_conv is not None:
            # Hook onto gate_conv for simple gate
            # Patch _apply_gate to store _last_gate
            original_apply = net._apply_gate

            def patched_apply(feats, gate):
                if gate is None:
                    net._last_gate = None
                    return original_apply(feats, gate)
                if gate.shape[1] != 1:
                    gate = gate.mean(dim=1, keepdim=True)
                gate = F.interpolate(gate, size=feats.shape[2:], mode='bilinear', align_corners=False)
                gate_out = torch.sigmoid(net.gate_conv(gate))
                net._last_gate = gate_out
                alpha = F.softplus(net.gate_alpha)
                return feats * (1 + alpha * gate_out)

            net._apply_gate = patched_apply
            # Hook onto the patched method — use a post-forward hook on gate_conv
            net.gate_conv.register_forward_hook(_hook_simple)

        return captured

    # ------------------------------------------------------------------

    @staticmethod
    def _load_state_dict_with_gate_migration(module, state_dict, module_name, target_gate_type):
        """Load state_dict with graceful gate_type migration.

        When migrating checkpoints across gate_type ('simple' ↔ 'structure_aware'),
        gate-specific keys will mismatch. This helper falls back to strict=False,
        allowing encoder/decoder weights to transfer while new gate params are
        randomly initialised.
        """
        try:
            module.load_state_dict(state_dict)
        except RuntimeError as e:
            missing = []
            unexpected = []
            for line in str(e).split('\n'):
                line = line.strip()
                if 'Missing key(s)' in line:
                    missing = [k.strip().strip('"') for k in line.split(':')[1].split(',') if k.strip()]
                if 'Unexpected key(s)' in line:
                    unexpected = [k.strip().strip('"') for k in line.split(':')[1].split(',') if k.strip()]
            all_gate_keys = missing + unexpected
            is_gate_mismatch = all(
                'structure_gate' in k or 'gate_conv' in k or 'gate_alpha' in k
                for k in all_gate_keys
            )
            if is_gate_mismatch and all_gate_keys:
                print(f'[i] {module_name}: gate_type mismatch detected '
                      f'(target={target_gate_type}). '
                      f'Loading compatible weights (strict=False).')
                if missing:
                    print(f'    New params (random init): {missing}')
                if unexpected:
                    print(f'    Old params (discarded):   {unexpected}')
                missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    # Only gate keys should remain — anything else is a real problem
                    non_gate_missing = [k for k in missing_keys
                                        if 'structure_gate' not in k
                                        and 'gate_conv' not in k
                                        and 'gate_alpha' not in k]
                    if non_gate_missing:
                        raise RuntimeError(
                            f'Non-gate keys still missing after migration: {non_gate_missing}')
            else:
                raise

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None
        target_gate_type = getattr(model.opt, 'gate_type', 'simple')

        if icnn_path is None:
            model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = _torch_load_compat(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            ERRNetModel._load_state_dict_with_gate_migration(
                model.net_i, state_dict['icnn'], 'net_i', target_gate_type)
            if model.use_rdnet and 'rdnet' in state_dict:
                model.rdnet.load_state_dict(state_dict['rdnet'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = _torch_load_compat(icnn_path, map_location=torch.device('cpu'))
            ERRNetModel._load_state_dict_with_gate_migration(
                model.net_i, state_dict['icnn'], 'net_i', target_gate_type)
            if model.use_rdnet and 'rdnet' in state_dict:
                model.rdnet.load_state_dict(state_dict['rdnet'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        if self.use_rdnet:
            state_dict.update({'rdnet': self.rdnet.state_dict()})

        return state_dict


class NetworkWrapper(ERRNetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- NetworkWrapper ---------------------')
        networks.print_network(self.net)

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.device = torch.device("cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[31]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
                
            else:
                raise NotImplementedError            
            
            self.loss_dic['t_cx'] = cxloss

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

            # define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
            self._init_optimizer([self.optimizer_D])
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)
        
    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False
                    
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
                
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            # self.loss_G += self.loss_icnn_pixel
            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
            # self.loss_G += self.loss_fm * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        raise NotImplementedError
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
