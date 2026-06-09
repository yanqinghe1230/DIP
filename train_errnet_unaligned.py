from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data

opt = TrainOptions().parse()

# --- RDNet + Gate configuration (must match the aligned checkpoint) ---
opt.model = 'errnet_model'
opt.use_rdnet = True
opt.rdnet_guidance = 'gate'           # 'gate' for soft modulation (required by structure_aware)
opt.gate_type = 'structure_aware'     # 'simple' or 'structure_aware'
opt.rdnet_freeze = False              # joint fine-tuning: RDNet weights are updated together with ERRNet

# Load aligned pre-trained weights via --icnn_path (CLI arg).
# Example: --icnn_path checkpoints/errnet_gate/errnet_latest.pt
# Using --icnn_path (not --resume) starts with a fresh optimizer state,
# which is the standard fine-tuning setup.

opt.name = opt.name or 'errnet_unaligned_sa'
opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.serial_batches = True
    opt.no_flip = True

cudnn.benchmark = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'
raw_datadir = './datasets/raw_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')
datadir_unaligned = join(raw_datadir, 'Dataset/DSLR/unaligned_train250')

train_dataset = datasets.CEILDataset(datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size)
train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_unaligned, train_dataset_real], [0.25,0.5,0.25])


train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)


engine = Engine(opt)

# Verify RDNet is trainable (not frozen)
if opt.use_rdnet:
    rdnet_trainable = any(p.requires_grad for p in engine.model.rdnet.parameters())
    print(f'[i] RDNet trainable: {rdnet_trainable} (expected: True)')

"""Main Loop"""
def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

# Fine-tuning: use a lower initial LR than aligned training
engine.model.opt.lambda_gan = 0.01
set_learning_rate(1e-4)
while engine.epoch < 80:
    if engine.epoch == 65:
        set_learning_rate(5e-5)
    if engine.epoch == 70:
        set_learning_rate(1e-5)

    engine.train(train_dataloader_fusion)
