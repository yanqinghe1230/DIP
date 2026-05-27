from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.model = 'rdnet_model'
opt.name = opt.name or 'rdnet_pretrain'
opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# processed datasets prepared by datasets/prepare_train_data.py
# use synthetic data only for RDNet pretraining

datadir = './datasets/processed_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataloader = datasets.DataLoader(
    train_dataset, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt)

while engine.epoch < opt.nEpochs:
    engine.train(train_dataloader)
