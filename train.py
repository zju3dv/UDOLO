# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, distributed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps',
                    help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160',
                    help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--distributed', action='store_true', help='Use distributed data parallel.')

parser.add_argument('--feedback', action='store_true', help='')
parser.add_argument('--aug_score_based', action='store_true', help='Use score in sample 3d boxes.')
parser.add_argument('--use_kalman', action='store_true', help='If save result.')
parser.add_argument('--voxel_size', type=tuple, default=(0.05, 0.05), help='voxel size for voxel map')
parser.add_argument('--map_area', type=tuple, default=(400, 400), help='voxel map area')
parser.add_argument('--detect_thresh', type=int, default=3, help='threshold for number of detection')
parser.add_argument('--score_decay', type=float, default=1.0, help='score decay')
parser.add_argument('--boxes_decay', type=float, default=0.0, help='boxes decay')
parser.add_argument('--valid_score', type=float, default=0.1,
                    help='valid score for backend box which condition frontend')
parser.add_argument('--num_point_thresh', type=int, default=10, help='number of point')
parser.add_argument('--num_proposal_per_box', type=int, default=1, help='number of point')
parser.add_argument('--iou_thresh', type=float, default=0.5, help='number of point')
parser.add_argument('--top_n', type=int, default=300, help='number of proposal')
parser.add_argument('--top_n_pred', type=int, default=256, help='number of prediction proposal')
parser.add_argument('--view_coord', type=str, default='camera', help='view coord')
parser.add_argument('--world_coord', type=str, default='depth', help='world coord')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

if torch.cuda.device_count() > 1:
    if FLAGS.distributed is True:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        torch.cuda.set_device(FLAGS.local_rank)
else:
    FLAGS.distributed = False

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)' % (LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s' % (LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Create Dataset and Dataloader
if FLAGS.dataset == "scannet":
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetSVDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetSVDatasetConfig

    DATASET_CONFIG = ScannetSVDatasetConfig()
    TRAIN_DATASET = ScannetSVDetectionDataset('train', num_points=NUM_POINT,
                                              augment=True,
                                              use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                              fix_seed=False)
    TEST_DATASET = ScannetSVDetectionDataset('val', num_points=NUM_POINT,
                                             augment=False,
                                             use_color=FLAGS.use_color, use_height=(not FLAGS.no_height), fix_seed=True)

else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)

if FLAGS.distributed:
    train_sampler = distributed.DistributedSampler(TRAIN_DATASET)
    test_sampler = distributed.DistributedSampler(TEST_DATASET)
else:
    train_sampler = None
    test_sampler = None
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET,
                              batch_size=BATCH_SIZE // torch.cuda.device_count() if FLAGS.distributed else BATCH_SIZE,
                              shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4,
                              worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET,
                             batch_size=BATCH_SIZE // torch.cuda.device_count() if FLAGS.distributed else BATCH_SIZE,
                             shuffle=(test_sampler is None), sampler=test_sampler, num_workers=4,
                             worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model)  # import network module
device = torch.device("cuda:{}".format(FLAGS.local_rank) if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

net = MODEL.VoteNet(num_class=DATASET_CONFIG.num_class,
                    num_heading_bin=DATASET_CONFIG.num_heading_bin,
                    num_size_cluster=DATASET_CONFIG.num_size_cluster,
                    mean_size_arr=DATASET_CONFIG.mean_size_arr,
                    input_feature_dim=num_input_channel,
                    FLAGS=FLAGS)
net = net.cuda()
if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    if FLAGS.distributed:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank,
                                                  # this should be removed if we update BatchNorm stats
                                                  broadcast_buffers=False,
                                                  )
    else:
        net = nn.DataParallel(net)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % FLAGS.local_rank}
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
    net.module.load_state_dict(checkpoint['model_state_dict'])
    # net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
               'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
               'per_class_proposal': True, 'conf_thresh': 0.05,
               'dataset_config': DATASET_CONFIG}


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    net.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):

        # Forward pass
        optimizer.zero_grad()
        loss, end_points = net(batch_data_label, DATASET_CONFIG)

        # Update parameters.
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0 and FLAGS.local_rank == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            TRAIN_VISUALIZER.log_scalars({key: stat_dict[key] / batch_interval for key in stat_dict},
                                         (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        # Save checkpoint
        if FLAGS.local_rank == 0:
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss,
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, str(epoch) + '_' + 'checkpoint.tar'))


if __name__ == '__main__':
    train(start_epoch)
