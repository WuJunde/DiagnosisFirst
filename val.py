# train.py
#!/usr/bin/env	python3

""" valuate network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import GsegDataset, REFUGEDataset, OracleDataset_light, AdvDataset, ISIC2016
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function


args = cfg.parse_args()
# if args.dataset == 'refuge' or args.dataset == 'refuge2':
#     args.data_path = '../dataset'

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

# args.path_helper = checkpoint['path_helper']
# logger = create_logger(args.path_helper['log_path'])
# print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

# args.path_helper = set_log_dir('logs', args.exp_name)
# logger = create_logger(args.path_helper['log_path'])
# logger.info(args)

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

# '''tongren data:'''
# nice_train_loader = get_valuation_dataloader(
#     args,
#     num_workers=args.w,
#     batch_size=args.b,
#     shuffle=True
# )

# nice_val_loader = get_test_dataloader(
#     args,
#     num_workers=args.w,
#     batch_size=args.b,
#     shuffle=False
# )
'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    # transforms.RandomCrop((img_size, img_size)),  # padding=10
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((args.image_size//2,args.image_size//2)),
    transforms.Resize((args.image_size,args.image_size)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.image_size, args.image_size)),
    
])
# val_dataset = GsegDataset(args,args.data_path, ['BinRushed','MESSIDOR'], transform = transform_test, transform_seg = transform_test_seg, mode = 'Val')
# gseg_val_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
'''data end'''

if args.dataset == 'tongren':
    '''tongren dataset'''
    nice_train_loader = get_training_dataloader(args, batch_size=16, num_workers=2, shuffle=True)
    nice_test_loader = get_valuation_dataloader(args, batch_size=16, num_workers=2, shuffle=True)
    '''end'''
elif args.dataset == 'refuge2':
    '''REFUGE2 DATA'''
    refuge_train_dataset = REFUGEDataset(args, args.data_path, transform = transform_train, transform_seg = transform_train_seg, mode = 'Train')
    refuge_val_dataset = REFUGEDataset(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Val')
    refuge_test_dataset = REFUGEDataset(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    '''END'''
elif args.dataset == 'isic':
    '''isic data'''
    isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_seg = transform_train_seg, mode = 'Training')
    isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''
elif args.dataset == 'refuge':
    '''oracle'''
    # train_dataset = OracleDataset(args, args.data_path, transform = transform_train, transform_seg = transform_train_seg, mode = 'Train')
    # test_dataset = OracleDataset(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')
    train_dataset = OracleDataset_light(args, args.data_path, transform = transform_train, transform_seg = transform_train_seg, mode = 'Train')
    test_dataset = OracleDataset_light(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    '''data end'''
elif args.dataset == 'isic_singmulti':
    '''isic single multi'''
    val_dataset = SingMulti(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')
    # adv_dataset = AdvDataset(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')
    nice_val_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    '''end'''
elif args.dataset == 'btcv':
    "monia"
    nice_train_loader, nice_test_loader = get_btcv_loader(args)

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

if args.mod == 'seg':
    net.eval()
    tol, (iou_d, iou_c , disc_dice, cup_dice) = function.seg_validate(args, nice_val_loader, start_epoch, net)
    logger.info(f'Total score: {tol}, IOU Disk: {iou_d}, IOU Cup: {iou_c}, Dice Disk: {disc_dice}, Dice Cup: {cup_dice} || @ epoch {start_epoch}.')

elif args.mod == 'cls':
    net.eval()
    # tol, iou_d, iou_c , disc_dice, cup_dice = function.seg_validate(args, gseg_val_loader, start_epoch, net)
    # logger.info(f'Total score: {tol}, IOU Disk: {iou_d}, IOU Cup: {iou_c}, Dice Disk: {disc_dice}, Dice Cup: {cup_dice} || @ epoch {start_epoch}.')
    auc = function.valuation_training(args, net,nice_val_loader, start_epoch)
    logger.info(f'Total score: {auc} || @ epoch {start_epoch}.')
elif args.mod == 'val_ad':
    net.eval()
    auc = function.First_Order_Adversary(args,net,nice_train_loader)
    logger.info(f'Total score: {auc} || @ epoch {start_epoch}.')
elif args.mod == 'monia_seg':
    net.eval()
    auc = function.validation_btcv(nice_test_loader)
    logger.info(f'Total score: {auc} || @ epoch {start_epoch}.')
        #start to save best performance model after learning rate decay to 0.01
        # if best_acc < acc:
        #     is_best = True
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'model': args.net,
        #         'state_dict': net.module.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'best_tol': best_acc,
        #         'path_helper': args.path_helper,
        #     }, is_best, args.path_helper['ckpt_path'], filename="checkpoint")
        #     best_acc = acc
        #     continue

        # if not epoch % settings.SAVE_EPOCH:
        #     print('Saving regular checkpoint')
        #     print(checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'model': args.net,
        #         'state_dict': net.module.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'best_tol': best_acc,
        #         'path_helper': args.path_helper,
        #     }, False, args.path_helper['ckpt_path'], filename="checkpoint")
