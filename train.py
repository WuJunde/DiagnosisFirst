# train.py
#!/usr/bin/env	python3

""" train network using pytorch
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
from dataset import GsegDataset, REFUGEDataset, OracleDataset, ISIC2016, OracleDataset_light
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 


args = cfg.parse_args()
# if args.dataset == 'refuge':
#     args.data_path = '../dataset'

GPUdevice = torch.device('cuda', args.gpu_device)
# SEED = 3407
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

param_size = 0
# for param in net.mask_decoder.parameters():
#     param_size += param.nelement() * param.element_size()
# for param in net.prompt_encoder.parameters():
#     param_size += param.nelement() * param.element_size()
for param in net.parameters():
    param_size += param.nelement() * param.element_size()

size_all_mb = (param_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.Adam(list(net.parameters()) + list(sim_net.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    # transforms.RandomCrop((img_size, img_size)),  # padding=10
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((args.image_size//2,args.image_size//2)),
    transforms.Resize((args.image_size,args.image_size)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
])
transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((args.image_size//2, args.image_size//2)),
    transforms.Resize((args.image_size,args.image_size)),
])

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
    isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, mode = 'Training')
    isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, mode = 'Test')

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
elif args.dataset == 'btcv':
    "monia"
    nice_train_loader, nice_test_loader = get_btcv_loader(args)


# train_dataset = GsegDataset(args,args.data_path, ['BinRushed','MESSIDOR'], transform = transform_train, transform_seg = transform_train_seg, mode='Train')
# val_dataset = GsegDataset(args,args.data_path, ['Magrabia'], transform = transform_test, transform_seg = transform_test_seg, mode = 'Val')
# nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
# nice_test_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
# '''data end'''

'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
for epoch in range(settings.EPOCH):
    if args.mod == 'seg':
        net.train()
        time_start = time.time()
        loss = function.train_seg(args, net, optimizer, nice_train_loader, epoch, writer, vis = 10)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            if args.dataset == 'refuge' or args.dataset == 'refuge2':
                tol, (iou_d, iou_c , disc_dice, cup_dice) = function.seg_validate(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU Disk: {iou_d}, IOU Cup: {iou_c}, Dice Disk: {disc_dice}, Dice Cup: {cup_dice} || @ epoch {epoch}.')
            else:
                tol, (eiou, edice) = function.seg_validate(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="checkpoint" + str(epoch))
        else:
            is_best = False
    
    elif args.mod == 'rec':
        net.train()
        time_start = time.time()
        loss = function.train_rec(args, net, optimizer, nice_test_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (iou_d, iou_c , disc_dice, cup_dice) = function.rec_validate(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU Disk: {iou_d}, IOU Cup: {iou_c}, Dice Disk: {disc_dice}, Dice Cup: {cup_dice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="checkpoint" + str(epoch))
        else:
            is_best = False
        
    elif args.mod == 'cls':
        # auc, acc, sen, spec = function.valuation_training(args, net, nice_test_loader, epoch, writer) ##try a quick

        net.train()
        time_start = time.time()
        loss = function.train_baseline(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            auc, acc, sen, spec = function.valuation_training(args, net, nice_test_loader, epoch, writer)
            logger.info(f'auc: {auc},acc: {acc},sen: {sen},spec: {spec}  || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()
                
            if auc > best_tol:
                best_tol = auc
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="checkpoint" + str(epoch))
        else:
            is_best = False
    
    elif args.mod == 'auto':
        # function.val_vae(args, nice_test_loader, epoch, net, writer)

        net.train()
        time_start = time.time()
        loss = function.train_vae(args, net, optimizer, nice_train_loader, epoch, writer, vis = 100)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol = function.val_vae(args, nice_test_loader, epoch, net, writer)
            print('mse loss is', tol)

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="checkpoint" + str(epoch))
        else:
            is_best = False
    
        

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
    elif args.mod == 'monia_seg':
        net.train()
        time_start = time.time()
        global_step = 0
        dice_val_best = 0.0
        global_step_best = 0
        global_step, dice_val_best, global_step_best = function.train_btcv(args, net, optimizer, epoch, nice_train_loader, nice_test_loader, dice_val_best, global_step_best)
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

writer.close()
