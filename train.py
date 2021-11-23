

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
from dataset import GsegDataset, OracleDataset
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
# SEED = 3407
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)


optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-04, weight_decay=0, amsgrad=False)
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

'''data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    # transforms.RandomCrop((img_size, img_size)),  # padding=10
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.ToTensor(),
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

'''oracle'''
train_dataset = OracleDataset(args, args.data_path, transform = transform_train, transform_seg = transform_train_seg, mode = 'Train')
test_dataset = OracleDataset(args, args.data_path, transform = transform_test, transform_seg = transform_test_seg, mode = 'Test')

nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
'''data end'''

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
        loss = function.train_seg(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        
    elif args.mod == 'cls':
        net.train()
        time_start = time.time()
        loss = function.train_cls(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        
writer.close()
