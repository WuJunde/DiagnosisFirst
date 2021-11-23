
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
from conf import settings
import time
import cfg
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
loss_function_class = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_function_map = torch.nn.MSELoss()

def train_cls(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None):
    total_number = len(train_loader.dataset)
    groundtruths = []
    predictions = []
    loss_class = 0
    loss_map = 0
    for batch_index, (images, ave_mask, ones, labels_seg, labels_class, name) in enumerate(tqdm(train_loader, total=total_number, desc=f'Epoch {epoch}', unit='img')):

        labels_class = labels_class.to(dtype = torch.float32,device = GPUdevice)
        labels_seg = labels_seg.to(device = GPUdevice)
        images = images.to(device = GPUdevice)

        images = torch.cat((images, labels_seg), 1)
        outputs_class = net(images)

        optimizer.zero_grad()

        outputs_class = outputs_class.squeeze()

        loss_explicit = loss_function_class(outputs_class, labels_class)

        loss = loss_explicit
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

        loss_class += loss_explicit.item()

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, prediction)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    writer.add_scalar('Train/AUC', auc, epoch)
    writer.add_scalar('Train/ACC', acc, epoch)
    writer.add_scalar('Train/SEN', sen, epoch)
    writer.add_scalar('Train/SPEC', spec, epoch)
    writer.add_scalar('Train/loss_class', loss_class/total_number, epoch)
    print('\t Train auc: %.4f' %auc)
    print('\t Train acc: %.4f' % acc)
    print('\t Confusion Matrix:\n %s\n' % str(cm))
    return loss_class/total_number

def train_seg(args, trans_net: nn.Module, trans_optimizer, train_loader,
          epoch, writer, schedulers=None):
    total_number = len(train_loader.dataset)
    hard = 0
    epoch_loss = 0
    criterion_G = nn.BCEWithLogitsLoss()
    # train mode
    trans_net.train()
    trans_optimizer.zero_grad()
    epoch_loss = 0
    mask_type = torch.float32

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_index, (images, labels_seg, ones, masks, labels_class, name) in enumerate(train_loader):
            labels_class = labels_class.to(dtype = mask_type,device = GPUdevice)
            labels_seg = labels_seg.to(dtype = mask_type, device = GPUdevice)
            imgs = images.to(dtype = mask_type, device = GPUdevice)

            '''init'''
            if hard:
                labels_seg = (labels_seg > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)

            '''Train'''
            mask_pred, coarse = trans_net(imgs)
            loss = criterion_G(mask_pred, labels_seg) + criterion_G(coarse, labels_seg)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            epoch_loss += loss.item()
            trans_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(trans_net.parameters(), 0.1)
            trans_optimizer.step()

            # '''vis images'''
            # if ind % 50 == 0:
            #     namecat = ''
            #     for na in name:
            #         namecat = namecat + na + '+'
            #     vis_image(imgs,mask_pred,true_mask_p, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'))

            pbar.update()

    return epoch_loss / total_number

def First_Order_Adversary(args,net:nn.Module,oracle_loader):
    total_number = len(oracle_loader.dataset)
    net = net.to(device = GPUdevice).eval()
    groundtruths = []
    predictions = []
    ind = 0
    for imgs, mask, ones, masks, labels, name in tqdm(oracle_loader, total=total_number, unit='img'):

        obj = objectives.channel('labels', 0)
        true_masks = torch.cat(ones,1).to(dtype = torch.float32,device = GPUdevice) #b,7,w,h
        imgs = imgs.to(dtype = torch.float32,device = GPUdevice)

        '''para and ad'''
        param_f = lambda: para_image(256,256, imgs, true_masks, fft=False, batch = args.b, channels=7)
        all_transforms = [
            transform.pad(32),
            transform.jitter(16),
            transform.random_scale([n/100. for n in range(80, 120)]),
            transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
            transform.jitter(4),
        ]
        map_param_f = lambda: mmapping(256,img = imgs, seg = true_masks, batch=args.b, device = GPUdevice)
        opt = lambda params: torch.optim.Adam(params, 5e-3)

        imgs = elimination(net, obj, param_f=map_param_f, optimizer = opt, transforms=[], show_inline=True, image_name=name, label = int(labels[0]))
        '''end'''
    return 0
