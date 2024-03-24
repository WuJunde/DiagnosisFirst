

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
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim

from lucent.modelzoo.util import get_model_layers
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_function_map = torch.nn.MSELoss()
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
eval_num = 5
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_cls(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None):
    total_number = len(train_loader.dataset)
    groundtruths = []
    predictions = []
    loss_class = 0
    loss_map = 0
    for batch_index, (images, labels_seg, one_seg,  ones, masks, labels_class, name) in enumerate(tqdm(train_loader, total=total_number, desc=f'Epoch {epoch}', unit='img')):

        labels_class = labels_class.to(dtype = torch.float32,device = GPUdevice)
        labels_seg = labels_seg.to(device = GPUdevice)
        one_seg = one_seg.to(dtype = mask_type, device = GPUdevice)
        images = images.to(device = GPUdevice)

        images = torch.cat((images, one_seg), 1)
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


def train_baseline(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None):
    total_number = len(train_loader.dataset)
    groundtruths = []
    predictions = []
    loss_class = 0
    loss_map = 0
    for batch_index, (images, masks, labels_class, name) in enumerate(tqdm(train_loader, total=total_number, desc=f'Epoch {epoch}', unit='img')):
        labels_class = labels_class.to(dtype = torch.float32,device = GPUdevice)
        images = images.to(dtype = torch.float32, device = GPUdevice)
        masks = masks.to(dtype = torch.float32, device = GPUdevice)

        images = torch.cat((images, masks), 1)
        outputs_class = net(images)

        optimizer.zero_grad()
        outputs_class = outputs_class.squeeze()
        # print('output class is:',nn.Sigmoid()(outputs_class).data.cpu().numpy())

        loss_explicit = criterion_G(outputs_class, labels_class)

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

def train_seg(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    criterion_G = nn.BCEWithLogitsLoss()
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for imgs, masks, labels_class, name in train_loader:
            labels_class = labels_class.to(dtype = torch.float32,device = GPUdevice)
            imgs = imgs.to(dtype = torch.float32, device = GPUdevice)
            masks = masks.to(dtype = torch.float32, device = GPUdevice)


            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            if b_size == 1:         ##batch norm cannot handel batch 1
                continue

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''Train'''
            pred = net(imgs)
            loss = criterion_G(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = ''
                    for na in name:
                        namecat = namecat + na + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)

            pbar.update()

    return loss

def validation_btcv(args, epoch, epoch_iterator_val, model):
    model.eval()
    global_step = epoch
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
            '''vis'''
        
            namecat = ''
            for na in batch["image"].meta["filename_or_obj"]:
                na = os.path.split(na)[1]
                namecat = namecat + na + '+'
                for s in range(val_inputs.size(4)):
                    imgs = val_inputs.cpu()[0:1, 0:1, :, :, s]
                    masks = val_labels.cpu()[0:1, 0:1, :, :, s]
                    pred = torch.argmax(val_outputs, dim=1).detach().cpu()[0:1, :, :, s].unsqueeze(0)
                    img_name = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '_slice' + str(s) + '.jpg')
                    # img_name_ori = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '_slice' + str(s) + '_ori')
                    img_name_pred = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '_slice' + str(s) + '_pred')
                    # img_name_gt = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '_slice' + str(s) + '_gt')
                    # np.save(img_name_ori, imgs.numpy())
                    np.save(img_name_pred, pred.numpy())
                    # np.save(img_name_gt, masks.numpy())
                    vis_image(imgs, pred, masks, img_name, reverse=False)

        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()


    return mean_dice_val

def train_btcv(args, model, optimizer, global_step, train_loader,val_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
    if (
        global_step % eval_num == 0 and global_step != 0
    ) or global_step == max_iterations:
        epoch_iterator_val = tqdm(
            val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        dice_val = validation_btcv(args, global_step, epoch_iterator_val, model)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        metric_values.append(dice_val)
        if dice_val > dice_val_best:
            dice_val_best = dice_val
            global_step_best = global_step
            torch.save(
                model.state_dict(), os.path.join(args.path_helper['ckpt_path'], "best_metric_model.pth")
            )
            print(
                "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                    dice_val_best, dice_val
                )
            )
        else:
            print(
                "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                    dice_val_best, dice_val
                )
            )

    return global_step, dice_val_best, global_step_best

def train_rec(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    criterion_G = nn.BCEWithLogitsLoss()
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for imgs, ave_seg, ones, masks , labels_class, name in train_loader:

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = ave_seg.size()
            if b_size == 1:         ##batch norm cannot handel batch 1
                continue

            '''note ave_seg here is not equal to average of ones, ave_seg is hard'''
            true_mask = torch.mean(torch.stack(masks), dim =0).to(dtype = mask_type,device = GPUdevice)
            ave_seg = ave_seg.to(dtype = mask_type,device = GPUdevice)
            prior = torch.ones(b_size,2,w,h)*0.5
            prior = prior.to(dtype = mask_type,device = GPUdevice)
            # true_mask_ave = torch.tensor([0]).expand(b_size, 2, w, w).to(dtype=mask_type, device = GPUdevice)
            # for n, rater_n_true in enumerate(masks):
            #     rater_n_true = rater_n_true.to(dtype=mask_type, device = GPUdevice)
            #     true_mask_ave += torch.mul(rater_n_true, 1 / 7)
            
            # true_mask_ave = ave_seg.to(dtype=mask_type, device = GPUdevice)

            # '''possibility attribution'''
            # #attr_vec = torch.randn(b_size, 6) * 0.15 + 0.5       # a normal distribution with mean: 0.5, variance 0.15
            cond = torch.ones(b_size,7).to(dtype=torch.float32,device = GPUdevice) *  (1/7)
            cond = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b_size,7,2,256,256)
            cond = rearrange(cond,"b c a h w -> b (c a) h w")
            # attr_map = torch.randint(1,11,(b_size,6,w,w)).to(dtype=torch.float32)
            # attr_map_ave = torch.tensor([1]).expand(b_size, 6, w, w).to(dtype=mask_type)
            attr_map = torch.randint(1,11,(b_size,7)).unsqueeze(-1).unsqueeze(-1).expand(b_size, 7, w, w).to(dtype=torch.float32, device=GPUdevice)
            attr_map = nn.Softmax(dim = 1)(attr_map)
            # attr_map_ave = nn.Softmax(dim = 1)(attr_map_ave).to(dtype=torch.float32).to(device=device)

            '''preprocess the true_masks'''
            true_mask_ave = torch.tensor([0]).expand(b_size, 2, w, w).to(dtype=mask_type, device = device)
            true_mask_marg = []
            for n, rater_n_true in enumerate(masks):
                rater_n_true = rater_n_true.to(dtype=mask_type, device = device)
                true_mask_ave += torch.mul(rater_n_true, 1 / 7)  # size [b,2,256,256]
                true_mask_marg.append(rater_n_true)
            '''end'''

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            true_mask_p = true_mask_ave
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''Train'''
            for rect in range(3):
                net.train()
                mask_pred, aux = net(imgs,cond)
                preds,  mergfs =  aux['maps'], aux['mergfs']
                mpt = F.sigmoid(mask_pred)

                pred_stack = torch.stack(preds) #7,b,c,w,w
                pred_stack_t = F.sigmoid(pred_stack)
                rater_stack = torch.stack(true_mask_marg)
                self_pred = (pred_stack_t ) * torch.div(pred_stack_t, torch.sum(pred_stack_t, dim = 0, keepdim=True))


                cond = rearrange(self_pred, "a b c h w -> b (a c) h w").contiguous().detach() #b,7c,w,w
                final_pred = torch.sum(self_pred, dim = 0 ) 
                true_mask = torch.sum((rater_stack * torch.div(pred_stack_t.data, torch.sum(pred_stack_t.data, dim = 0, keepdim=True))), dim = 0 )
                
                # assert torch.sum(torch.div(pred_stack_t.data, torch.sum(pred_stack_t.data, dim = 0, keepdim=True))) == (b_size*2*w*h)
                # loss = criterion_G(pred_stack, rater_stack) + criterion_G(mask_pred, torch.clamp(true_mask_p, min = 0, max = 1))
                # true_mask_p = torch.sum((torch.stack(masks, dim = 1).to(dtype = mask_type,device = GPUdevice) * cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b_size,7,2,128,128)), dim = 1)
                '''shuffle loss'''
                indexes = torch.randperm(pred_stack_t.shape[0])
                shuffle_pred = pred_stack_t[indexes]
                shuffle_rater = rater_stack[indexes]

                cond_pred = (shuffle_pred ) * torch.div(pred_stack_t, torch.sum(pred_stack_t, dim = 0, keepdim=True))
                cond_pred = rearrange(cond_pred, "a b c h w -> b (a c) h w").contiguous()

                cond_rater = (shuffle_rater ) * torch.div(pred_stack_t, torch.sum(pred_stack_t, dim = 0, keepdim=True))
                cond_rater = rearrange(cond_rater, "a b c h w -> b (a c) h w").contiguous().detach()

                _, auxp = net(imgs, cond_pred, mod = 'shuffle')
                _, auxr = net(imgs, cond_rater, mod = 'shuffle')
                mp = auxp['mergfs']
                mr = auxr['mergfs']

                shuffle_loss = torch.nn.CosineEmbeddingLoss()(mp[1],mr[1].detach(),torch.tensor(1).to(device = GPUdevice))
                '''end'''

                # '''SSIM LOSS'''
                # ssim_loss = pytorch_ssim.SSIM(window_size = 11)
                # true_mask_p = torch.clamp(true_mask_p, min = 0, max = 1)
                # agg_loss = ssim_loss(mask_pred[:,0,:,:].unsqueeze(1), true_mask_p[:,0,:,:].unsqueeze(1)) + ssim_loss(mask_pred[:,1,:,:].unsqueeze(1), true_mask_p[:,1,:,:].unsqueeze(1))
                # '''END'''

                loss = nn.BCELoss()(torch.clamp(final_pred, min = 0, max = 1), torch.clamp(true_mask, min = 0, max = 1)) + criterion_G(mask_pred, torch.clamp(true_mask, min = 0, max = 1)) + shuffle_loss
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()
                loss.backward()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
                true_mask_p = true_mask

                '''vis images'''
                if vis:
                    if ind % vis == 0:
                        namecat = ''
                        for na in name:
                            namecat = namecat + na + '+_rec' + str(rect) + '+'
                        vis_image(imgs,mpt,true_mask, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=True)

            pbar.update()

    return loss

def train_sim(args, trans_net: nn.Module, trans_optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
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
        for ind, (images, labels_seg, ones, masks, labels_class, name) in enumerate(train_loader):
            labels_class = labels_class.to(dtype = mask_type,device = GPUdevice)
            labels_seg = labels_seg.to(dtype = mask_type, device = GPUdevice)
            imgs = images.to(dtype = mask_type, device = GPUdevice)

            tm_mul = torch.mean(torch.stack(masks), dim =0).to(dtype = mask_type,device = GPUdevice)

            '''init'''
            if hard:
                tm_mul = (tm_mul > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)

            '''Train'''
            mask_pred, coarse= trans_net(imgs)
            loss = criterion_G(mask_pred, tm_mul) + criterion_G(coarse, labels_seg)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            epoch_loss += loss.item()
            trans_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(trans_net.parameters(), 0.1)
            trans_optimizer.step()

            '''vis images'''
            if ind % 50 == 0:
                namecat = ''
                for na in name:
                    namecat = namecat + na + '+'
                vis_image(imgs,mask_pred,tm_mul, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'))

            pbar.update()

    return epoch_loss / total_number


def valuation_training(args, net: nn.Module, valuation_loader,
          epoch, writer = None, schedulers=None):

    test_loss_seg = 0.0 # cost function error
    test_loss_class = 0.0
    total_number = len(valuation_loader.dataset)
    groundtruths = []
    predictions = []
    for batch_index, (images, masks, labels_class, name) in enumerate(tqdm(valuation_loader, total=total_number, desc=f'Epoch {epoch}', unit='img')):
        labels_class = labels_class.to(dtype = torch.float32,device = GPUdevice)
        images = images.to(dtype = torch.float32, device = GPUdevice)
        masks = masks.to(dtype = torch.float32, device = GPUdevice)
        labels_class = labels_class.to(dtype = torch.float32,device = GPUdevice)

        images = torch.cat((images, masks), 1)

        outputs_class = net(images)
        outputs_class = outputs_class.squeeze()

        loss_class = criterion_G(outputs_class, labels_class)
        test_loss_class += loss_class.item()
        # print('loss is:',loss_class.item())

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)

    # with open("./mh.txt","w") as f:
    #     for j in range(len(groundtruths)):
    #         f.write(str(predictions[j]))
    #         f.write(':')
    #         f.write(str(groundtruths[j]))
    #         f.write('\n')
    #add informations to tensorboard
    # writer.add_scalar('valuation/Average class loss', test_loss_class / total_number, epoch)
    # writer.add_scalar('valuation/AUC', auc, epoch)
    # writer.add_scalar('valuation/ACC', acc, epoch)
    # writer.add_scalar('valuation/SEN', sen, epoch)
    # writer.add_scalar('valuation/SPEC', spec, epoch)
    print('\t Test auc: %.4f' %auc)
    print('\t Test acc: %.4f' % acc)
    print('\t Test sen: %.4f' % sen)
    print('\t Test spec: %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    return auc, acc, sen, spec


def rec_validate(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    criterion_G = nn.BCEWithLogitsLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, (imgs, ave_seg, ones, masks , labels_class, name) in enumerate(val_loader):
            imgs = imgs.to(device=GPUdevice, dtype=torch.float32)
            b_size = imgs.size(0)
            if b_size == 1:         ##batch norm cannot handel batch 1
                continue
            w = masks[0].size(2)

            with torch.no_grad():
                cond = torch.ones(b_size,7).to(dtype=torch.float32,device = GPUdevice) *  (1/7)
                cond = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b_size,7,2,256,256)
                cond = rearrange(cond,"b c a h w -> b (c a) h w")
                true_mask_ave = torch.mean(torch.stack(masks), dim =0).to(dtype = mask_type,device = GPUdevice)
                true_mask_p = true_mask_ave
                
                '''Flow'''
                for rect in range(3):
                    mask_pred,preds,cond = net(imgs,cond)
                    mpt = F.sigmoid(mask_pred)

                    pred_stack = torch.stack(preds) #7,b,c,w,w
                    pred_stack_t = F.sigmoid(pred_stack)
                    rater_stack = torch.stack(masks).to(device=GPUdevice, dtype=torch.float32)

                    final_pred = torch.sum(
                        (pred_stack_t * torch.div(pred_stack_t, torch.sum(pred_stack_t, dim = 0, keepdim=True))), dim = 0 
                        ) 

                    true_mask = torch.sum((rater_stack * torch.div(pred_stack_t.data, torch.sum(pred_stack_t.data, dim = 0, keepdim=True))), dim = 0 )

                    # tot += criterion_G(pred_stack, rater_stack).item()
                    tot += nn.BCELoss()(torch.clamp(final_pred, min = 0, max = 1), torch.clamp(true_mask, min = 0, max = 1)) + criterion_G(mask_pred, torch.clamp(true_mask_p, min = 0, max = 1))
                    true_mask_p = true_mask

                    '''vis images'''
                    if ind % 10 == 0:
                        namecat = ''
                        for na in name:
                            img_name = na.split('\\')[-1]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs,mpt,true_mask_ave, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=True)
                

                temp = eval_seg(mpt, true_mask_ave, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    return tot/ (n_val * 3) , tuple([a/n_val for a in mix_res])

def seg_validate(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    criterion_G = nn.BCEWithLogitsLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, (imgs, masks, labels_class, name) in enumerate(val_loader):
            imgs = imgs.to(device=GPUdevice, dtype=torch.float32)
            b_size = imgs.size(0)
            if b_size == 1:         ##batch norm cannot handel batch 1
                continue
            w = masks[0].size(2)

            with torch.no_grad():

                # attr_map = seed.unsqueeze(-1).unsqueeze(-1).expand(b_size, 7, w, w).to(dtype=torch.float32, device=GPUdevice)
                # attr_map = nn.Softmax(dim = 1)(attr_map)
                # true_mask_p = torch.tensor([0]).expand(b_size, 2, w, w).to(dtype=mask_type, device = GPUdevice)

                # true_mask_ave = torch.mean(torch.stack(masks), dim =0).to(dtype = mask_type,device = GPUdevice)

                #for no multi
                masks = masks.to(dtype = mask_type,device = GPUdevice)
                true_mask_p = masks

                # for n, rater_n_true in enumerate(masks):
                #     rater_n_true = rater_n_true.to(dtype=mask_type, device = device)
                #     true_mask_p += torch.mul(rater_n_true,attr_map[:,n,:,:].unsqueeze(1))
                
                '''Train'''
                pred = net(imgs)

                tot += criterion_G(pred, true_mask_p)
                '''vis images'''
                if ind % args.vis == 0:
                    namecat = ''
                    for na in name:
                        img_name = na.split('\\')[-1]
                        namecat = namecat + img_name + '+'
                    vis_image(imgs,pred,true_mask_p, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=True)
                

                temp = eval_seg(pred, true_mask_p, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def First_Order_Adversary(args,net:nn.Module,oracle_loader):
    total_number = len(oracle_loader.dataset)
    net = net.to(device = GPUdevice).eval()
    # print(get_model_layers(net))
    groundtruths = []
    predictions = []
    ind = 0
    for imgs, ave_mask, ones, masks, labels, name in tqdm(oracle_loader, total=total_number, unit='img'):
        # print('label is:',int(labels[0]))
        # if int(labels[0]) == 0:
        #     continue
        obj = objectives.channel('labels', 0)
        true_masks = torch.cat(ones,1).to(dtype = torch.float32,device = GPUdevice) #b,7,w,h
        print("true masks size is",true_masks.size)
        # true_masks = allone.to(dtype = torch.float32,device = GPUdevice)
        imgs = imgs.to(dtype = torch.float32,device = GPUdevice)

        '''init loss '''
        # outputs_class = net(torch.cat((imgs, torch.mean(true_masks,1,keepdim = True)), 1))
        
        outputs_class = net(torch.cat((imgs, ave_mask.to(dtype = torch.float32,device = GPUdevice)), 1))
        # outputs_class = net(imgs)
        outputs_class = outputs_class.squeeze()

        labels = labels.to(dtype = torch.float32,device = GPUdevice)

        loss_class = criterion_G(outputs_class, labels)
        print('inital loss is:',loss_class.item())
        '''end'''

        '''para and ad'''
        param_f = lambda: para_image(256,256, imgs, seg = true_masks, fft=True, batch = args.b, channels=7)
        all_transforms = [
            transform.pad(32),
            transform.jitter(16),
            transform.random_scale([n/100. for n in range(80, 120)]),
            transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
            transform.jitter(4),
        ]
        cppn_param_f = lambda: cppn(args, size = 256,img = imgs, seg = true_masks, batch=args.b, device = GPUdevice)
        opt = lambda params: torch.optim.Adam(params, 5e-3)
        
        imgs = render_vis(args, net, obj, imgs, param_f=param_f, optimizer = opt, transforms=[], show_inline=True, image_name=name,save_image=True, label = int(labels[0]))
        '''end'''

        '''last loss'''
        outputs_class = net(imgs)
        outputs_class = outputs_class.squeeze()

        loss_class = criterion_G(outputs_class, labels)
        print('after loss is',loss_class.item())
        '''end'''

        groundtruths.extend(labels.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    print('\t Test auc: %.4f' %auc)
    print('\t Test acc: %.4f' % acc)
    print('\t Confusion Matrix:\n %s\n' % str(cm))
    return auc

def Adversary_Fusion(oracle_loader):
    total_number = len(oracle_loader.dataset)
    # print(get_model_layers(net))
    groundtruths = []
    predictions = []
    ind = 0
    for imgs, ave_mask, ones, masks, labels, name in tqdm(oracle_loader, total=total_number, unit='img'):
        # print('label is:',int(labels[0]))
        # if int(labels[0]) == 0:
        #     continue
        obj = objectives.channel('labels', 0)
        true_masks = torch.cat(ones,1).to(dtype = torch.float32,device = GPUdevice) #b,7,w,h
        print("true masks size is",true_masks.size)
        # true_masks = allone.to(dtype = torch.float32,device = GPUdevice)
        imgs = imgs.to(dtype = torch.float32,device = GPUdevice)

        '''init loss '''
        # outputs_class = net(torch.cat((imgs, torch.mean(true_masks,1,keepdim = True)), 1))
        
        # outputs_class = net(torch.cat((imgs, ave_mask.to(dtype = torch.float32,device = GPUdevice)), 1))
        outputs_class = net(imgs)
        outputs_class = outputs_class.squeeze()

        labels = labels.to(dtype = torch.float32,device = GPUdevice)

        loss_class = loss_function_class(outputs_class, labels)
        print('inital loss is:',loss_class.item())
        '''end'''

        '''para and ad'''
        param_f = lambda: para_image(256,256, imgs, fft=False, batch = args.b)
        all_transforms = [
            transform.pad(32),
            transform.jitter(16),
            transform.random_scale([n/100. for n in range(80, 120)]),
            transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
            transform.jitter(4),
        ]
        cppn_param_f = lambda: cppn(256,img = imgs, seg = true_masks, batch=args.b, device = GPUdevice)
        opt = lambda params: torch.optim.Adam(params, 5e-3)
        
        imgs = render_vis(args, net, obj, param_f=cppn_param_f, optimizer = opt, transforms=[], show_inline=True, image_name=name,save_image=True, label = int(labels[0]))
        '''end'''

        '''last loss'''
        outputs_class = net(imgs)
        outputs_class = outputs_class.squeeze()

        loss_class = loss_function_class(outputs_class, labels)
        print('after loss is',loss_class.item())
        '''end'''

        groundtruths.extend(labels.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    print('\t Test auc: %.4f' %auc)
    print('\t Test acc: %.4f' % acc)
    print('\t Confusion Matrix:\n %s\n' % str(cm))
    return auc



def First_Order_Adversary(args,net:nn.Module,oracle_loader):
    total_number = len(oracle_loader.dataset)
    net = net.to(device = GPUdevice).eval()
    groundtruths = []
    predictions = []
    ind = 0
    for imgs, mask, one_seg, ones, masks, labels, name in tqdm(oracle_loader, total=total_number, unit='img'):

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
