
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate


class GsegDataset(Dataset):
    def __init__(self, args, data_path , DF, transform = None, transform_seg = None, mode = 'Train'):

        self.DF = pd.DataFrame(
            columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                     'discFlag','rater'])
        for split in DF:
            DF_all = pd.read_csv(data_path + '/' + 'Glaucoma_multirater_' + split + '.csv', encoding='gbk')

            DF_this = DF_all.loc[DF_all['rater'] == 0]
            DF_this = DF_this.reset_index(drop=True)
            DF_this = DF_this.drop('Unnamed: 0', 1)
            self.DF = pd.concat([self.DF, DF_this])

        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg
        self.scale_size = args.image_size
        self.mode = mode
        self.out_size = args.image_size

    @classmethod
    def preprocess(cls, pil_img, is_mask,PT = 0):
        # w, h = pil_img.size
        # newW, newH = scale_size, scale_size
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        # if is_mask:
        #     pil_img = pil_img.resize((out_size, out_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if PT:
            img_nd = rotate(
                cv2.linearPolar(img_nd, (256, 256), 256, cv2.WARP_FILL_OUTLIERS), -90)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        if is_mask:                 ## background 0, disc 1, cup 2
            disc = img_nd.copy()
            #disc[disc == 1] = 0
            disc[disc != 1] = 0
            cup = img_nd.copy()
            cup[cup != 0] = 1
            img_nd = np.dstack((disc,cup))
            disc = disc[:,:,0]*255
            cup = cup[:,:,0]*255
            return Image.fromarray(disc.astype(np.uint8)),Image.fromarray(cup.astype(np.uint8))

        # # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))

        return Image.fromarray(img_nd)

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']

        fullPathName = os.path.join(self.data_path, imgName)
        # pt_name = imgName.split('\\')[-1].split('.')[0] + '.pt'
        # print('====fullPathName is', fullPathName)
        # pt_path = os.path.join(fullPathName.rsplit('\\',1)[0], pt_name).replace('\\', '/')
        fullPathName = fullPathName.replace('\\', '/')

        Img = Image.open(fullPathName).convert('RGB')

        # Img = self.preprocess(Img, 0)
        if self.transform:
            Img = self.transform(Img)

        # '''Get prior'''
        # prior = torch.load(pt_path)

        """Get the segmentation images"""
        masks = []
        data_path = self.data_path
        for n in range(1,7):     # n:1-6
            maskName = self.DF.loc[index, 'maskName'].replace('FinalLabel','Rater'+str(n))
            fullPathName = os.path.join(data_path, maskName)
            fullPathName = fullPathName.replace('\\', '/')

            Mask = Image.open(fullPathName).convert('L')

            disc, cup = self.preprocess(Mask, 1)
            if self.transform_seg:
                disc = self.transform_seg(disc)
                cup = self.transform_seg(cup)
            Mask = torch.cat((disc,cup),0)

            masks.append(Mask)

        return Img, masks, Mask,masks,masks, fullPathName.split('/')[-1]


    def __len__(self):
        return len(self.DF)

class Dataset_DiscRegion(Dataset):
    def __init__(self, data_path, DF, transform = None, transform_seg = None):
        if DF[0] == 'train' or DF[0] == 'val' or DF[0] == 'imp' or DF[0] == 'ORIGAtrain' or DF[0] == 'ORIGAimp' or DF[0] == 'ORIGAtest':
            self.DF = pd.read_csv(data_path + '/' + 'Glaucoma' + '_' + DF[0] + '.csv', encoding='gbk')
        elif DF[0] == 'LAG_train' or DF[0] == 'LAG_test':
            self.DF = pd.read_csv(data_path + '/' + DF[0] + '.csv',encoding='gbk')
        elif DF[0] == 'REFUGETrain' or DF[0] == 'REFUGEVal' or DF[0] == 'REFUGETest':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        else:
            DF_all = pd.read_csv(data_path + '/' + 'GlaucomaLabels_6Center_191206_cleaned.csv', encoding='gbk')
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = DF_all.loc[DF_all['center'] == split]
                DF_this = DF_this.reset_index(drop=True)
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']
        data_path = self.data_path + '/' + 'Images'
        fullPathName = os.path.join(data_path, imgName)
        fullPathName = fullPathName.replace('\\', '/')

        Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)

        discFlag = self.DF.loc[index, 'discFlag']
        if discFlag == 1:
            xmin = self.DF.loc[index, 'xmin']
            ymin = self.DF.loc[index, 'ymin']
            xmax = self.DF.loc[index, 'xmax']
            ymax = self.DF.loc[index, 'ymax']
            width = self.DF.loc[index, 'width']
            height = self.DF.loc[index, 'height']

            discHeight = ymax - ymin
            discWidth = xmax - xmin
            discX = int((xmax + xmin) / 2)
            discY = int((ymax + ymin) / 2)

            cropRadius1 = 1.5 * np.maximum(discHeight, discWidth)
            cropRadius2 = 0.2 * (np.maximum(height, width))
            # cropRadius = int(np.maximum(cropRadius1, cropRadius2))
            cropRadius = int(cropRadius1)
            # print(cropRadius, cropRadius1, cropRadius2)

            borderWidth = int(1.3 * cropRadius)
            ImgPad = cv2.copyMakeBorder(Img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT,value=0)

            disturbance = np.random.randint(-int(0.15 * cropRadius), int(0.15 * cropRadius), 4)
            xmin_crop = discX + borderWidth - cropRadius + disturbance[0]
            ymin_crop = discY + borderWidth - cropRadius + disturbance[1]
            xmax_crop = discX + borderWidth + cropRadius + disturbance[2]
            ymax_crop = discY + borderWidth + cropRadius + disturbance[3]

            DiscCrop = ImgPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]

        else:

            DiscCrop = Img

        DiscCrop = transforms.ToPILImage()(DiscCrop)
        if self.transform is not None:
            DiscCrop = self.transform(DiscCrop)

        label = self.DF.loc[index, 'label']

        data_path = self.data_path
        maskName = self.DF.loc[index, 'maskName']
        fullPathName = os.path.join(data_path, maskName)
        fullPathName = fullPathName.replace('\\', '/')

        Img = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)

        discFlag = self.DF.loc[index, 'discFlag']
        if discFlag == 1:

            ImgPad = cv2.copyMakeBorder(Img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT,
                                        value=0)
            Seg_DiscCrop = ImgPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

        else:

            Seg_DiscCrop = Img

        Seg_DiscCrop = transforms.ToPILImage()(Seg_DiscCrop)
        if self.transform_seg is not None:
            Seg_DiscCrop = self.transform_seg(Seg_DiscCrop)

        return DiscCrop, Seg_DiscCrop, label, imgName


    def __len__(self):
        return len(self.DF)

class Dataset_FullImg(Dataset):
    def __init__(self, data_path, DF, transform = None, transform_seg = None):
        if DF[0] == 'train' or DF[0] == 'val' or DF[0] == 'imp' or DF[0] == 'ORIGAtrain' or DF[0] == 'ORIGAimp' or DF[0] == 'ORIGAtest':
            self.DF = pd.read_csv(data_path + '/' + 'Glaucoma' + '_' + DF[0] + '.csv', encoding='gbk')
        elif DF[0] == 'LAG_train' or DF[0] == 'LAG_test':
            self.DF = pd.read_csv(data_path + '/' + DF[0] + '.csv',encoding='gbk')
        elif DF[0] == 'REFUGETrain' or DF[0] == 'REFUGEVal' or DF[0] == 'REFUGETest':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        elif DF[0] == 'RIM-ONEv3_SIL' or DF[0] == 'RIM-ONEv3_SIR':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        elif DF[0] == 'DRIGHTI_train' or DF[0] == 'DRIGHTI_test':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        elif DF[0] == 'BinRushed' or DF[0] == 'Magrabia' or DF[0] == 'MESSIDOR':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + 'Glaucoma_seg_' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        else:
            DF_all = pd.read_csv(data_path + '/' + 'GlaucomaLabels_6Center_191206_cleaned.csv', encoding='gbk')
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = DF_all.loc[DF_all['center'] == split]
                DF_this = DF_this.reset_index(drop=True)
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']
        data_path = self.data_path + '/' + 'Images'
        fullPathName = os.path.join(data_path, imgName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Img = Image.open(fullPathName).convert('RGB')
        if self.transform is not None:
            Img = self.transform(Img)

        """Get the segmentation images"""
        data_path = self.data_path
        maskName = self.DF.loc[index, 'maskName']
        fullPathName = os.path.join(data_path, maskName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Seg = Image.open(fullPathName).convert('L')
        if self.transform_seg is not None:
            Seg = self.transform_seg(Seg)

        label = self.DF.loc[index, 'label']

        return Img, Seg, label, imgName


    def __len__(self):
        return len(self.DF)

class OracleDataset(Dataset):
    def __init__(self, args, data_path , transform = None, transform_seg = None, mode = 'Train',plane = False):
        df = pd.read_csv(os.path.join(data_path, 'REFUGE','REFUGE1' + mode + '.csv'), encoding='gbk')
        self.name_list = df['imgName'].tolist()
        self.mask_list = df['maskName'].tolist()
        self.mmask_list = df['multimaskName'].tolist()
        self.label_list = df['label'].tolist()
        self.data_path = data_path

        self.transform_seg = transform_seg
        self.transform = transform

        if plane:
            self.name_list = [a.split('_')[0] + '.jpg' for a in self.mmask_list]

    @classmethod
    def to2(cls,mask):
        img_nd = np.array(mask)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        disc = img_nd.copy()
        disc[disc != 0] = 1
        cup = img_nd.copy()
        cup[cup != 1] = 0

        disc = disc[:,:,0]*255
        cup = cup[:,:,0]*255

        img_nd = np.dstack((disc,cup))

        # return Image.fromarray(disc.astype(np.uint8)),Image.fromarray(cup.astype(np.uint8))
        return img_nd
    
    @classmethod
    def allone(cls, disc,cup):
        disc = np.array(disc) / 255
        cup = np.array(cup) / 255
        return  np.clip(disc * 0.5 + cup,0,1)
    
    @classmethod
    def reversecolor(cls,seg):
        seg = 255 - np.array(seg)
        return seg

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        msk_path = os.path.join(self.data_path, self.mask_list[index])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        # ave_mask = self.reversecolor(self.to2(ave_mask))
        mask = self.to2(mask)
        
        label = int(self.label_list[index])

        multiname = self.mmask_list[index].split('.')[0].split('_')[0]

        masks = []
        ones = []
        data_path = self.data_path
        for n in range(1,8):     # n:1-7
            cup_path = os.path.join(data_path, multiname + '_seg_cup_' + str(n) + '.png')
            disc_path = os.path.join(data_path, multiname + '_seg_disc_' + str(n) + '.png')

            cup = Image.open(cup_path).convert('L')
            disc = Image.open(disc_path).convert('L')

            one =  self.allone(disc, cup)

            if self.transform_seg:
                disc = self.transform_seg(disc)
                cup = self.transform_seg(cup)
                one = self.transform_seg(one)
            
            Mask = torch.cat((disc,cup),0)

            masks.append(Mask)
            ones.append(one)

        if self.transform:
            img = self.transform(img)

        if self.transform_seg:
            mask = self.transform_seg(mask)

        return img, mask, ones, masks, label, name.split('/')[-1]

    
    
