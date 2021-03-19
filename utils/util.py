import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from optparse import OptionParser
import numpy as np
from torch import optim
from PIL import Image
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
import glob
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
import cv2
import  time
import nibabel as nib

class Flip(object):
    """
    Flip the image left or right for data augmentation, but prefer original image.
    """

    def __init__(self, ori_probability=0.60):
        self.ori_probability = ori_probability

    def __call__(self, sample):
        if random.uniform(0, 1) < self.ori_probability:
            return sample
        else:
            img, label = sample['img'], sample['label']
            img_flip = img[:, :, ::-1]
            label_flip = label[:, ::-1]

            return {'img': img_flip, 'label': label_flip}

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label ,index, d, w, h= sample['img'], sample['label'], sample['index'], sample['d'], sample['w'], sample['h']

        return {'img': torch.from_numpy(image.copy()).type(torch.FloatTensor),
                'label': torch.from_numpy(label.copy()).type(torch.FloatTensor),
                'index':index,
                'd':d,
                'w':w,
                'h':h}

# the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_masks, transforms=None):
        self.image_masks = image_masks
        self.transforms = transforms

    def __len__(self):  # return count of sample we have
        return len(self.image_masks)

    def __getitem__(self, index):
        image = self.image_masks[index][0]  # H, W, C
        mask = self.image_masks[index][1]
        ii = self.image_masks[index][2]
        d = self.image_masks[index][3]
        w = self.image_masks[index][4]
        h = self.image_masks[index][5]
        #image = np.transpose(image, axes=[2, 0, 1])  # C, H, W
        sample = {'img': image, 'label': mask, 'index': ii, 'd': d, 'w': w, 'h': h}
        if transforms:
            sample = self.transforms(sample)
        return sample

class ToTensor_full(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label= sample['img'], sample['label']

        return {'img': torch.from_numpy(image.copy()).type(torch.FloatTensor),
                'label': torch.from_numpy(label.copy()).type(torch.FloatTensor)}

# the dataset class
class CustomDataset_full(Dataset):
    def __init__(self, image_masks, transforms=None):
        self.image_masks = image_masks
        self.transforms = transforms

    def __len__(self):  # return count of sample we have
        return len(self.image_masks)

    def __getitem__(self, index):
        image = self.image_masks[index][0]  # H, W, C
        mask = self.image_masks[index][1]
        #image = np.transpose(image, axes=[2, 0, 1])  # C, H, W
        sample = {'img': image, 'label': mask}
        if transforms:
            sample = self.transforms(sample)
        return sample ,index

def save_checkpoint(save_name, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_name)

def load_checkpoint(save_name, model, optimizer):
    if model is None:
        pass
    else:
        model_CKPT = torch.load(save_name)
        model.load_state_dict(model_CKPT['state_dict'])
        if optimizer is None:
            pass
        else:
            optimizer.load_state_dict(model_CKPT['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print('loading checkpoint!')
    return model, optimizer

def dice_coeff_cpu(prediction, target):
    s = []
    eps = 1.0
    for i, (a, b) in enumerate(zip(prediction, target)):
        A = a.flatten()
        B = b.flatten()
        inter = np.dot(A, B)
        union = np.sum(A) + np.sum(B) + eps
        # Calculate DICE
        d = (2 * inter + eps) / union
        s.append(d)
    return s

def eval_net_batch(net, dataset, mask_list, device):
    # set net mode to evaluation
    net.eval()

    tt1=time.time()
    with torch.no_grad():
        reconstucted_mask = []
        original_mask = []
        mask_blank_list = []
        mapping = {}
        real_re_mask_list=[]
        real_re_mask_list2 = []
        real_mask_list=[]
        for i in range(len(mask_list)):
            mask = mask_list[i][0]
            mapping[mask_list[i][1]] = i
            mask_blank_list.append((np.zeros((mask.shape)), np.zeros((mask.shape))))
            original_mask.append(mask)
        # 这里需要考虑batch_size

        for i, b in enumerate(dataset):
            t1 = time.time()
            batch_img = b['img'].to(device)#b*c*64*64*64
            batch_index = b['index']#b
            batch_d = b['d']
            batch_w = b['w']
            batch_h = b['h']
            # Feed the image to network to get predicted mask
            t3=time.time()
            batch_mask_pred = net(batch_img)#b*1*64*64*64
            t4=time.time()
            # print(t4-t3)
            for ii in range(batch_mask_pred.shape[0]):
                index = batch_index[ii]
                mask_pred = batch_mask_pred[ii]
                d = batch_d[ii]
                w = batch_w[ii]
                h = batch_h[ii]
                mask_blank_list[mapping[index.item()]][0][0:1, d:d + 64, w:w + 64, h:h + 64] += mask_pred.detach().squeeze(0).detach().cpu().numpy()
                mask_blank_list[mapping[index.item()]][1][0:1, d:d + 64, w:w + 64, h:h + 64] += 1
            t2 = time.time()
            # print(t2-t1)
        tt2=time.time()
        print(tt2-tt1)
        for i in range(len(mask_blank_list)):
            mask_blank_list[i][1][mask_blank_list[i][1] == 0] = 1
            reconstucted_mask.append(mask_blank_list[i][0] / mask_blank_list[i][1])
        #majority vote or just add for every 3 reconstructed_mask
        for i in range(0,len(reconstucted_mask),3):
            temp0=reconstucted_mask[i + 0]
            temp1=reconstucted_mask[i + 1].transpose(0, 3, 1, 2)#2,0,1
            temp2=reconstucted_mask[i + 2].transpose(0, 2, 3, 1)#1,2,0
            #if add
            # real_re_mask=(temp0+temp1+temp2)/3
            # real_re_mask_list.append(real_re_mask)
            # real_mask_list.append(mask_list[i][0])
            #if vote
            temp0[temp0 >= 0.5] = 1
            temp0[temp0 < 0.5] = 0
            temp1[temp1 >= 0.5] = 1
            temp1[temp1 < 0.5] = 0
            temp2[temp2 >= 0.5] = 1
            temp2[temp2 < 0.5] = 0
            tt = temp0 + temp1 + temp2
            tt[tt <= 1.1] = 0
            tt[tt >= 1.9] = 1
            # tt=np.zeros(temp0.shape)
            # for d in range(tt.shape[1]):
            #     for w in range(tt.shape[2]):
            #         for h in range(tt.shape[3]):
            #             if (temp0[0][d][w][h]>0.5 and temp1[0][d][w][h]>0.5) or (temp0[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5)\
            #                 or (temp1[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5):
            #                 tt[0][d][w][h]=1
            real_mask_list.append(mask_list[i][0])
            real_re_mask_list2.append(tt)
    # return dice_coeff_cpu(reconstucted_mask, original_mask)
    return dice_coeff_cpu(real_re_mask_list2, real_mask_list)
    # return dice_coeff_cpu(real_re_mask_list, real_mask_list),dice_coeff_cpu(real_re_mask_list2,real_mask_list)

def eval_net(net, dataset, mask_list, device):
    # set net mode to evaluation
    net.eval()
    reconstucted_mask = []
    original_mask = []
    mask_blank_list = []
    mapping = {}
    real_re_mask_list=[]
    real_re_mask_list2 = []
    real_mask_list=[]
    for i in range(len(mask_list)):
        mask = mask_list[i][0]
        mapping[mask_list[i][1]] = i
        mask_blank_list.append((np.zeros((mask.shape)), np.zeros((mask.shape))))
        original_mask.append(mask)
    # 这里需要考虑batch_size,不过根本不会太大
    for i, b in enumerate(dataset):
        img = b['img'].to(device)
        index = b['index']
        d = b['d']
        w = b['w']
        h = b['h']
        # Feed the image to network to get predicted mask
        with torch.no_grad():#do not track gradient
            mask_pred = net(img)
        mask_blank_list[mapping[index.item()]][0][0:1, d:d + 64, w:w + 64, h:h + 64] += mask_pred.detach().squeeze(0).cpu().numpy()
        mask_blank_list[mapping[index.item()]][1][0:1, d:d + 64, w:w + 64, h:h + 64] += 1

    for i in range(len(mask_blank_list)):
        mask_blank_list[i][1][mask_blank_list[i][1] == 0] = 1
        reconstucted_mask.append(mask_blank_list[i][0] / mask_blank_list[i][1])

    #majority vote or just add for every 3 reconstructed_mask
    for i in range(0,len(reconstucted_mask),3):
        temp0=reconstucted_mask[i + 0]
        temp1=reconstucted_mask[i + 1].transpose(0, 3, 1, 2)#2,0,1
        temp2=reconstucted_mask[i + 2].transpose(0, 2, 3, 1)#1,2,0
        #if add
        # real_re_mask=(temp0+temp1+temp2)/3
        # real_re_mask_list.append(real_re_mask)

        #if vote
        temp0[temp0>=0.5] = 1
        temp0[temp0< 0.5] = 0
        temp1[temp1 >= 0.5] = 1
        temp1[temp1 < 0.5] = 0
        temp2[temp2 >= 0.5] = 1
        temp2[temp2 < 0.5] = 0
        tt=temp0+temp1+temp2

        tt[tt <= 1.1] = 0
        tt[tt>=1.9]=1
        # tt=np.zeros(temp0.shape)
        # for d in range(tt.shape[1]):
        #     for w in range(tt.shape[2]):
        #         for h in range(tt.shape[3]):
        #             if (temp0[0][d][w][h]>0.5 and temp1[0][d][w][h]>0.5) or (temp0[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5)\
        #                 or (temp1[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5):
        #                 tt[0][d][w][h]=1
        real_mask_list.append(mask_list[i][0])
        real_re_mask_list2.append(tt)
    # return dice_coeff_cpu(reconstucted_mask, original_mask)
    return dice_coeff_cpu(real_re_mask_list2, real_mask_list)
    # return dice_coeff_cpu(real_re_mask_list, real_mask_list),dice_coeff_cpu(real_re_mask_list2,real_mask_list)

def eval_net_full(net, dataset, device):
    # set net mode to evaluation
    net.eval()
    mask_pred_list=[]
    mask_ori_list=[]
    real_mask_list=[]
    real_pred_list2=[]
    indexs=[]
    # 这里batch_size=1
    for i, [b,index] in enumerate(dataset):
        img = b['img'].to(device)
        true_mask = b['label'].squeeze(0).numpy()
        print(img.shape)
        # Feed the image to network to get predicted mask
        with torch.no_grad():#do not track gradient
            mask_pred = net(img).squeeze(0).detach().cpu().numpy()
        mask_pred_list.append(mask_pred)
        mask_ori_list.append(true_mask)
        indexs.append(index)
    ii=sorted(range(len(indexs)), key=lambda k: indexs[k])
    #majority vote or just add for every 3 reconstructed_mask
    for i in range(0,len(mask_pred_list),3):
        temp0=mask_pred_list[ii[i + 0]]
        temp1=mask_pred_list[ii[i + 1]].transpose(0, 3, 1, 2)#2,0,1
        temp2=mask_pred_list[ii[i + 2]].transpose(0, 2, 3, 1)#1,2,0
        #if vote
        temp0[temp0 >= 0.5] = 1
        temp0[temp0 < 0.5] = 0
        temp1[temp1 >= 0.5] = 1
        temp1[temp1 < 0.5] = 0
        temp2[temp2 >= 0.5] = 1
        temp2[temp2 < 0.5] = 0
        tt=temp0+temp1+temp2
        tt[tt <= 1.1] = 0
        tt[tt>=1.9]=1

        real_mask_list.append(mask_ori_list[ii[i]])
        real_pred_list2.append(tt)
    # return dice_coeff_cpu(reconstucted_mask, original_mask)
    # return dice_coeff_cpu(real_pred_list2, real_mask_list)
    return dice_coeff_cpu(mask_pred_list, mask_ori_list)
    # return dice_coeff_cpu(real_re_mask_list, real_mask_list),dice_coeff_cpu(real_re_mask_list2,real_mask_list)

def eval_net_batch_three_view(net,net2,net3, dataset, mask_list, device):
    # set net mode to evaluation
    net.eval()
    net2.eval()
    net3.eval()
    tt1 = time.time()
    with torch.no_grad():
        reconstucted_mask = []
        original_mask = []
        mask_blank_list = []
        mapping = {}
        real_re_mask_list = []
        real_re_mask_list2 = []
        real_mask_list = []
        for i in range(len(mask_list)):
            mask = mask_list[i][0]
            mapping[mask_list[i][1]] = i
            mask_blank_list.append((np.zeros((mask.shape)), np.zeros((mask.shape))
                                    , np.zeros((mask.transpose(0,2, 3, 1).shape)), np.zeros((mask.transpose(0,2, 3, 1).shape))
                                    , np.zeros((mask.transpose(0,3,1,2).shape)), np.zeros((mask.transpose(0,3,1,2).shape))))
            # print(mask.shape,mask.transpose(0,2, 3, 1).shape,mask.transpose(0,3,1,2).shape)
            original_mask.append(mask)
        # 这里需要考虑batch_size

        for i, b in enumerate(dataset):
            t1 = time.time()
            batch_img = b['img'].to(device)  # b*c*64*64*64
            batch_img2 = b['img'].permute(0,1,3, 4, 2).to(device)
            batch_img3 = b['img'].permute(0,1,4, 2, 3).to(device)

            batch_index = b['index']  # b
            batch_d = b['d']
            batch_w = b['w']
            batch_h = b['h']

            batch_d2 = b['w']
            batch_w2 = b['h']
            batch_h2 = b['d']

            batch_d3 = b['h']
            batch_w3 = b['d']
            batch_h3 = b['w']

            # Feed the image to network to get predicted mask
            t3 = time.time()
            batch_mask_pred = net(batch_img)  # b*1*64*64*64
            batch_mask_pred2 = net2(batch_img2)  # b*1*64*64*64
            batch_mask_pred3 = net3(batch_img3)  # b*1*64*64*64
            t4 = time.time()
            # print(t4-t3)
            for ii in range(batch_mask_pred.shape[0]):
                index = batch_index[ii]
                mask_pred = batch_mask_pred[ii]
                d = batch_d[ii]
                w = batch_w[ii]
                h = batch_h[ii]
                mask_blank_list[mapping[index.item()]][0][0:1, d:d + 64, w:w + 64,
                h:h + 64] += mask_pred.detach().squeeze(0).detach().cpu().numpy()
                mask_blank_list[mapping[index.item()]][1][0:1, d:d + 64, w:w + 64, h:h + 64] += 1

                mask_pred2 = batch_mask_pred2[ii]
                d2 = batch_d2[ii]
                w2 = batch_w2[ii]
                h2 = batch_h2[ii]
                mask_blank_list[mapping[index.item()]][2][0:1, d2:d2 + 64, w2:w2 + 64,
                h2:h2 + 64] += mask_pred2.detach().squeeze(0).detach().cpu().numpy()
                mask_blank_list[mapping[index.item()]][3][0:1, d2:d2 + 64, w2:w2 + 64, h2:h2 + 64] += 1

                mask_pred3 = batch_mask_pred3[ii]
                d3 = batch_d3[ii]
                w3 = batch_w3[ii]
                h3 = batch_h3[ii]
                mask_blank_list[mapping[index.item()]][4][0:1, d3:d3 + 64, w3:w3 + 64,
                h3:h3 + 64] += mask_pred3.detach().squeeze(0).detach().cpu().numpy()
                mask_blank_list[mapping[index.item()]][5][0:1, d3:d3 + 64, w3:w3 + 64, h3:h3 + 64] += 1


            t2 = time.time()
            # print(t2-t1)
        tt2 = time.time()
        print(tt2 - tt1)
        for i in range(len(mask_blank_list)):
            mask_blank_list[i][1][mask_blank_list[i][1] == 0] = 1
            reconstucted_mask.append(mask_blank_list[i][0] / mask_blank_list[i][1])

            mask_blank_list[i][3][mask_blank_list[i][3] == 0] = 1
            reconstucted_mask.append(mask_blank_list[i][2] / mask_blank_list[i][3])

            mask_blank_list[i][5][mask_blank_list[i][5] == 0] = 1
            reconstucted_mask.append(mask_blank_list[i][4] / mask_blank_list[i][5])
        # majority vote or just add for every 3 reconstructed_mask
        for i in range(0, len(reconstucted_mask), 3):
            temp0 = reconstucted_mask[i + 0]
            temp1 = reconstucted_mask[i + 1].transpose(0, 3, 1, 2)  # 2,0,1
            temp2 = reconstucted_mask[i + 2].transpose(0, 2, 3, 1)  # 1,2,0
            # if add
            # real_re_mask=(temp0+temp1+temp2)/3
            # real_re_mask_list.append(real_re_mask)
            # real_mask_list.append(mask_list[i][0])
            # if vote
            temp0[temp0 >= 0.5] = 1
            temp0[temp0 < 0.5] = 0
            temp1[temp1 >= 0.5] = 1
            temp1[temp1 < 0.5] = 0
            temp2[temp2 >= 0.5] = 1
            temp2[temp2 < 0.5] = 0
            tt = temp0 + temp1 + temp2
            tt[tt <= 1.1] = 0
            tt[tt >= 1.9] = 1
            # tt=np.zeros(temp0.shape)
            # for d in range(tt.shape[1]):
            #     for w in range(tt.shape[2]):
            #         for h in range(tt.shape[3]):
            #             if (temp0[0][d][w][h]>0.5 and temp1[0][d][w][h]>0.5) or (temp0[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5)\
            #                 or (temp1[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5):
            #                 tt[0][d][w][h]=1
            real_mask_list.append(mask_list[i][0])
            real_re_mask_list2.append(tt)
    # return dice_coeff_cpu(reconstucted_mask, original_mask)
    return dice_coeff_cpu(real_re_mask_list2, real_mask_list)
    # return dice_coeff_cpu(real_re_mask_list, real_mask_list),dice_coeff_cpu(real_re_mask_list2,real_mask_list)

def load_data(path,i):
    # get all the image and mask path and number of images
    train_index = []
    val_index = []
    mask_list = []
    train_img_masks = []
    val_img_masks = []
    voxel = 1.0 #64 * 64 * 64 * 0.0001
    # 遍历根目录
    index = 0
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []
    for root, dirs, files in os.walk(path):
        if dirs == []:
            for name in files:
                path = os.path.join(root, name)
                if path.split('/')[2] == "training0"+str(i+1):  # test,5折交叉验证
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_test_img_list.append(img)
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_test_mask_list.append(mask)
                else:  # train
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_train_img_list.append(img)
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_train_mask_list.append(mask)

    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    if np.sum(patch_mask) >=voxel:
                        patch_img = np.expand_dims(patch_img, 0)
                        patch_mask = np.expand_dims(patch_mask, 0)
                        train_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        train_index.append(index)
        index += 1

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    patch_img = np.expand_dims(patch_img, 0)
                    patch_mask = np.expand_dims(patch_mask, 0)
                    val_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        val_index.append(index)
        index += 1

    real_train_mask_list = []
    real_test_mask_list = []
    for mask, index in mask_list:
        if index in train_index:
            real_train_mask_list.append((mask, index))
        if index in val_index:
            real_test_mask_list.append((mask, index))
    return train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list

def load_data2(path,i):
    # get all the image and mask path and number of images
    train_index = []
    val_index = []
    mask_list = []
    train_img_masks = []
    val_img_masks = []
    voxel = 1.0 #64 * 64 * 64 * 0.0001
    # 遍历根目录
    index = 0
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []
    for root, dirs, files in os.walk(path):
        if dirs == []:
            for name in files:
                path = os.path.join(root, name)
                if path.split('/')[3] == "0"+str(i+1):
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_test_img_list.append(img,img.transpose(1,2,0),img.transpose(2,0,1))
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_test_mask_list.append(mask,mask.transpose(1,2,0),mask.transpose(2,0,1))
                else:  # train
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        print(path)
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_train_img_list.append(img,img.transpose(1,2,0),img.transpose(2,0,1))
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        print(path)
                        mask = np.array(nib.load(path).get_fdata())
                        ori_train_mask_list.append(mask,mask.transpose(1,2,0),mask.transpose(2,0,1))

    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    if np.sum(patch_mask) >=voxel:
                        patch_img = np.expand_dims(patch_img, 0)
                        patch_mask = np.expand_dims(patch_mask, 0)
                        train_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        train_index.append(index)
        index += 1

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    patch_img = np.expand_dims(patch_img, 0)
                    patch_mask = np.expand_dims(patch_mask, 0)
                    val_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        val_index.append(index)
        index += 1

    real_train_mask_list = []
    real_test_mask_list = []
    for mask, index in mask_list:
        if index in train_index:
            real_train_mask_list.append((mask, index))
        if index in val_index:
            real_test_mask_list.append((mask, index))
    return train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list

def load_data_aug(path,i):
    # get all the image and mask path and number of images
    train_index = []
    val_index = []
    mask_list = []
    train_img_masks = []
    val_img_masks = []
    voxel = 1 #64 * 64 * 64 * 0.0001
    # 遍历根目录
    index = 0
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []
    for root, dirs, files in os.walk(path):
        if dirs == []:
            for name in files:
                path = os.path.join(root, name)
                if path.split('/')[2] == "training0"+str(i+1):  # test,5折交叉验证
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_test_img_list.append(img)
                        ori_test_img_list.append(img.transpose(1, 2, 0))
                        ori_test_img_list.append(img.transpose(2, 0, 1))
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_test_mask_list.append(mask)
                        ori_test_mask_list.append(mask.transpose(1, 2, 0))
                        ori_test_mask_list.append(mask.transpose(2, 0, 1))
                else:  # train
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_train_img_list.append(img)
                        ori_train_img_list.append(img.transpose(1, 2, 0))
                        ori_train_img_list.append(img.transpose(2, 0, 1))
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_train_mask_list.append(mask)
                        ori_train_mask_list.append(mask.transpose(1, 2, 0))
                        ori_train_mask_list.append(mask.transpose(2, 0, 1))

    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    if np.sum(patch_mask) >=voxel:
                        patch_img = np.expand_dims(patch_img, 0)
                        patch_mask = np.expand_dims(patch_mask, 0)
                        train_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        train_index.append(index)
        index += 1

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    patch_img = np.expand_dims(patch_img, 0)
                    patch_mask = np.expand_dims(patch_mask, 0)
                    val_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        val_index.append(index)
        index += 1

    real_train_mask_list = []
    real_test_mask_list = []
    for mask, index in mask_list:
        if index in train_index:
            real_train_mask_list.append((mask, index))
        if index in val_index:
            real_test_mask_list.append((mask, index))
    return train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list

def load_data_aug_4(path,i):
    # get all the image and mask path and number of images
    train_index = []
    val_index = []
    mask_list = []
    train_img_masks = []
    val_img_masks = []
    voxel = 1 #64 * 64 * 64 * 0.0001
    # 遍历根目录
    index = 0
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []

    for root, dirs, files in os.walk(path):
        try:
            root.split('/')[2]
        except:
            continue
        if root.split('/')[2] == "training0" + str(i+1):  #
            for dir in dirs:  # each timepoint
                flair_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_flair_pp.nii")
                mprage_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mprage_pp.nii")
                t2_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_t2_pp.nii")
                pd_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_pd_pp.nii")
                mask_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mask1.nii")

                flair = np.expand_dims(np.array(nib.load(flair_path).get_fdata()), 0)
                mprage_pp = np.expand_dims(np.array(nib.load(mprage_pp_path).get_fdata()), 0)
                t2_pp = np.expand_dims(np.array(nib.load(t2_pp_path).get_fdata()), 0)
                pd_pp = np.expand_dims(np.array(nib.load(pd_pp_path).get_fdata()), 0)
                flair = flair / np.max(flair)
                mprage_pp = mprage_pp / np.max(mprage_pp)
                t2_pp = t2_pp / np.max(t2_pp)
                pd_pp = pd_pp / np.max(pd_pp)

                mask = np.expand_dims(np.array(nib.load(mask_path).get_fdata()), 0)

                multi_channel_img = np.vstack((flair, mprage_pp, t2_pp, pd_pp))
                ori_test_img_list.append(multi_channel_img)
                ori_test_img_list.append(multi_channel_img.transpose(0, 2, 3, 1))
                ori_test_img_list.append(multi_channel_img.transpose(0, 3, 1, 2))
                ori_test_mask_list.append(mask)
                ori_test_mask_list.append(mask.transpose(0, 2, 3, 1))
                ori_test_mask_list.append(mask.transpose(0, 3, 1, 2))
        else:
            if root.split('/')[2][0:9] == "training0":
                for dir in dirs:
                    print(root, dir)
                    flair_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_flair_pp.nii")
                    mprage_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mprage_pp.nii")
                    t2_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_t2_pp.nii")
                    pd_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_pd_pp.nii")
                    mask_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mask1.nii")

                    flair = np.expand_dims(np.array(nib.load(flair_path).get_fdata()), 0)
                    mprage_pp = np.expand_dims(np.array(nib.load(mprage_pp_path).get_fdata()), 0)
                    t2_pp = np.expand_dims(np.array(nib.load(t2_pp_path).get_fdata()), 0)
                    pd_pp = np.expand_dims(np.array(nib.load(pd_pp_path).get_fdata()), 0)
                    flair = flair / np.max(flair)
                    mprage_pp = mprage_pp / np.max(mprage_pp)
                    t2_pp = t2_pp / np.max(t2_pp)
                    pd_pp = pd_pp / np.max(pd_pp)

                    mask = np.expand_dims(np.array(nib.load(mask_path).get_fdata()), 0)

                    multi_channel_img = np.vstack((flair, mprage_pp, t2_pp, pd_pp))
                    ori_train_img_list.append(multi_channel_img)
                    ori_train_img_list.append(multi_channel_img.transpose(0, 2, 3, 1))
                    ori_train_img_list.append(multi_channel_img.transpose(0, 3, 1, 2))
                    ori_train_mask_list.append(mask)
                    ori_train_mask_list.append(mask.transpose(0, 2, 3, 1))
                    ori_train_mask_list.append(mask.transpose(0, 3, 1, 2))

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        # pad zero
        pad_size = [0, 0, 0, 0]
        div = [0, 64, 64, 64]
        pad = False
        for i in range(1,len(img.shape),1):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            pad3 = (pad_size[3] // 2, pad_size[3] - pad_size[3] // 2)
            img = np.pad(img, ((0,0), pad1, pad2,pad3), 'constant')
            mask = np.pad(mask, ((0,0), pad1, pad2,pad3), 'constant')
        depth = mask.shape[1]
        width = mask.shape[2]
        height = mask.shape[3]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[:,d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[:,d:d + 64, w:w + 64, h:h + 64]
                    val_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask_list.append((mask, index))
        val_index.append(index)
        index += 1

    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        # pad zero
        pad_size = [0, 0, 0, 0]
        div = [0, 64, 64, 64]
        pad = False
        for i in range(1,len(img.shape),1):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            pad3 = (pad_size[3] // 2, pad_size[3] - pad_size[3] // 2)
            img = np.pad(img, ((0,0), pad1, pad2, pad3), 'constant')
            mask = np.pad(mask, ((0,0), pad1, pad2, pad3), 'constant')
        depth = mask.shape[1]
        width = mask.shape[2]
        height = mask.shape[3]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[:,d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[:,d:d + 64, w:w + 64, h:h + 64]
                    if np.sum(patch_mask) >=voxel:
                        train_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask_list.append((mask, index))
        train_index.append(index)
        index += 1



    real_train_mask_list = []
    real_test_mask_list = []
    for mask, index in mask_list:
        if index in train_index:
            real_train_mask_list.append((mask, index))
        if index in val_index:
            real_test_mask_list.append((mask, index))
    return train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list

def load_data_full(path,i):
    # get all the image and mask path and number of images
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []

    ori_train_img_mask_list=[]
    ori_test_img_mask_list=[]
    for root, dirs, files in os.walk(path):
        if dirs == []:
            for name in files:
                path = os.path.join(root, name)
                if path.split('/')[2] == "training0" + str(i + 1):  # test, 5折交叉验证
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_test_img_list.append(img)
                        ori_test_img_list.append(img.transpose(1, 2, 0))
                        ori_test_img_list.append(img.transpose(2, 0, 1))
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_test_mask_list.append(mask)
                        ori_test_mask_list.append(mask.transpose(1, 2, 0))
                        ori_test_mask_list.append(mask.transpose(2, 0, 1))
                else:  # train
                    if name.split('_', 2)[2] == 'flair_pp.nii':
                        img = np.array(nib.load(path).get_fdata())
                        max_img = np.max(img)
                        img = img / max_img
                        ori_train_img_list.append(img)
                        ori_train_img_list.append(img.transpose(1, 2, 0))
                        ori_train_img_list.append(img.transpose(2, 0, 1))
                    elif name.split('_', 2)[2] == 'mask1.nii':
                        mask = np.array(nib.load(path).get_fdata())
                        ori_train_mask_list.append(mask)
                        ori_train_mask_list.append(mask.transpose(1, 2, 0))
                        ori_train_mask_list.append(mask.transpose(2, 0, 1))

    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        ori_train_img_mask_list.append([img,mask])

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        ori_test_img_mask_list.append([img,mask])

    return ori_train_img_mask_list, ori_test_img_mask_list

def load_data_full_4(path,i):#先crop再旋转
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []

    ori_train_img_mask_list=[]
    ori_test_img_mask_list=[]

    for root, dirs, files in os.walk(path):
        try:
            root.split('/')[2]
        except:
            continue
        if root.split('/')[2] == "training0" + str(i + 1):  #
            for dir in dirs:  # each timepoint
                flair_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_flair_pp.nii")
                mprage_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mprage_pp.nii")
                t2_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_t2_pp.nii")
                pd_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_pd_pp.nii")
                mask_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mask1.nii")

                flair = np.expand_dims(np.array(nib.load(flair_path).get_fdata()), 0)
                mprage_pp = np.expand_dims(np.array(nib.load(mprage_pp_path).get_fdata()), 0)
                t2_pp = np.expand_dims(np.array(nib.load(t2_pp_path).get_fdata()), 0)
                pd_pp = np.expand_dims(np.array(nib.load(pd_pp_path).get_fdata()), 0)
                flair = flair / np.max(flair)
                mprage_pp = mprage_pp / np.max(mprage_pp)
                t2_pp = t2_pp / np.max(t2_pp)
                pd_pp = pd_pp / np.max(pd_pp)

                mask = np.expand_dims(np.array(nib.load(mask_path).get_fdata()), 0)
                multi_channel_img = np.vstack((flair, mprage_pp, t2_pp, pd_pp))

                # multi_channel_img = multi_channel_img[:, 10:170, 12:204, 10:170]
                # mask = mask[:, 10:170, 12:204, 10:170]
                ori_test_img_list.append(multi_channel_img)
                ori_test_img_list.append(multi_channel_img.transpose(0, 2, 3, 1))
                ori_test_img_list.append(multi_channel_img.transpose(0, 3, 1, 2))
                ori_test_mask_list.append(mask)
                ori_test_mask_list.append(mask.transpose(0, 2, 3, 1))
                ori_test_mask_list.append(mask.transpose(0, 3, 1, 2))
        else:
            if root.split('/')[2][0:9] == "training0":
                for dir in dirs:
                    print(root, dir)
                    flair_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_flair_pp.nii")
                    mprage_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mprage_pp.nii")
                    t2_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_t2_pp.nii")
                    pd_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_pd_pp.nii")
                    mask_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mask1.nii")

                    flair = np.expand_dims(np.array(nib.load(flair_path).get_fdata()), 0)
                    mprage_pp = np.expand_dims(np.array(nib.load(mprage_pp_path).get_fdata()), 0)
                    t2_pp = np.expand_dims(np.array(nib.load(t2_pp_path).get_fdata()), 0)
                    pd_pp = np.expand_dims(np.array(nib.load(pd_pp_path).get_fdata()), 0)
                    flair = flair / np.max(flair)
                    mprage_pp = mprage_pp / np.max(mprage_pp)
                    t2_pp = t2_pp / np.max(t2_pp)
                    pd_pp = pd_pp / np.max(pd_pp)

                    mask = np.expand_dims(np.array(nib.load(mask_path).get_fdata()), 0)
                    multi_channel_img = np.vstack((flair, mprage_pp, t2_pp, pd_pp))

                    # multi_channel_img = multi_channel_img[:, 10:170, 12:204, 10:170]
                    # mask = mask[:, 10:170, 12:204, 10:170]
                    ori_train_img_list.append(multi_channel_img)
                    ori_train_img_list.append(multi_channel_img.transpose(0, 2, 3, 1))
                    ori_train_img_list.append(multi_channel_img.transpose(0, 3, 1, 2))
                    ori_train_mask_list.append(mask)
                    ori_train_mask_list.append(mask.transpose(0, 2, 3, 1))
                    ori_train_mask_list.append(mask.transpose(0, 3, 1, 2))

    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        # pad zero
        pad_size = [0, 0, 0, 0]
        div = [0, 64, 64, 64]
        pad = False
        for i in range(1, len(img.shape), 1):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            pad3 = (pad_size[3] // 2, pad_size[3] - pad_size[3] // 2)
            img = np.pad(img, ((0, 0), pad1, pad2, pad3), 'constant')
            mask = np.pad(mask, ((0, 0), pad1, pad2, pad3), 'constant')
        ori_train_img_mask_list.append([img,mask])

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        # pad zero
        pad_size = [0, 0, 0, 0]
        div = [0, 64, 64, 64]
        pad = False
        for i in range(1, len(img.shape), 1):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            pad3 = (pad_size[3] // 2, pad_size[3] - pad_size[3] // 2)
            img = np.pad(img, ((0, 0), pad1, pad2, pad3), 'constant')
            mask = np.pad(mask, ((0, 0), pad1, pad2, pad3), 'constant')
        ori_test_img_mask_list.append([img,mask])
    return ori_train_img_mask_list, ori_test_img_mask_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    load_data_aug_4("../../ISBI2015/",1)
