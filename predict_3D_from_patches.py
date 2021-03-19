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
from utils.util import *
from Model.HRNet_3D import  *
from Model.UNet_3D import *
import config
import nibabel as nib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def dice_coeff_cpu(prediction, target):
    s=[]
    eps = 1.0
    for i, (a, b) in enumerate(zip(prediction, target)):
        A = a.flatten()
        B = b.flatten()
        inter = np.dot(A, B)
        union = np.sum(A) + np.sum(B) + eps
        # Calculate DICE
        d = (2 * inter+eps) / union
        s.append(d)
    return s
'''
def eval_net(net, dataset, mask_list):
    # set net mode to evaluation
    net.eval()
    reconstucted_mask = []
    original_mask = []
    mask_blank_list = []
    mapping = {}
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
        ################################################ [TODO] ###################################################
        # Feed the image to network to get predicted mask
        mask_pred = net(img)
        mask_blank_list[mapping[index.item()]][0][0:1, d:d + 64, w:w + 64, h:h + 64] += mask_pred.detach().squeeze(
            0).cpu().numpy()
        mask_blank_list[mapping[index.item()]][1][0:1, d:d + 64, w:w + 64, h:h + 64] += 1

    for i in range(len(mask_blank_list)):
        mask_blank_list[i][1][mask_blank_list[i][1] == 0] = 1
        reconstucted_mask.append(mask_blank_list[i][0] / mask_blank_list[i][1])
    return dice_coeff_cpu(reconstucted_mask, original_mask)
def load_data(path,i):
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
                    if np.sum(patch_mask) > voxel:
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
'''
if __name__=="__main__":
    train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list = load_data_aug('../ISBI2015/', 2)
    print("successfully loaded data")
    train_dataset = CustomDataset(train_img_masks, transforms=transforms.Compose([ToTensor()]))
    val_dataset = CustomDataset(val_img_masks, transforms=transforms.Compose([ToTensor()]))
    #net=get_pose_net(config.cfg, 1)
    net = torch.nn.DataParallel(get_pose_net(config.cfg, 1,1), device_ids=[0, 1])
    # net=UNet3D(1,1)
    net.to(device)

    # net.load_state_dict(torch.load('results_HRNet_ISBI_march/1_flod/model_best.pth',map_location='cuda:1'))
    # net.load_state_dict(torch.load('results_HRNet_ISBI_march/1_flod/model_best.pth'))
    load_checkpoint('results_HRNet_ISBI_march/3_flod/model_best.pth',net,None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_dice = eval_net(net, train_loader, real_train_mask_list,device)
    print('Validation Dice Coeff: {}'.format(val_dice))

    val_dice = eval_net(net, val_loader, real_test_mask_list,device)
    print('Validation Dice Coeff: {}'.format(val_dice))