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
        image, label ,index,d,w,h= sample['img'], sample['label'],sample['index'],sample['d'],sample['w'],sample['h']
        new_img_tensor=[]
        for i in range(len(image)):
            new_img_tensor.append(torch.from_numpy(image[i].copy()).type(torch.FloatTensor))


        return {'img':  new_img_tensor,
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
        image = self.image_masks[index][0]
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
def save_checkpoint(save_name, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_name + '.pth')
    print('model saved to {}'.format(save_name))


def load_checkpoint(save_name, model, optimizer):
    if model is None:
        pass
    else:
        model_CKPT = torch.load(save_name + '.pth')
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
