import numpy as np
import os
import nibabel as nib
import torch
ori_train_img_list = []
ori_train_mask_list = []
ori_test_img_list = []
ori_test_mask_list = []
for root, dirs, files in os.walk("../ISBI2015"):
    try:
        root.split('/')[2]
    except:
        continue
    if root.split('/')[2] == "training0" + str(5):  #
        for dir in dirs:#each timepoint
            flair_path = os.path.join(root, dir,root.split('/')[2]+"_"+dir+"_flair_pp.nii")
            mprage_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mprage_pp.nii")
            t2_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_t2_pp.nii")
            pd_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_pd_pp.nii")
            mask_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mask1.nii")

            flair = np.array(nib.load(flair_path).get_fdata())
            mprage_pp = np.array(nib.load(mprage_pp_path).get_fdata())
            t2_pp = np.array(nib.load(t2_pp_path).get_fdata())
            pd_pp = np.array(nib.load(pd_pp_path).get_fdata())
            mask = np.array(nib.load(mask_path).get_fdata())
            flair = flair / np.max(flair)
            mprage_pp = mprage_pp / np.max(mprage_pp)
            t2_pp = t2_pp / np.max(t2_pp)
            pd_pp = pd_pp / np.max(pd_pp)

            ori_test_img_list.append(flair)
            ori_test_img_list.append(mprage_pp)
            ori_test_img_list.append(t2_pp)
            ori_test_img_list.append(pd_pp)

            ori_test_img_list.append(flair.transpose(1, 2, 0))
            ori_test_img_list.append(mprage_pp.transpose(1, 2, 0))
            ori_test_img_list.append(t2_pp.transpose(1, 2, 0))
            ori_test_img_list.append(pd_pp.transpose(1, 2, 0))

            ori_test_img_list.append(flair.transpose(2, 0, 1))
            ori_test_img_list.append(mprage_pp.transpose(2, 0, 1))
            ori_test_img_list.append(t2_pp.transpose(2, 0, 1))
            ori_test_img_list.append(pd_pp.transpose(2, 0, 1))

            ori_test_mask_list.append(mask)
            ori_test_mask_list.append(mask.transpose(1, 2, 0))
            ori_test_mask_list.append(mask.transpose(2, 0, 1))
    else:
        if root.split('/')[2][0:9] == "training0":
            for dir in dirs:
                flair_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_flair_pp.nii")
                mprage_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mprage_pp.nii")
                t2_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_t2_pp.nii")
                pd_pp_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_pd_pp.nii")
                mask_path = os.path.join(root, dir, root.split('/')[2] + "_" + dir + "_mask1.nii")

                flair = np.array(nib.load(flair_path).get_fdata())
                mprage_pp = np.array(nib.load(mprage_pp_path).get_fdata())
                t2_pp = np.array(nib.load(t2_pp_path).get_fdata())
                pd_pp = np.array(nib.load(pd_pp_path).get_fdata())
                mask = np.array(nib.load(mask_path).get_fdata())
                flair = flair / np.max(flair)
                mprage_pp = mprage_pp / np.max(mprage_pp)
                t2_pp = t2_pp / np.max(t2_pp)
                pd_pp = pd_pp / np.max(pd_pp)

                ori_train_img_list.append(flair)
                ori_train_img_list.append(mprage_pp)
                ori_train_img_list.append(t2_pp)
                ori_train_img_list.append(pd_pp)

                ori_train_img_list.append(flair.transpose(1, 2, 0))
                ori_train_img_list.append(mprage_pp.transpose(1, 2, 0))
                ori_train_img_list.append(t2_pp.transpose(1, 2, 0))
                ori_train_img_list.append(pd_pp.transpose(1, 2, 0))

                ori_train_img_list.append(flair.transpose(2, 0, 1))
                ori_train_img_list.append(mprage_pp.transpose(2, 0, 1))
                ori_train_img_list.append(t2_pp.transpose(2, 0, 1))
                ori_train_img_list.append(pd_pp.transpose(2, 0, 1))

                ori_train_mask_list.append(mask)
                ori_train_mask_list.append(mask.transpose(1, 2, 0))
                ori_train_mask_list.append(mask.transpose(2, 0, 1))

def load_data_aug_4(path,i):
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
        try:
            root.split('/')[2]
        except:
            continue
        if root.split('/')[2] == "training0" + str(5):  #
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

    print(len(ori_train_img_list))
    print(len(ori_train_mask_list))
    '''
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
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, ((0,0), pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, ((0,0), pad0, pad1, pad2), 'constant')
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
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        train_index.append(index)
        index += 1

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
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, ((0,0),pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, ((0,0),pad0, pad1, pad2), 'constant')
        depth = mask.shape[1]
        width = mask.shape[2]
        height = mask.shape[3]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[:,d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[:,d:d + 64, w:w + 64, h:h + 64]
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
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__=="__main__":
    load_data_aug_4("../ISBI2015/",5)





