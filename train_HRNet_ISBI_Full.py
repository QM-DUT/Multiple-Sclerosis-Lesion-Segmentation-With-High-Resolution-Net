from torch.utils.data import Dataset
import cv2
from utils.util import *
from Model.HRNet_3D import *
import config
import nibabel as nib
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

def eval_net(net, dataset, mask_list):
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
        mask_pred = net(img)
        mask_blank_list[mapping[index.item()]][0][0:1, d:d + 64, w:w + 64, h:h + 64] += mask_pred.detach().squeeze(
            0).cpu().numpy()
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
        real_re_mask=(temp0+temp1+temp2)/3
        real_re_mask_list.append(real_re_mask)
        real_mask_list.append(mask_list[i][0])
        #if vote
        tt=np.zeros(temp0.shape)
        for d in range(tt.shape[1]):
            for w in range(tt.shape[2]):
                for h in range(tt.shape[3]):
                    if (temp0[0][d][w][h]>0.5 and temp1[0][d][w][h]>0.5) or (temp0[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5)\
                        or (temp1[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5):
                        tt[0][d][w][h]=1
        real_re_mask_list2.append(tt)
    # return dice_coeff_cpu(reconstucted_mask, original_mask)
    return dice_coeff_cpu(real_re_mask_list, real_mask_list),dice_coeff_cpu(real_re_mask_list2,real_mask_list)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HRNet')
    parser.add_argument('-r', '--root', type=str, default='../ISBI2015/', help='data_root')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('-s', '--seed', type=str, default=20, help="seed string")  # 随机种子不要一样
    parser.add_argument('--log_interval', type=str, default=1, help="log_interval")
    parser.add_argument('--resume', type=bool, default=False, help="if load model")
    parser.add_argument('--save_dir', type=str, default="./results_HRNet_ISBI_Full", help="save directory")
    opt = parser.parse_args()
    print(opt)
    if not os.path.isdir(opt.save_dir):
        os.mkdir(opt.save_dir)
    setup_seed(opt.seed)
    k_flod=5
    for k in range(k_flod):
        print("The "+str(k+1)+" flod")
        ori_train_img_mask_list, ori_test_img_mask_list =load_data_full_4(opt.root,k)
        print("successfully loaded data")

        train_dataset = CustomDataset_full(ori_train_img_mask_list, transforms=transforms.Compose([ToTensor_full()]))
        val_dataset = CustomDataset_full(ori_test_img_mask_list, transforms=transforms.Compose([ToTensor_full()]))
        print("train data number: ",len(train_dataset))
        print("test data number: ",len(val_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        model_save_path = os.path.join(opt.save_dir,str(k+1)+"_flod")  # directory to same the model after each epoch.

        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        # net=  get_pose_net(config.cfg, 1)
        net = torch.nn.DataParallel(get_pose_net(config.cfg, 1,4), device_ids=[0, 1])
        net.to(device)
        # optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr,momentum=0.9,weight_decay=1e-6)
        if opt.resume:
            save_name = os.path.join(model_save_path, 'model_best.pth')
            load_checkpoint(save_name, net, optimizer)
        criterion = nn.BCELoss()
        N_train = len(train_dataset)
        max_dice = 0
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters in network: ', n_params)

        # val_dice = eval_net_full(net, val_loader, device)
        # print('Validation Dice Coeff: {}'.format(val_dice))

        for epoch in range(opt.epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, opt.epochs))
            net.train()
            epoch_loss = 0
            for i, [b,index] in enumerate(train_loader):
                imgs = b['img'].to(device)
                true_masks = b['label'].to(device)

                masks_pred = net(imgs).squeeze(0)
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()
                if i%20==0:
                    dice_score = (2 * torch.dot(masks_probs_flat, true_masks_flat) + 1) / (
                                torch.sum(masks_probs_flat) + torch.sum(true_masks_flat) + 1)
                    print('{0:.4f} --- loss: {1:.6f}, dice_score:{2:.6f}'.
                          format(i * opt.batch_size / N_train, loss.item(),dice_score))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch finished ! Loss: {}'.format(epoch_loss / (len(train_loader)*opt.batch_size)))

            val_dice = eval_net_full(net, train_loader, device)
            print('Validation Dice Coeff: {}'.format(val_dice))
            val_dice = eval_net_full(net, val_loader,device)
            print('Validation Dice Coeff: {}'.format(val_dice))
            if np.mean(val_dice) > max_dice:
                max_dice = np.mean(val_dice)
                print('New best performance! saving')
                save_name = os.path.join(model_save_path, 'model_best.pth')
                save_checkpoint(save_name, net, optimizer)
                print('model saved to {}'.format(save_name))
            if (epoch + 1) % opt.log_interval == 0:
                save_name = os.path.join(model_save_path, 'model_routine.pth')
                save_checkpoint(save_name, net, optimizer)
                print('model saved to {}'.format(save_name))
