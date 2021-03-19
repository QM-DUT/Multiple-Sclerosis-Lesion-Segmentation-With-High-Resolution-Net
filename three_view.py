from torch.utils.data import Dataset
import cv2
from utils.util import *
from Model.HRNet_3D import *
import config

import argparse
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HRNet')
    parser.add_argument('-r', '--root', type=str, default='../ISBI2015/', help='data_root')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-s', '--seed', type=str, default=20, help="seed string")  # 随机种子不要一样
    parser.add_argument('--log_interval', type=str, default=1, help="log_interval")
    parser.add_argument('--resume', type=bool, default=False, help="if load model")
    parser.add_argument('--save_dir', type=str, default="./results_HRNet_ISBI_march_three_view", help="save directory")
    opt = parser.parse_args()
    print(opt)
    if not os.path.isdir(opt.save_dir):
        os.mkdir(opt.save_dir)
    setup_seed(opt.seed)
    k_flod=5
    for k in range(k_flod):
    # for k in [4,0]:
        print("The "+str(k+1)+" flod")
        train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list=load_data(opt.root,k)
        print("successfully loaded data")
        train_dataset = CustomDataset(train_img_masks, transforms=transforms.Compose([ToTensor()]))
        val_dataset = CustomDataset(val_img_masks, transforms=transforms.Compose([ToTensor()]))
        print("train data number: ",len(train_dataset))
        print("test data number: ",len(val_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        model_save_path = os.path.join(opt.save_dir,str(k+1)+"_flod")  # directory to same the model after each epoch.

        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        # net=  get_pose_net(config.cfg, 1)
        #create three different HRNet
        net = torch.nn.DataParallel(get_pose_net(config.cfg,1, 1), device_ids=[0, 1])
        net.to(device)

        net2 = torch.nn.DataParallel(get_pose_net(config.cfg, 1, 1), device_ids=[0, 1])
        net2.to(device)

        net3 = torch.nn.DataParallel(get_pose_net(config.cfg, 1, 1), device_ids=[0, 1])
        net3.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=opt.lr)
        optimizer3 = torch.optim.Adam(net3.parameters(), lr=opt.lr)

        if opt.resume:
            save_name = os.path.join(model_save_path, 'model_best.pth')
            save_name2 = os.path.join(model_save_path, 'model_best2.pth')
            save_name3 = os.path.join(model_save_path, 'model_best3.pth')
            load_checkpoint(save_name, net, optimizer)
            load_checkpoint(save_name2, net2, optimizer2)
            load_checkpoint(save_name3, net3, optimizer3)
        criterion = nn.BCELoss()
        N_train = len(train_img_masks)
        max_dice = 0
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters in network: ', n_params*3)

        val_dice = eval_net_batch_three_view(net,net2,net3, val_loader, real_test_mask_list, device)
        print('Validation Dice Coeff: {}'.format(val_dice))
        '''
        for epoch in range(opt.epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, opt.epochs))
            net.train()
            net2.train()
            net3.train()
            epoch_loss = 0
            #training
            for i, b in enumerate(train_loader):
                imgs = b['img'].to(device)
                true_masks = b['label'].to(device)

                imgs2 = b['img'].permute(1, 2, 0).to(device)
                true_masks2 = b['label'].permute(1, 2, 0).to(device)

                imgs3 = b['img'].permute(2, 0, 1).to(device)
                true_masks3 = b['label'].permute(2, 0, 1).to(device)


                masks_pred = net(imgs).squeeze(0)
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)
                loss1 = criterion(masks_probs_flat, true_masks_flat)

                masks_pred2 = net2(imgs2).squeeze(0)
                masks_probs_flat2 = masks_pred2.view(-1)
                true_masks_flat2 = true_masks2.view(-1)
                loss2 = criterion(masks_probs_flat2, true_masks_flat2)

                masks_pred3 = net3(imgs3).squeeze(0)
                masks_probs_flat3 = masks_pred3.view(-1)
                true_masks_flat3 = true_masks3.view(-1)
                loss3 = criterion(masks_probs_flat3, true_masks_flat3)

                loss=loss1+loss2+loss3
                epoch_loss += loss.item()
                if i%20==0:
                    dice_score1 = (2 * torch.dot(masks_probs_flat, true_masks_flat) + 1) / (
                                torch.sum(masks_probs_flat) + torch.sum(true_masks_flat) + 1)
                    dice_score2 = (2 * torch.dot(masks_probs_flat2, true_masks_flat2) + 1) / (
                            torch.sum(masks_probs_flat2) + torch.sum(true_masks_flat2) + 1)
                    dice_score3 = (2 * torch.dot(masks_probs_flat3, true_masks_flat3) + 1) / (
                            torch.sum(masks_probs_flat3) + torch.sum(true_masks_flat3) + 1)

                    dice_score=(dice_score1+dice_score2+dice_score3)/3

                    print('{0:.4f} --- loss: {1:.6f}, dice_score:{2:.6f}'.
                          format(i * opt.batch_size / N_train, loss.item(),dice_score))
                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                loss.backward()
                optimizer.step()
                optimizer2.step()
                optimizer3.step()

            print('Epoch finished ! Loss: {}'.format(epoch_loss / (len(train_loader)*opt.batch_size)))

            if epoch>=0:
                val_dice = eval_net_batch_three_view(net,net2,net3, val_loader, real_test_mask_list, device)
                print('Validation Dice Coeff: {}'.format(val_dice))
                if np.mean(val_dice) > max_dice:
                    max_dice = np.mean(val_dice)
                    print('New best performance! saving')
                    save_name = os.path.join(model_save_path, 'model_best.pth')
                    save_name2 = os.path.join(model_save_path, 'model_best2.pth')
                    save_name3 = os.path.join(model_save_path, 'model_best3.pth')
                    save_checkpoint(save_name, net, optimizer)
                    save_checkpoint(save_name2, net2, optimizer2)
                    save_checkpoint(save_name3, net3, optimizer3)
                    print('model saved to {}'.format(save_name))
                if (epoch + 1) % opt.log_interval == 0:
                    save_name = os.path.join(model_save_path, 'model_routine.pth')
                    save_name2 = os.path.join(model_save_path, 'model_routine2.pth')
                    save_name3 = os.path.join(model_save_path, 'model_routine3.pth')

                    save_checkpoint(save_name, net, optimizer)
                    save_checkpoint(save_name2, net2, optimizer2)
                    save_checkpoint(save_name3, net3, optimizer3)
                    print('model saved to {}'.format(save_name))
    '''