#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --time=40:00:00
#SBATCH --mem 32000
#SBATCH --job-name=mbHRNet
#SBATCH --mail-type=END
#SBATCH --mail-user=qc690@nyu.edu
#SBATCH -p nvidia
module purge
module load all
module load cuda/10.0
cd /home/qc690/Video/MS_Lesion_Seg
/home/qc690/anaconda3_3.4.1/bin/python  -u train_mbHRNet.py>log_mbHRNet_samedataset.txt