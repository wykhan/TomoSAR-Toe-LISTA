##dataset
import argparse
import os

parser = argparse.ArgumentParser(description='Generate TomoSAR training and testing data')
parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'res/data_8td_randA_randpphi_snr_train_k1_2'))
parser.add_argument('--training-points', type=int, default=10 * 128, help='samples per SNR or interval setting')
parser.add_argument('--testing-points', type=int, default=2 * 128, help='samples per SNR or interval setting')
args = parser.parse_args()
if args.training_points < 1 or args.testing_points < 1:
    parser.error('sample counts must be positive')
data_path = os.path.abspath(args.data_dir)
os.environ['TOMOSAR_DATA_DIR'] = data_path

from torch.utils.data import DataLoader
import torch
from paras12 import *
from algorithm_toe import *

trainingPoints = args.training_points
testingPoints = args.testing_points
if not os.path.exists(data_path):
    # 如果不存在，创建文件夹
    os.makedirs(data_path)
dataset_training_all = sparse_dataset(N, sparsityLevel, 1, snr=5, A=D)
dataset_testing_all = sparse_dataset(N, sparsityLevel, 1, snr=5, A=D)
# 遍历不同的 sig 值


for snr in range(1, 10, 1):
    # 初始化空列表，用于存储不同稀疏级别的数据集
    # sig = sig/10
    # interval=int(0.1*deta_rou*rou_s)
    # 遍历不同的稀疏级别
    # for sparsityLevel in range(1,3,1):
        # 生成训练和测试数据集
    sparsityLevel=1
    interval = 0
    dataset_training = sparse_dataset_double(N, sparsityLevel, trainingPoints, interval,snr=snr, A=D)
    dataset_testing = sparse_dataset_double(N, sparsityLevel, testingPoints, interval,snr=snr, A=D)
    
    # 将数据集的 X, Y, N 添加到对应的列表中
    # dataset_training_all = dataset_training
    dataset_training_all.X = torch.cat([dataset_training_all.X,dataset_training.X],0)
    dataset_training_all.Y = torch.cat([dataset_training_all.Y,dataset_training.Y],0)
    dataset_training_all.N = torch.cat([dataset_training_all.N,dataset_training.N],0)
    dataset_testing_all.X = torch.cat([dataset_testing_all.X,dataset_testing.X],0)
    dataset_testing_all.Y = torch.cat([dataset_testing_all.Y,dataset_testing.Y],0)
    dataset_testing_all.N = torch.cat([dataset_testing_all.N,dataset_testing.N],0)


snr = 5
for deta_rou in range(1, 30, 1):
    # 初始化空列表，用于存储不同稀疏级别的数据集
    # sig = sig/10
    interval=int(0.1*deta_rou*rou_s)
    # 遍历不同的稀疏级别
    # for sparsityLevel in range(1,3,1):
        # 生成训练和测试数据集
    sparsityLevel=2
    dataset_training = sparse_dataset_double(N, sparsityLevel, trainingPoints, interval,snr=snr, A=D)
    dataset_testing = sparse_dataset_double(N, sparsityLevel, testingPoints, interval,snr=snr, A=D)
    
    # 将数据集的 X, Y, N 添加到对应的列表中
    # dataset_training_all = dataset_training
    dataset_training_all.X = torch.cat([dataset_training_all.X,dataset_training.X],0)
    dataset_training_all.Y = torch.cat([dataset_training_all.Y,dataset_training.Y],0)
    dataset_training_all.N = torch.cat([dataset_training_all.N,dataset_training.N],0)
    dataset_testing_all.X = torch.cat([dataset_testing_all.X,dataset_testing.X],0)
    dataset_testing_all.Y = torch.cat([dataset_testing_all.Y,dataset_testing.Y],0)
    dataset_testing_all.N = torch.cat([dataset_testing_all.N,dataset_testing.N],0)



torch.save(dataset_training_all, os.path.join(data_path,"ula_training_data_randA_randphi_snr_8td.pt"))
torch.save(dataset_testing_all, os.path.join(data_path,"ula_testing_data_randA_randphi_snr_8td.pt"))
print(f"Saved training={len(dataset_training_all)} testing={len(dataset_testing_all)} to {data_path}")
from scipy import io
import os

io.savemat(os.path.join(data_path,'D.mat'),{'D':D}) 
# io.savemat(os.path.join(data_path,'A_u.mat'),{'A_u':A_u}) 
io.savemat(os.path.join(data_path,'dataset_training_X_8td.mat'),{'dataset_training_X':dataset_training.X.numpy()}) 
io.savemat(os.path.join(data_path,'dataset_training_Y_8td.mat'),{'dataset_training_Y':dataset_training.Y.numpy()}) 

## 存储实数复数拼接数据
