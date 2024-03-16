import sys
sys.path.append(r'../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import scipy.linalg as la
from model import *
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import scipy as sp
import ot
import matplotlib.pyplot as plt
import random

# convey the parameters from command line
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--f_sensor', type=int, default=11)
parser.add_argument('--latent_dim', type=int, default=4)
parser.add_argument('--batch_val', type=int, default=500)
parser.add_argument('--epoch', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_p', type=int, default=101)
args = parser.parse_args()

##设备
if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)


##设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 111
setup_seed(seed)

def train_loader(dataset, batch_num):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_num,
        shuffle=True,  # change the sequence of the data every time
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader

def std_cal(A):
    mean = torch.mean(A, dim=0)
    A = (A - mean)
    std = 0
    for i in range(A.size(0)):
        std += torch.norm(A[i, :]) ** 2
    std = torch.sqrt(std / A.size(0))
    return std


class MyDataset(Dataset):
    def __init__(self, data, x):
        self.f = torch.from_numpy(data).float()
        self.x = torch.from_numpy(x).float()

    def __getitem__(self, index):
        f = self.f[index]
        coor = self.x[index]

        return f, coor

    def __len__(self):
        return len(self.f)


##损失
class MMD_loss(nn.Module):
    'description'

    # function class which calculates the MMD distance of 2 distributions

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):##默认kernel_mul=2.0, kernel_num=5
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
MMD = MMD_loss()




def W_distance(data1, data2):
    data1 = data1.clone().detach()
    data2 = data2.clone().detach()
    n = len(data1)
    data_dist1 = torch.zeros((n, 1))
    data_dist2 = torch.zeros((n, 1))
    for index in range(n):
        data_dist1[index, 0] = torch.norm(data1[index, :])
        data_dist2[index, 0] = torch.norm(data2[index, :])
    C1 = sp.spatial.distance.cdist(data_dist1, data_dist1)  # Compute distance between each pair of the two collections of inputs
    C2 = sp.spatial.distance.cdist(data_dist2, data_dist2)

    p = ot.unif(n)  # return a uniform histogram of length n_samples
    q = ot.unif(n)  # return a uniform histogram of length n_samples
    gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=False, log=True)

    return log0['gw_dist']


def PCA(data):
    mean_val = torch.mean(data, 0)
    data_normalize = (data - mean_val)
    cov = (1 / len(data)) * (data_normalize.T @ data_normalize)
    a, b = la.eig(cov.cpu())
    return a[0:15]


# training function
loss_D, loss_G = [], []
def train(train_loader, enc, gen, optimizer_enc, optimizer_gen):
    enc.train()
    gen.train()
    for f, f_coor in train_loader:
        f = f.to(device)
        f_coor = f_coor.to(device)
        z_p = torch.randn(args.batch_val, args.latent_dim).to(device)
        ##优化编码器
        optimizer_enc.zero_grad()
        z, mu, logvar = enc(f)
        f_recon = gen(z, f_coor)
        f_gen = gen(z_p, f_coor)
        z_gen, mu_gen, logvar_gen = enc(f_gen.detach())
        loss_enc = (MMD(z, z_p) - MMD(z_gen, z_p)) + MMD(f_recon, f)

        # losse1 = (MMD(z, z_p) - MMD(z_gen, z_p)).item()
        # losse2 = MMD(f_recon, f).item()
        loss_enc.backward()
        optimizer_enc.step()
        ##优化生成器
        optimizer_gen.zero_grad()
        f_gen = gen(z_p, f_coor)
        z_gen, mu_gen, logvar_gen = enc(f_gen)
        loss_gen = MMD(z_gen, z_p) + MMD(f_gen, f)

        # lossg1 = MMD(z_gen, z_p).item()
        # lossg2 = MMD(f_gen, f).item()
        loss_gen.backward()
        optimizer_gen.step()

    # return losse1, losse2, lossg1, lossg2


##加载数据
cl = 0.2
test_data = np.load(f'square_exp//6sensor,l={cl}.npy')[0: args.data_size]  ##(1000, 101)
test_coor = np.linspace(-1, 1, args.num_p) * np.ones([args.data_size, args.num_p])  ##(1000, 101)
index = np.floor(np.linspace(0, 1, args.f_sensor) * (args.num_p-1)).astype(int)

train_data = test_data[:, index]  ##(1000, 6)
train_coor = test_coor[:, index]  ##(1000, 6)
true_mean_f = torch.mean(torch.from_numpy(test_data), dim=0).type(torch.float).to(device)
true_std_f = std_cal(torch.from_numpy(test_data)).type(torch.float).to(device)

dataset = MyDataset(train_data, train_coor)
train_loader = train_loader(dataset, args.batch_val)

# define models
nblock = 2  # 3 blocks = 4 hidden layers
width = 64
##非高斯分布：0.2:3×128 + 3×64    0.5 and 1： 3×60 单sigmoid
##高斯分布： 3×60 单sigmoid 特征值很好
enc = Encoder(args.latent_dim, args.f_sensor, nblock, 64, device)
gen = Generator(args.latent_dim, args.f_sensor, width, nblock, device)
optimizer_enc = torch.optim.Adam(enc.parameters(), lr=args.lr, betas=(0.5, 0.9))
optimizer_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))

# train the network
if __name__ == "__main__":
    epochs = []
    # enc.initialize_weights()
    # gen.initialize_weights()
    lossE1, lossE2, lossG1, lossG2 = [[] for i in range(4)]
    f_mean_error, f_std_error = [], []
    for epoch in range(1, args.epoch+1):
        if epoch % 500 == 0 or epoch == 1:
            print('epoch:', epoch)

            with torch.no_grad():
                # epochs.append(epoch)
                f = torch.tensor(test_data[0:1000]).to(device)
                z = torch.randn(1000, args.latent_dim).to(device)
                coordinate = (torch.linspace(-1, 1, steps=args.num_p) * torch.ones((1000, args.num_p))).to(device)
                f_recon = gen(z, coordinate)
                # mean_f = torch.mean(f_recon, dim=0)
                # std_f = std_cal(f_recon)
                # mean_L2_error_f = (torch.norm(mean_f - true_mean_f) / torch.norm(true_mean_f)).cpu().numpy()
                # std_L2_error_f = (torch.norm(std_f - true_std_f) / torch.norm(true_std_f)).cpu().numpy()

                # print('f mean error:', mean_L2_error_f, 'f std error:', std_L2_error_f)
                # if epoch >= args.epoch - 1000:
                #     f_mean_error.append(mean_L2_error_f)
                #     f_std_error.append(std_L2_error_f)
                # w距离
                w_distance = W_distance(f, f_recon)
                with open(f'w_distance_{args.f_sensor}sensor_{cl}.txt', 'a') as f:
                    f.write(str(w_distance) + "\n")
                print(f'epoch={epoch}, w_distance={w_distance}')
        train(train_loader, enc, gen, optimizer_enc, optimizer_gen)
        # lossE1.append(losse1)
        # lossE2.append(losse2)
        # lossG1.append(lossg1)
        # lossG2.append(lossg2)
    # np.save("loss_G1", lossG1)
    # np.save("loss_G2", lossG2)
    # np.save("loss_E1", lossE1)
    # np.save("loss_E2", lossE2)
    # np.save(f"f_mean_error_{cl}", f_mean_error)  ##results//u_mean_error_{seed}
    # np.save(f"f_std_error_{cl}", f_std_error)
    # print(f"Last ====== "
    #       f"f_mean_error = {np.mean(f_mean_error)}, f_std_error = {np.mean(f_std_error)}")
    torch.save(gen, f'GEA_{args.f_sensor}sensor_{cl}.pth')
    ##特征值
    # gen.eval()
    # with torch.no_grad():
    #     f = torch.tensor(test_data).to(device)
    #     coor = torch.tensor(test_coor, dtype=torch.float32).to(device)
    #     z = torch.randn(args.data_size, args.latent_dim).to(device)
    #     f_recon = gen(z, coor)
    #     f_eig = PCA(f)
    #     f_recon_eig = PCA(f_recon.detach().cpu())
    #     plt.plot(range(15), f_eig, 'r-', label=u'f_eig')
    #     plt.plot(range(15), f_recon_eig, 'b-', label=u'f_recon_eig')
    #     plt.legend()
    #     plt.xlabel(u'component')
    #     plt.ylabel(u'eigen')
    #     plt.title('eigen')
    #     # plt.savefig('component.jpg')
    #     plt.show()

    ##可视化
    # plt.plot(epochs, w_distance, 'b-', label=u'w_distance')
    # plt.legend()
    # plt.xlabel(u'epoch')
    # plt.ylabel(u'w_distance')
    # plt.title('w_distance')
    # # plt.savefig(f'w_distance_{args.f_sensor}sensor.jpg')
    # plt.show()

























