import sys

sys.path.append(r'../')
from torchstat import stat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import random
import os
import scipy as sp
import ot


from lib.models import Encoder, Generator
from lib.data_loader import trainingset_construct_SDE

# convey the parameters from command line
import argparse
##latent_dim = 4, batch = 500, lr = 0.0001, 全部MMD损失, 3×128
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--u_sensor', type=int, default=2)
parser.add_argument('--k_sensor', type=int, default=13)
parser.add_argument('--f_sensor', type=int, default=21)
parser.add_argument('--latent_dim', type=int, default=20)
parser.add_argument('--batch_val', type=int, default=500)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--mesh_size', type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)

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


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 87
setup_seed(seed)##3407 114514
def std_cal(A):
    mean = torch.mean(A, dim=0)
    A = (A - mean)
    std = 0
    for i in range(A.size(0)):
        std += torch.norm(A[i, :]) ** 2
    std = torch.sqrt(std / A.size(0))
    return std

class MMD_loss(nn.Module):
    'description'

    # function class which calculates the MMD distance of 2 distributions

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
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

def train(train_loader, enc, gen, optimizer_enc, optimizer_gen):
    enc.train()
    gen.train()
    for u, k, f, u_coor, k_coor, f_coor in train_loader:
        u = u.to(device)  ##[1000, 2]
        k = k.to(device)  ##[1000, 13]
        f = f.to(device)  ##[1000, 21]
        u_coor = u_coor.to(device)  ##[1000, 2]
        k_coor = k_coor.to(device)  ##[1000, 13]
        f_coor = f_coor.to(device)  ##[1000, 21]
        G = torch.cat((u, k, f), 1)
        z_p = torch.randn(args.batch_val, args.latent_dim).to(device)
        ##优化编码器
        optimizer_enc.zero_grad()
        z, mu, logvar = enc(u, k, f)
        u_gen, k_gen, f_gen = gen(z_p, u_coor, k_coor, f_coor)
        z_gen, mu_gen, logvar_gen = enc(u_gen.detach(), k_gen.detach(), f_gen.detach())
        u_recon, k_recon, f_recon = gen(z, u_coor, k_coor, f_coor)
        G_recon = torch.cat((u_recon, k_recon, f_recon), 1)
        loss_enc = MMD(z, z_p) - MMD(z_gen, z_p) + MMD(G_recon, G)
        loss_enc.backward()
        optimizer_enc.step()
        # loss_e1 = 0#MMD(z_p, z).item()
        # loss_e2 = 0#MMD(z_p, z_gen).item()
        # loss_e3 = 0#MMD(G_recon, G).item()
        ##优化生成器
        optimizer_gen.zero_grad()
        u_gen, k_gen, f_gen = gen(z_p, u_coor, k_coor, f_coor)
        G_gen = torch.cat((u_gen, k_gen, f_gen), 1)
        z_gen, mu_gen, logvar_gen = enc(u_gen, k_gen, f_gen)
        loss_gen = MMD(z_gen, z_p) + MMD(G_gen, G)
        loss_gen.backward()
        optimizer_gen.step()
        # loss_g1 = 0#MMD(z_gen, z_p).item()
        # loss_g2 = 0#MMD(G_gen, G).item()

    # return loss_g1, loss_g2, loss_e1, loss_e2, loss_e3


u_data = np.load(file=r'../database/SDE/u_ODE.npy')[0:args.data_size]
k_data = np.load(file=r'../database/SDE/k_ODE.npy')[0:args.data_size]
f_data = np.load(file=r'../database/SDE/f_ODE.npy')[0:args.data_size]
# calculate ground true for comparison
n_validate = 101  # number of validation points
test_coor = np.floor(np.linspace(0, 1, n_validate) * args.mesh_size).astype(int)
u_test = u_data[:, test_coor]
k_test = k_data[:, test_coor]
f_test = f_data[:, test_coor]
true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
# true_std_u = torch.std(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)  ##201
# true_std_k = torch.std(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)
true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)

nblock = 3  # 3 blocks = 4 hidden layers
width = 128
enc = Encoder(args.latent_dim, args.u_sensor, args.k_sensor, args.f_sensor, nblock, width, device)
gen = Generator(args.latent_dim, args.u_sensor, args.k_sensor, args.f_sensor, width, nblock, device)
optimizer_enc = optim.Adam(enc.parameters(), lr=args.lr, betas=(0.5, 0.9))
optimizer_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
# scheduler_enc = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=5000, gamma=0.1, last_epoch=-1)
# scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=5000, gamma=0.1, last_epoch=-1)

# define training data loader
u_coor = np.linspace(-1, 1, args.u_sensor) * np.ones([len(u_data), args.u_sensor])  ##(1000, 2)
k_coor = np.linspace(-1, 1, args.k_sensor) * np.ones([len(k_data), args.k_sensor])
f_coor = np.linspace(-1, 1, args.f_sensor) * np.ones([len(f_data), args.f_sensor])
x_u_coor = np.floor(np.linspace(0, 1, args.u_sensor) * args.mesh_size).astype(int)
x_k_coor = np.floor(np.linspace(0, 1, args.k_sensor) * args.mesh_size).astype(int)
x_f_coor = np.floor(np.linspace(0, 1, args.f_sensor) * args.mesh_size).astype(int)
k_training_data = k_data[0:args.data_size, x_k_coor]  ##(1000, 13)
u_training_data = u_data[0:args.data_size, x_u_coor]  ##(1000, 2)
f_training_data = f_data[0:args.data_size, x_f_coor]  ##(1000, 21)


train_loader = trainingset_construct_SDE(u_data=u_training_data, k_data=k_training_data, f_data=f_training_data,
                                             x_u=u_coor, x_k=k_coor, x_f=f_coor, batch_val=args.batch_val)

# train the network
def main1(seed):
    u = torch.tensor(u_test[0:1000]).to(device)
    # enc.initialize_weights()
    # gen.initialize_weights()
    u_mean_error, u_std_error, k_mean_error, k_std_error, time_history, loss_history, epochs = [[] for i in range(7)]
    loss_G1, loss_G2, loss_E1, loss_E2, loss_E3 = [[] for i in range(5)]
    for epoch in range(1, args.epoch+1):

        if epoch % 100 == 0 or epoch == 1:
            print('epoch:', epoch)

            with torch.no_grad():
                z = torch.randn(1000, args.latent_dim).to(device)
                coordinate = (torch.linspace(-1, 1, steps=n_validate) * torch.ones((1000, n_validate))).to(device)
                u_recon, k_recon = gen.reconstruct(z, coordinate, coordinate)
                mean_u = torch.mean(u_recon, dim=0)
                std_u = std_cal(u_recon)
                mean_k = torch.mean(k_recon, dim=0)
                std_k = std_cal(k_recon)
                mean_L2_error_u = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
                std_L2_error_u = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
                mean_L2_error_k = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
                std_L2_error_k = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()

                print('u mean error:', mean_L2_error_u, 'u std error:', std_L2_error_u)
                print('k mean error:', mean_L2_error_k, 'k std error:', std_L2_error_k)
                # if epoch >= args.epoch - 3000:
                u_mean_error.append(mean_L2_error_u)
                u_std_error.append(std_L2_error_u)
                k_mean_error.append(mean_L2_error_k)
                k_std_error.append(std_L2_error_k)
                ###W距离
                w_distance = W_distance(u, u_recon)
                # with open(f'w_distance_{args.latent_dim}.txt', 'a') as f:
                #     f.write(str(w_distance) + "\n")
                print(f'epoch={epoch}, w_distance={w_distance}')
        time_start = time.time()
        # loss_g1, loss_g2, loss_e1, loss_e2, loss_e3= train(train_loader, enc, gen, optimizer_enc, optimizer_gen)
        train(train_loader, enc, gen, optimizer_enc, optimizer_gen)
        time_stop = time.time()
        # scheduler_enc.step()
        # scheduler_gen.step()
        time_history.append(time_stop - time_start)
        # loss_G1.append(loss_g1)
        # loss_G2.append(loss_g2)
        # loss_E1.append(loss_e1)
        # loss_E2.append(loss_e2)
        # loss_E3.append(loss_e3)
    # np.save(f"results//loss_G1_{seed}", loss_G1)
    # np.save(f"results//loss_G2_{seed}", loss_G2)
    # np.save(f"results//loss_E1_{seed}", loss_E1)
    # np.save(f"results//loss_E2_{seed}", loss_E2)
    # np.save(f"results//loss_E3_{seed}", loss_E3)
    # np.save(f"results//u_mean_error", u_mean_error)##results//u_mean_error_{seed}
    # np.save(f"results//u_std_error", u_std_error)
    # np.save(f"results//k_mean_error", k_mean_error)
    # np.save(f"results//k_std_error", k_std_error)
    # np.save(f"time_history", time_history)

    print(f"Last ====== "
          f"u_mean_error = {np.mean(u_mean_error)}, u_std_error = {np.mean(u_std_error)},"
          f"k_mean_error = {np.mean(k_mean_error)}, k_std_error = {np.mean(k_std_error)}, time = {np.mean(time_history)}")
    torch.save(gen, f"results//gen.pth")

main1(seed)
