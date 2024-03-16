import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import scipy.linalg as la
import time
import random
import scipy as sp
import seaborn as sns
import ot


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

##设备
if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)

def PCA(data):
    mean_val = torch.mean(data, 0)
    data_normalize = (data - mean_val)
    cov = (1 / len(data)) * (data_normalize.T @ data_normalize)
    a, b = la.eig(cov.cpu())
    return a[0:15]


##load model
cl = 0.2
data_size = 1000
batch_val = 1000
latent_dim = 4
n_blocks = 3
width = 128
f_data_dim = 11



setup_seed(2)
x_axis_data = []
for x in range(15):
    x_axis_data.append(x)
GEA_1 = torch.load(f"square_exp//results//PI_GEA//GEA_6sensor_1.pth")
GEA_2 = torch.load(f"square_exp//results//PI_GEA//GEA_6sensor_0.5.pth")
GEA_3 = torch.load(f"square_exp//results//PI_GEA//GEA_6sensor_0.2.pth")

GEA_4 = torch.load(f"square_exp//result_new//GEA_11sensor_1.pth")
GEA_5 = torch.load(f"square_exp//result_new//GEA_11sensor_0.5.pth")
GEA_6 = torch.load(f"square_exp//result_new//GEA_11sensor_0.2.pth")

test_data_1 = np.load(f'square_exp//6sensor,l=1.npy')
test_data_2 = np.load(f'square_exp//6sensor,l=0.5.npy')
test_data_3 = np.load(f'square_exp//6sensor,l=0.2.npy')
test_coor = np.linspace(-1, 1, 101) * np.ones([data_size, 101])
f_1 = torch.tensor(test_data_1).to(device)
f_2 = torch.tensor(test_data_2).to(device)
f_3 = torch.tensor(test_data_3).to(device)
coor = torch.tensor(test_coor, dtype=torch.float32).to(device)
z = torch.randn(batch_val, latent_dim).to(device)


f_GEA_1 = GEA_1(z, coor)##生成的样本
f_GEA_2 = GEA_2(z, coor)
f_GEA_3 = GEA_3(z, coor)
f_GEA_4 = GEA_4(z, coor)
f_GEA_5 = GEA_5(z, coor)
f_GEA_6 = GEA_6(z, coor)

f_eig_1 = PCA(f_1)
f_GEA_eig_1 = PCA(f_GEA_1.detach().cpu())##6sensor1
f_eig_2 = PCA(f_2)
f_GEA_eig_2 = PCA(f_GEA_2.detach().cpu())##6sensor0.5
f_eig_3 = PCA(f_3)
f_GEA_eig_3 = PCA(f_GEA_3.detach().cpu())##6sensor0.2

f_GEA_eig_4 = PCA(f_GEA_4.detach().cpu())##11sensor1
f_GEA_eig_5 = PCA(f_GEA_5.detach().cpu())##11sensor0.5
f_GEA_eig_6 = PCA(f_GEA_6.detach().cpu())##11sensor0.2

plt.figure(figsize=(10,10))

##first
plt.subplot(2,2,1)

## w距离
y_axis_data_1 = np.loadtxt("square_exp//results//PI_GEA//w_distance_6sensor_1.txt")
y_axis_data_2 = np.loadtxt("square_exp//results//PI_GEA//w_distance_6sensor_0.5.txt")
y_axis_data_3 = np.loadtxt("square_exp//results//PI_GEA//w_distance_6sensor_0.2.txt")
y_axis_data_4 = np.loadtxt("square_exp//result_new//w_distance_11sensor_1.txt")
y_axis_data_5 = np.loadtxt("square_exp//result_new//w_distance_11sensor_0.5.txt")
y_axis_data_6 = np.loadtxt("square_exp//result_new//w_distance_11sensor_0.2.txt")
x_axis_data = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]  # x
plt.plot(x_axis_data, y_axis_data_1, 'rs-', alpha=0.5, linewidth=1, label='6 sensors, $l$=1.0')# 'bo-'表示蓝色_实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x_axis_data, y_axis_data_2, 'ro-', alpha=0.5, linewidth=1, label='6 sensors, $l$=0.5')
plt.plot(x_axis_data, y_axis_data_3, 'r*-', alpha=0.5, linewidth=1, label='6 sensors, $l$=0.2')
plt.plot(x_axis_data, y_axis_data_4, 'bs-', alpha=0.5, linewidth=1, label='11 sensors, $l$=1.0')
plt.plot(x_axis_data, y_axis_data_5, 'bo-', alpha=0.5, linewidth=1, label='11 sensors, $l$=0.5')
plt.plot(x_axis_data, y_axis_data_6, 'b*-', alpha=0.5, linewidth=1, label='11 sensors, $l$=0.2')
plt.legend()  # 显示上面的label
plt.xlabel('Training step')  # x_label
plt.ylabel('Wasserstein distance')  # y_label
# plt.subplots_adjust(bottom=0.01)

###second
plt.subplot(2,2,2)
epochs = []
for epoch in range(15):
    epochs.append(epoch)
plt.plot(epochs, f_eig_1, 'kp-', alpha=0.5, linewidth=1, label='Reference')
plt.plot(epochs, f_GEA_eig_1, 'r.--', alpha=0.5, linewidth=1, label='6 sensors')
plt.plot(epochs, f_GEA_eig_4, 'b.--', alpha=0.5, linewidth=1, label='11 sensors')
plt.legend()
plt.xlabel(u'Component')
plt.ylabel(u'Eigenvalue')
plt.title(f'$l$=1')

###third
plt.subplot(2,2,3)
plt.plot(epochs, f_eig_2, 'kp-', alpha=0.5, linewidth=1, label='Reference')
plt.plot(epochs, f_GEA_eig_2, 'r.--', alpha=0.5, linewidth=1, label='6 sensors')
plt.plot(epochs, f_GEA_eig_5, 'b.--', alpha=0.5, linewidth=1, label='11 sensors')
plt.legend()
plt.xlabel(u'Component')
plt.ylabel(u'Eigenvalue')
plt.title(f'$l$=0.5')

###fourth
plt.subplot(2,2,4)
plt.plot(epochs, f_eig_3, 'kp-', alpha=0.5, linewidth=1, label='Reference')
plt.plot(epochs, f_GEA_eig_3, 'r.--', alpha=0.5, linewidth=1, label='6 sensors')
plt.plot(epochs, f_GEA_eig_6, 'b.--', alpha=0.5, linewidth=1, label='11 sensors')
plt.legend()
plt.xlabel(u'Component')
plt.ylabel(u'Eigenvalue')
plt.title(f'$l$=0.2')


plt.subplots_adjust(hspace=0.3, wspace=0.15)
plt.savefig('sp_results.eps', bbox_inches='tight', dpi=300)
plt.show()

