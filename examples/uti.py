import matplotlib.pyplot as plt
from matplotlib import pyplot
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
from lib.models import Encoder, Generator, VEGAN_Encoder, VEGAN_Discriminator, VEGAN_Decoder, PIGAN_Generator, PIGAN_Discriminator
from torchstat import stat
from thop import profile

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
def std_cal(A):
    mean = torch.mean(A)
    A = (A - mean)
    std = 0
    for i in range(A.size(0)):
        std += torch.norm(A[i,:])**2
    std = torch.sqrt(std / A.size(0))
    return std
def mean_std(data):
    data_mean = torch.mean(data, dim=0).cpu().detach().numpy()
    data_std = torch.std(data, dim=0).cpu().detach().numpy()
    data_low = data_mean - data_std
    data_up = data_mean + data_std
    return (data_up, data_mean, data_low)

def memory_complexity():
    vegan_enc = VEGAN_Encoder(10, 2, 13, 21, 3, 128, device)
    vegan_gen = VEGAN_Decoder(10, 3, width, device)
    vegan_dis = VEGAN_Discriminator(2+13+21, width, device)
    # wgan_gen = PIGAN_Generator(latent_dim, 2, 13, 21, width, n_blocks, device)
    # wgan_dis = PIGAN_Discriminator(2+13+21, width, device)
    # GEA_enc = Encoder(10, 2, 13, 21, 3, width, device)
    # GEA_gen = Generator(10, 2, 13, 21, width, 3, device)


    stat(vegan_enc, (36,))

def distribution():
    GEA_1 = torch.load(f"forward//a=0.08//PI-GEA//gen.pth")
    # u_data_1 = np.load(f'E:/code/database/SDE/u_0.08_5000.npy')
    u_data_1_ref = np.load(f'E:/code/database/SDE/u_0.08_ref.npy')
    # k_data_1 = np.load(f'E:/code/database/SDE/k_0.08_5000.npy')
    k_data_1_ref = np.load(f'E:/code/database/SDE/k_0.08_ref.npy')

    GEA_2 = torch.load(f"forward//a=0.02//PI-GEA//gen.pth")
    # u_data_2 = np.load(f'E:/code/database/SDE/u_0.02_5000.npy')
    u_data_2_ref = np.load(f'E:/code/database/SDE/u_0.02_ref.npy')
    # k_data_2 = np.load(f'E:/code/database/SDE/k_0.02_5000.npy')
    k_data_2_ref = np.load(f'E:/code/database/SDE/k_0.02_ref.npy')

    test_coor = np.linspace(-1, 1, 101) * np.ones([data_size, 101])

    # u_1 = torch.tensor(u_data_1).to(device)
    # k_1 = torch.tensor(k_data_1).to(device)
    coor = torch.tensor(test_coor, dtype=torch.float32).to(device)
    z = torch.randn(batch_val, 10).to(device)
    u_recon_1, k_recon_1 = GEA_1.reconstruct(z, coor, coor)
    u_1_ref = torch.tensor(u_data_1_ref).to(device)
    k_1_ref = torch.tensor(k_data_1_ref).to(device)

    # u_2 = torch.tensor(u_data_2).to(device)
    # k_2 = torch.tensor(k_data_2).to(device)
    coor = torch.tensor(test_coor, dtype=torch.float32).to(device)
    z = torch.randn(batch_val, 20).to(device)
    u_recon_2, k_recon_2 = GEA_2.reconstruct(z, coor, coor)
    u_2_ref = torch.tensor(u_data_2_ref).to(device)
    k_2_ref = torch.tensor(k_data_2_ref).to(device)

    u_ref_1 = mean_std(u_1_ref)
    k_ref_1 = mean_std(k_1_ref)
    u_rec_1 =  mean_std(u_recon_1)
    k_rec_1 = mean_std(k_recon_1)

    u_ref_2 = mean_std(u_2_ref)
    k_ref_2 = mean_std(k_2_ref)
    u_rec_2 = mean_std(u_recon_2)
    k_rec_2 = mean_std(k_recon_2)

    plt.figure(figsize=(12, 12))

    plt.subplot(2,2,1)
    plt.plot(coor[0].cpu().detach().numpy(), u_ref_1[1], 'b-', alpha=0.5, linewidth=2, label='Mean reference')
    plt.plot(coor[0].cpu().detach().numpy(), u_rec_1[1], 'r-', alpha=0.5, linewidth=2, label='Mean generated')

    plt.fill_between(coor[0].cpu().detach().numpy(), u_ref_1[1], u_ref_1[2], facecolor="grey", label = 'Mean ± Std reference')
    plt.fill_between(coor[0].cpu().detach().numpy(), u_ref_1[1], u_ref_1[0], facecolor="grey")
    plt.fill_between(coor[0].cpu().detach().numpy(), u_rec_1[1], u_rec_1[2], facecolor="brown", label = 'Mean ± Std generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), u_rec_1[1], u_rec_1[0], facecolor="brown")
    plt.xlabel(u'$x$')
    plt.ylabel(u'$u(x)$')
    plt.legend()
    plt.title("$a = 0.08$")


    plt.subplot(2, 2, 2)
    plt.plot(coor[0].cpu().detach().numpy(), k_ref_1[1], 'b-', alpha=0.5, linewidth=2, label='Mean reference')
    plt.plot(coor[0].cpu().detach().numpy(), k_rec_1[1], 'r-', alpha=0.5, linewidth=2, label='Mean generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), k_ref_1[1], k_ref_1[2], facecolor="grey", label = 'Mean ± Std reference')
    plt.fill_between(coor[0].cpu().detach().numpy(), k_ref_1[1], k_ref_1[0], facecolor="grey")
    plt.fill_between(coor[0].cpu().detach().numpy(), k_rec_1[1], k_rec_1[2], facecolor="brown", label = 'Mean ± Std generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), k_rec_1[1], k_rec_1[0], facecolor="brown")
    plt.xlabel(u'$x$')
    plt.ylabel(u'$k(x)$')
    plt.legend()
    plt.title("$a = 0.08$")

    plt.subplot(2,2,3)
    plt.plot(coor[0].cpu().detach().numpy(), u_ref_2[1], 'b-', alpha=0.5, linewidth=2, label='Mean reference')
    plt.plot(coor[0].cpu().detach().numpy(), u_rec_2[1], 'r-', alpha=0.5, linewidth=2, label='Mean generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), u_ref_2[1], u_ref_2[2], facecolor="grey", label='Mean ± Std reference')
    plt.fill_between(coor[0].cpu().detach().numpy(), u_ref_2[1], u_ref_2[0], facecolor="grey")
    plt.fill_between(coor[0].cpu().detach().numpy(), u_rec_2[1], u_rec_2[2], facecolor="brown", label='Mean ± Std generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), u_rec_2[1], u_rec_2[0], facecolor="brown")
    plt.xlabel(u'$x$')
    plt.ylabel(u'$u(x)$')
    plt.legend()
    plt.title("$a = 0.02$")


    plt.subplot(2, 2, 4)
    plt.plot(coor[0].cpu().detach().numpy(), k_ref_2[1], 'b-', alpha=0.5, linewidth=2, label='Mean reference')
    plt.plot(coor[0].cpu().detach().numpy(), k_rec_2[1], 'r-', alpha=0.5, linewidth=2, label='Mean generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), k_ref_2[1], k_ref_2[2], facecolor="grey", label='Mean ± Std reference')
    plt.fill_between(coor[0].cpu().detach().numpy(), k_ref_2[1], k_ref_2[0], facecolor="grey")
    plt.fill_between(coor[0].cpu().detach().numpy(), k_rec_2[1], k_rec_2[2], facecolor="brown", label='Mean ± Std generated')
    plt.fill_between(coor[0].cpu().detach().numpy(), k_rec_2[1], k_rec_2[0], facecolor="brown")
    plt.xlabel(u'$x$')
    plt.ylabel(u'$k(x)$')
    plt.legend()
    plt.title("$a = 0.02$")


    #plt.title(f'l={cl}')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(f'distribution.eps', bbox_inches='tight', dpi=300)
    plt.show()

def eigenvalue():
    x_axis_data = []
    for x in range(15):
        x_axis_data.append(x)
    vegan = Decoder(latent_dim, n_blocks, width, device)
    #wgan = PIGAN_Generator(latent_dim, 2, 13, 41, width, n_blocks)
    vegan_dict = torch.load(f"forward//high_dim//a=0.08//model")
    vegan.load_state_dict(vegan_dict.state_dict())
    wgan_dict = torch.load(f"PI_WGAN//high_dim//model")
    #wgan.load_state_dict(wgan_dict)
    test_data = np.load(f'u_high_a=0.08.npy')
    test_coor = np.linspace(-1, 1, 101) * np.ones([data_size, 101])
    f = torch.tensor(test_data).to(device)
    coor = torch.tensor(test_coor, dtype=torch.float32).to(device)
    z = torch.randn(batch_val, latent_dim).to(device)

    f_eig = PCA(f)##训练的样本
    f_recon_vegan = vae_6sensor.decode(z, coor)##生成的样本
    f_recon_wgan = wgan_6sensor(z, coor)##生成的样本


    f_recon_6eig = PCA(f_recon_6.detach().cpu())
    f_recon_11eig = PCA(f_recon_11.detach().cpu())
    f_recon_6eig_wgan = PCA(f_recon_6_wgan.detach().cpu())
    f_recon_11eig_wgan = PCA(f_recon_11_wgan.detach().cpu())

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_axis_data, f_eig, 'k*-', alpha=0.5, linewidth=2, label='reference')
    ax.plot(x_axis_data, f_recon_6eig, 'rs--', alpha=0.5, linewidth=2, label='PI-VEGAN 6 sensors')
    ax.plot(x_axis_data, f_recon_6eig_wgan, 'bs--', alpha=0.5, linewidth=2, label='PI-WGAN 6 sensors')
    ax.plot(x_axis_data, f_recon_11eig, 'ro--', alpha=0.5, linewidth=2, label='PI-VEGAN 11 sensors')
    ax.plot(x_axis_data, f_recon_11eig_wgan, 'bo--', alpha=0.5, linewidth=2, label='PI-WGAN 11 sensors')
    ##绘制子图
    axins = ax.inset_axes((8/14, 8/28, 0.4, 0.3))
    axins.plot(x_axis_data, f_eig, 'k*-', alpha=0.5, linewidth=2, label='reference')
    axins.plot(x_axis_data, f_recon_6eig, 'rs--', alpha=0.5, linewidth=2, label='PI-VEGAN 6 sensors')
    axins.plot(x_axis_data, f_recon_6eig_wgan, 'bs--', alpha=0.5, linewidth=2, label='PI-WGAN 6 sensors')
    axins.plot(x_axis_data, f_recon_11eig, 'ro--', alpha=0.5, linewidth=2, label='PI-VEGAN 11 sensors')
    axins.plot(x_axis_data, f_recon_11eig_wgan, 'bo--', alpha=0.5, linewidth=2, label='PI-WGAN 11 sensors')
    # 设置放大区间
    zone_left = 5
    zone_right = 9

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05# y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = x_axis_data[zone_left]-(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
    xlim1 = x_axis_data[zone_right]+(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((f_eig[zone_left:zone_right], f_recon_6eig[zone_left:zone_right],
                   f_recon_6eig_wgan[zone_left:zone_right],f_recon_11eig[zone_left:zone_right],
                   f_recon_11eig_wgan[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")

    # 画两条线
    xy = (xlim0,ylim0)
    xy2 = (xlim0,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
    axins.add_artist(con)

    xy = (xlim1,ylim0)
    xy2 = (xlim1,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
    axins.add_artist(con)

    ax.legend()
    ax.set_xlabel(u'Component')
    ax.set_ylabel(u'Eigenvalue')
    plt.title(f'l={cl}')
    plt.savefig(f'component_l={cl}.pdf', bbox_inches='tight')
    plt.show()

def draw_line(name_of_alg,color_index,datas):
    palette = pyplot.get_cmap('Set1')
    color=palette(color_index)
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    plt.plot(epochs, avg, color=color,label=name_of_alg,linewidth=3.5)
    plt.fill_between(epochs, r1, r2, color=color, alpha=0.2)
def many_loss():
    plt.style.use('seaborn-whitegrid')
    palette = pyplot.get_cmap('Set1')
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 32,
    }

    fig=plt.figure(figsize=(8,7))
    #这里随机给了alldata1和alldata2数据用于测试


    loss1 = np.load(f"results//forward//loss_D_22.npy", allow_pickle=True)
    loss2 = np.load(f"results//forward//loss_D_98.npy", allow_pickle=True)
    loss3 = np.load(f"results//forward//loss_D_1000.npy", allow_pickle=True)
    loss4 = np.load(f"results//forward//loss_D_3407.npy", allow_pickle=True)
    loss5 = np.load(f"results//forward//loss_D_3803.npy", allow_pickle=True)
    loss1_wgan = np.load(f"results//forward//PI_WGAN_loss_D_22.npy", allow_pickle=True)
    loss2_wgan = np.load(f"results//forward//PI_WGAN_loss_D_98.npy", allow_pickle=True)
    loss3_wgan = np.load(f"results//forward//PI_WGAN_loss_D_1000.npy", allow_pickle=True)
    loss4_wgan = np.load(f"results//forward//PI_WGAN_loss_D_3407.npy", allow_pickle=True)
    loss5_wgan = np.load(f"results//forward//PI_WGAN_loss_D_3803.npy", allow_pickle=True)
    # loss1 = np.load(f"results//u_std_error_22.npy", allow_pickle=True)
    # loss2 = np.load(f"results//u_std_error_98.npy", allow_pickle=True)
    # loss3 = np.load(f"results//u_std_error_1000.npy", allow_pickle=True)
    # loss4 = np.load(f"results//u_std_error_3407.npy", allow_pickle=True)
    # loss5 = np.load(f"results//u_std_error_3803.npy", allow_pickle=True)
    # loss1_wgan = np.load(f"results//PI_WGAN_u_std_error_22.npy", allow_pickle=True)
    # loss2_wgan = np.load(f"results//PI_WGAN_u_std_error_98.npy", allow_pickle=True)
    # loss3_wgan = np.load(f"results//PI_WGAN_u_std_error_1000.npy", allow_pickle=True)
    # loss4_wgan = np.load(f"results//PI_WGAN_u_std_error_3407.npy", allow_pickle=True)
    # loss5_wgan = np.load(f"results//PI_WGAN_u_std_error_3803.npy", allow_pickle=True)

    # loss1 = max_abs_scaler.fit_transform(loss1)
    # loss2 = max_abs_scaler.fit_transform(loss2)
    # loss3 = max_abs_scaler.fit_transform(loss3)
    # loss4 = max_abs_scaler.fit_transform(loss4)
    # loss5 = max_abs_scaler.fit_transform(loss5)


    alldata1=[]#算法2所有纵坐标数据
    alldata1.append(loss1)
    alldata1.append(loss2)
    alldata1.append(loss3)
    alldata1.append(loss4)
    alldata1.append(loss5)
    alldata1=np.array(alldata1)

    alldata2=[]#算法2所有纵坐标数据
    alldata2.append(loss1_wgan)
    alldata2.append(loss2_wgan)
    alldata2.append(loss3_wgan)
    alldata2.append(loss4_wgan)
    alldata2.append(loss5_wgan)
    alldata2=np.array(alldata2)


    # from scipy.signal import savgol_filter
    # alldata2 = savgol_filter(alldata2, 5, 3, mode= 'nearest')


    epochs = []
    for epoch in range(10000):
        epochs.append(epoch)

    draw_line("alg1",1,alldata1)
    # draw_line("alg2",2,alldata2)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('epochs',fontsize=32)
    plt.ylabel('loss',fontsize=32)
    plt.legend(loc='upper left',prop=font1)
    plt.title("loss",fontsize=34)
    plt.show()

def error_curve():
    epochs = []
    for epoch in range(10001):
        if epoch % 1000 == 0:
            epochs.append(epoch)
    i = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    u_mean_error = np.load("forward//dim=4//all process//u_mean_error.npy")[i]
    k_mean_error = np.load("forward//dim=4//all process//k_mean_error.npy")[i]
    u_std_error = np.load("forward//dim=4//all process//u_std_error.npy")[i]
    k_std_error = np.load("forward//dim=4//all process//k_std_error.npy")[i]


    plt.plot(epochs, u_mean_error, 'ro-', alpha=0.5, linewidth=2, label='u mean error')# 'bo-'表示蓝色_实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.plot(epochs, u_std_error, 'ro--', alpha=0.5, linewidth=2, label='u std error')
    plt.plot(epochs, k_mean_error, 'bo-', alpha=0.5, linewidth=2, label='k mean error')
    plt.plot(epochs, k_std_error, 'bo--', alpha=0.5, linewidth=2, label='k std error')

    plt.legend()  # 显示上面的label
    plt.xlabel('Training step')  # x_label
    plt.ylabel('Relative error')  # y_label

    plt.savefig('error_epoch.eps', dpi=300, bbox_inches='tight')
    plt.show()

def eigenvalue_():
    cl = 1
    data_size = 1000
    batch_val = 1000
    latent_dim = 20

    wgan_6sensor = Decoder(latent_dim, 3, 128, device)
    # wgan_11sensor = Generator()
    wgan_6sensor_dict = torch.load(f"forward//high_dim//a=0.08//model")
    wgan_6sensor.load_state_dict(wgan_6sensor_dict.state_dict())
    # wgan_11sensor_dict = torch.load(f"PI_WGAN//exp//model_sensor=11, l={cl}")
    # wgan_11sensor.load_state_dict(wgan_11sensor_dict)

    # vae_6sensor = torch.load(f"exp//model_6sensor_{cl}")
    # vae_11sensor = torch.load(f"exp//model_11sensor_{cl}")
    #vae_10000_6sensor = torch.load(f"square_exp//10000snapshots//model10000_l={cl}_sensor=6")
    #vae_10000_11sensor = torch.load(f"square_exp//10000snapshots//model10000_l={cl}_sensor=11")

    test_data = np.load(f'u_0.08_re.npy')
    test_coor = np.linspace(-1, 1, 101) * np.ones([data_size, 101])
    f = torch.tensor(test_data).to(device)
    coor = torch.tensor(test_coor, dtype=torch.float32).to(device)
    z = torch.randn(batch_val, latent_dim).to(device)

    f_eig = PCA(f)##训练的样本
    # f_recon_6 = vae_6sensor.decode(z, coor)##生成的样本
    # f_recon_11 = vae_11sensor.decode(z, coor)##生成的样本
    f_recon_6_wgan, _ = wgan_6sensor.funval_cal(z, coor, coor)##生成的样本
    # f_recon_11_wgan = wgan_11sensor(z, coor)##生成的样本
    #f_recon_6_10000 = vae_10000_6sensor.decode(z, coor)
    #f_recon_11_10000 = vae_10000_11sensor.decode(z, coor)


    # f_recon_6eig = PCA(f_recon_6.detach().cpu())
    # f_recon_11eig = PCA(f_recon_11.detach().cpu())
    f_recon_6eig_wgan = PCA(f_recon_6_wgan.detach().cpu())
    # f_recon_11eig_wgan = PCA(f_recon_11_wgan.detach().cpu())
    #f_recon_6eig_10000 = PCA(f_recon_6_10000.detach().cpu())
    #f_recon_11eig_10000 = PCA(f_recon_11_10000.detach().cpu())

    plt.plot(range(15), f_eig, 'k*-', alpha=0.5, linewidth=2, label='reference')
    # plt.plot(range(15), f_recon_6eig, 'rs--', alpha=0.5, linewidth=2, label='PI-GVAE 6 sensors')
    plt.plot(range(15), f_recon_6eig_wgan, 'ro--', alpha=0.5, linewidth=2, label='PI-VEGAN')
    # plt.plot(range(15), f_recon_11eig, 'bs--', alpha=0.5, linewidth=2, label='PI-GVAE 11 sensors')
    # plt.plot(range(15), f_recon_11eig_wgan, 'bo--', alpha=0.5, linewidth=2, label='PI-WGAN 11 sensors')
    #plt.plot(range(15), f_recon_6eig_10000, 'rp--', alpha=0.5, linewidth=2, label='PI-GVAE 6 sensors 10000 snapshots')
    #plt.plot(range(15), f_recon_11eig_10000, 'bp--', alpha=0.5, linewidth=2, label='PI-GVAE 11 sensors 10000 snapshots')
    plt.legend()
    plt.xlabel(u'Component')
    plt.ylabel(u'Eigenvalue')
    #plt.title(f'l={cl}')
    # plt.savefig(f'component_l={cl}.jpg')
    plt.savefig(f'high_dim_component.pdf')
    plt.show()

def w_distance():
    y_axis_data_1 = np.loadtxt("exp//w_distance_11sensor_1.txt")
    y_axis_data_2 = np.loadtxt("exp//w_distance_11sensor_0.5.txt")
    y_axis_data_3 = np.loadtxt("exp//w_distance_11sensor_0.2.txt")
    y_axis_data_4 = np.loadtxt("exp//w_distance_6sensor_1.txt")
    y_axis_data_5 = np.loadtxt("exp//w_distance_6sensor_0.5.txt")
    y_axis_data_6 = np.loadtxt("exp//w_distance_6sensor_0.2.txt")

    x_axis_data = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # x

    plt.plot(x_axis_data, y_axis_data_1, 'b*--', alpha=0.5, linewidth=2,
             label='11 sensors, l=1.0')  # 'bo-'表示蓝色_实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.plot(x_axis_data, y_axis_data_2, 'bs--', alpha=0.5, linewidth=2, label='11 sensors, l=0.5')
    plt.plot(x_axis_data, y_axis_data_3, 'bo--', alpha=0.5, linewidth=2, label='11 sensors, l=0.2')
    plt.plot(x_axis_data, y_axis_data_4, 'r*--', alpha=0.5, linewidth=2, label='6 sensors, l=1.0')
    plt.plot(x_axis_data, y_axis_data_5, 'rs--', alpha=0.5, linewidth=2, label='6 sensors, l=0.5')
    plt.plot(x_axis_data, y_axis_data_6, 'ro--', alpha=0.5, linewidth=2, label='6 sensors, l=0.2')

    plt.legend()  # 显示上面的label
    plt.xlabel('Training step')  # x_label
    plt.ylabel('Wasserstein distance')  # y_label

    plt.savefig('w_distance_exp.jpg', bbox_inches='tight')
    plt.show()

def loss():
    plt.figure(figsize=(14,6))

    epochs = []
    for epoch in range(10000):
        epochs.append(epoch)
    seed = 3407
    loss1 = np.load(f"forward//dim=4//PI-GEA//loss_E1_{seed}.npy", allow_pickle=True)
    loss2 = np.load(f"forward//dim=4//PI-GEA//loss_E2_{seed}.npy", allow_pickle=True)
    loss3 = np.load(f"forward//dim=4//PI-GEA//loss_E3_{seed}.npy", allow_pickle=True)
    loss4 = np.load(f"forward//dim=4//PI-GEA//loss_G1_{seed}.npy", allow_pickle=True)
    loss5 = np.load(f"forward//dim=4//PI-GEA//loss_G2_{seed}.npy", allow_pickle=True)

    plt.subplot(1,2,1)
    plt.plot(epochs, loss3, 'y-', label=u'Reconstruction loss of the encoder', linewidth=1.0)##(G, G_recon)
    # ax.plot(epochs, loss4, 'r-', label=u'Adversarial loss of the generator', linewidth=1.0)##kl_gen 对抗
    plt.plot(epochs, loss5, 'g-', label=u'Generation loss of the generator', linewidth=1.0)##(z_p, z_gen)
    # ax.plot(epochs, (loss1-loss2), 'b-', label=u'Adversarial loss of the encoder', linewidth=1.0)##对抗
    plt.legend()
    plt.xlabel(u'Training step')
    plt.ylabel(u'Loss')
    # plt.title('PI-GEA')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss4, 'r-', label=u'Adversarial loss of the generator', linewidth=1.0)##kl_gen 对抗
    plt.plot(epochs, (loss1-loss2), 'b-', label=u'Adversarial loss of the encoder', linewidth=1.0)##对抗
    plt.legend()
    plt.xlabel(u'Training step')
    plt.ylabel(u'Loss')

    # plt.tight_layout()
    plt.savefig('loss.eps', dpi=300, bbox_inches='tight')
    plt.show()

def error(type, n=30):
    if type == 1:
        u_data = np.load(file=r'../database/SDE/u_ODE.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_ODE_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)


        u_means_1 = np.load("forward//dim=4//PI-GEA//u_mean_error.npy")[-n:-1]##forward//a=0.08//
        u_stds_1 = np.load("forward//dim=4//PI-GEA//u_std_error.npy")[-n:-1]

        u_means_2 = np.load("forward//dim=4//PI-WGAN//PI_WGAN_u_mean_error.npy")[-n:-1]##30
        u_stds_2 = np.load("forward//dim=4//PI-WGAN//PI_WGAN_u_std_error.npy")[-n:-1]


        u_means_3 = np.load("forward//dim=4//PI-VEGAN//VEGAN_u_mean_error.npy")[-n:-1]##30
        u_stds_3 = np.load("forward//dim=4//PI-VEGAN//VEGAN_u_std_error.npy")[-n:-1]

        u_means_4 = np.load("forward//2000//PI-GEA//u_mean_error.npy")[-n:-1]##30
        u_stds_4 = np.load("forward//2000//PI-GEA//u_std_error.npy")[-n:-1]

        u_means_5 = np.load("forward//2000//PI-VAE//vae_u_mean_error.npy")[-n:-1]##30
        u_stds_5 = np.load("forward//2000//PI-VAE//vae_u_std_error.npy")[-n:-1]


        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))

        u_mean_5 = torch.mean(torch.from_numpy(u_means_5), dim=0)
        u_mean_error_5 = torch.std(torch.from_numpy(u_means_5))
        u_std_5 = torch.mean(torch.from_numpy(u_stds_5), dim=0)
        u_std_error_5 = torch.std(torch.from_numpy(u_stds_5))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 18*bar_width, 3*bar_width)


        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_5, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_5, u_std_ref]

        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, u_mean_error_5, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, u_std_error_5, 0]
        tick_label = ['PI-GEA-1', 'PI-WGAN', 'PI-VEGAN', 'PI-GEA-2', 'PI-VAE', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 0.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('forward_error_compare.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    if type == 2:
        u_data = np.load(file=r'../database/SDE/u_ODE.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_ODE_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)
        # calculate u reference solution


        u_means_1 = np.load("forward//dim=2//u_mean_error.npy")[-n:-1]##forward//a=0.08//
        u_stds_1 = np.load("forward//dim=2//u_std_error.npy")[-n:-1]

        u_means_2 = np.load("forward//dim=4//PI-GEA//u_mean_error.npy")[-n:-1]##30
        u_stds_2 = np.load("forward//dim=4//PI-GEA//u_std_error.npy")[-n:-1]

        u_means_3 = np.load("forward//dim=20//PI-GEA//u_mean_error.npy")[-n:-1]##30
        u_stds_3 = np.load("forward//dim=20//PI-GEA//u_std_error.npy")[-n:-1]

        u_means_4 = np.load("forward//300全//u_mean_error.npy")[-n:-1]##30
        u_stds_4 = np.load("forward//300全//u_std_error.npy")[-n:-1]

        u_means_5 = np.load("forward//2000//PI-GEA//u_mean_error.npy")[-n:-1]##30
        u_stds_5 = np.load("forward//2000//PI-GEA//u_std_error.npy")[-n:-1]

        u_means_6 = np.load("forward//5000//u_mean_error.npy")[-n:-1]  ##30
        u_stds_6 = np.load("forward//5000//u_std_error.npy")[-n:-1]

        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))

        u_mean_5 = torch.mean(torch.from_numpy(u_means_5), dim=0)
        u_mean_error_5 = torch.std(torch.from_numpy(u_means_5))
        u_std_5 = torch.mean(torch.from_numpy(u_stds_5), dim=0)
        u_std_error_5 = torch.std(torch.from_numpy(u_stds_5))

        u_mean_6 = torch.mean(torch.from_numpy(u_means_6), dim=0)
        u_mean_error_6 = torch.std(torch.from_numpy(u_means_6))
        u_std_6 = torch.mean(torch.from_numpy(u_stds_6), dim=0)
        u_std_error_6 = torch.std(torch.from_numpy(u_stds_6))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 21 * bar_width, 3 * bar_width)

        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_5, u_mean_6, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_5, u_std_6, u_std_ref]

        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, u_mean_error_5, u_mean_error_6, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, u_std_error_5, u_std_error_6, 0]
        tick_label = ['PI-GEA-1', 'PI-GEA-2', 'PI-GEA-3', 'PI-GEA-4', 'PI-GEA-5', 'PI-GEA-6', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 0.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('forward_error.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    if type == 3:

        u_data = np.load(file=r'../database/SDE/u_ODE.npy')[0:data_size]
        k_data = np.load(file=r'../database/SDE/k_ODE.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_ODE_ref.npy')[0:data_size]
        k_data_re = np.load(file=r'../database/SDE/k_ODE_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        k_test = k_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        k_test_re = k_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)  ##201
        true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)
        mean_k = torch.mean(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        std_k = std_cal(torch.from_numpy(k_test_re)).type(torch.float).to(device)
        # calculate u reference solution

        u_means_1 = np.load("inverse//PI-GEA//1000//u_mean_error.npy")[-n:-1]
        k_means_1 = np.load("inverse//PI-GEA//1000//k_mean_error.npy")[-n:-1]
        u_stds_1 = np.load("inverse//PI-GEA//1000//u_std_error.npy")[-n:-1]
        k_stds_1 = np.load("inverse//PI-GEA//1000//k_std_error.npy")[-n:-1]

        u_means_2 = np.load("inverse//PI-WGAN//PI_WGAN_u_mean_error.npy")[-n:-1]  ##30
        k_means_2 = np.load("inverse//PI-WGAN//PI_WGAN_k_mean_error.npy")[-n:-1]
        u_stds_2 = np.load("inverse//PI-WGAN//PI_WGAN_u_std_error.npy")[-n:-1]
        k_stds_2 = np.load("inverse//PI-WGAN//PI_WGAN_k_std_error.npy")[-n:-1]

        u_means_3 = np.load("inverse//PI-VEGAN//u_mean_error.npy")[-n:-1]  ##30
        k_means_3 = np.load("inverse//PI-VEGAN//k_mean_error.npy")[-n:-1]
        u_stds_3 = np.load("inverse//PI-VEGAN//u_std_error.npy")[-n:-1]
        k_stds_3 = np.load("inverse//PI-VEGAN//k_std_error.npy")[-n:-1]

        u_means_4 = np.load("inverse//PI-GEA//2000//u_mean_error.npy")[-n:-1]  ##30
        k_means_4 = np.load("inverse//PI-GEA//2000//k_mean_error.npy")[-n:-1]
        u_stds_4 = np.load("inverse//PI-GEA//2000//u_std_error.npy")[-n:-1]
        k_stds_4 = np.load("inverse//PI-GEA//2000//k_std_error.npy")[-n:-1]

        u_means_5 = np.load("inverse//PI-VAE//vae_u_mean_error.npy")[-n:-1]  ##30
        k_means_5 = np.load("inverse//PI-VAE//vae_k_mean_error.npy")[-n:-1]
        u_stds_5 = np.load("inverse//PI-VAE//vae_u_std_error.npy")[-n:-1]
        k_stds_5 = np.load("inverse//PI-VAE//vae_k_std_error.npy")[-n:-1]


        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))
        k_mean_1 = torch.mean(torch.from_numpy(k_means_1), dim=0)
        k_mean_error_1 = torch.std(torch.from_numpy(k_means_1))
        k_std_1 = torch.mean(torch.from_numpy(k_stds_1), dim=0)
        k_std_error_1 = torch.std(torch.from_numpy(k_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))
        k_mean_2 = torch.mean(torch.from_numpy(k_means_2), dim=0)
        k_mean_error_2 = torch.std(torch.from_numpy(k_means_2))
        k_std_2 = torch.mean(torch.from_numpy(k_stds_2), dim=0)
        k_std_error_2 = torch.std(torch.from_numpy(k_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))
        k_mean_3 = torch.mean(torch.from_numpy(k_means_3), dim=0)
        k_mean_error_3 = torch.std(torch.from_numpy(k_means_3))
        k_std_3 = torch.mean(torch.from_numpy(k_stds_3), dim=0)
        k_std_error_3 = torch.std(torch.from_numpy(k_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))
        k_mean_4 = torch.mean(torch.from_numpy(k_means_4), dim=0)
        k_mean_error_4 = torch.std(torch.from_numpy(k_means_4))
        k_std_4 = torch.mean(torch.from_numpy(k_stds_4), dim=0)
        k_std_error_4 = torch.std(torch.from_numpy(k_stds_4))

        u_mean_5 = torch.mean(torch.from_numpy(u_means_5), dim=0)
        u_mean_error_5 = torch.std(torch.from_numpy(u_means_5))
        u_std_5 = torch.mean(torch.from_numpy(u_stds_5), dim=0)
        u_std_error_5 = torch.std(torch.from_numpy(u_stds_5))
        k_mean_5 = torch.mean(torch.from_numpy(k_means_5), dim=0)
        k_mean_error_5 = torch.std(torch.from_numpy(k_means_5))
        k_std_5 = torch.mean(torch.from_numpy(k_stds_5), dim=0)
        k_std_error_5 = torch.std(torch.from_numpy(k_stds_5))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
        k_mean_ref = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
        k_std_ref = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 30 * bar_width, 5 * bar_width)
        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_5, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_5, u_std_ref]
        k_mean = [k_mean_1, k_mean_2, k_mean_3, k_mean_4, k_mean_5, k_mean_ref]
        k_std = [k_std_1, k_std_2, k_std_3, k_std_4, k_std_5, k_std_ref]
        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, u_mean_error_5, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, u_std_error_5, 0]
        k_mean_err = [k_mean_error_1, k_mean_error_2, k_mean_error_3, k_mean_error_4, k_mean_error_5, 0]
        k_std_err = [k_std_error_1, k_std_error_2, k_std_error_3, k_std_error_4, k_std_error_5, 0]
        tick_label = ['PI-GEA-1', 'PI-WGAN', 'PI-VEGAN', 'PI-GEA-2', 'PI-VAE', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        plt.figure(figsize=(16, 6))
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, k_mean, bar_width, color=['orange'], yerr=k_mean_err, error_kw=error_params1, label='k mean')
        plt.bar(x + bar_width, k_std, bar_width, color=['pink'], yerr=k_std_err, error_kw=error_params2, label='k std')
        plt.bar(x + 2 * bar_width, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + 3 * bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 1.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('inverse_error.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    if type == 4:

        u_data = np.load(file=r'../database/SDE/u_ODE.npy')[0:data_size]
        k_data = np.load(file=r'../database/SDE/k_ODE.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_ODE_ref.npy')[0:data_size]
        k_data_re = np.load(file=r'../database/SDE/k_ODE_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        k_test = k_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        k_test_re = k_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)  ##201
        true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        # std_u = torch.std(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)
        mean_k = torch.mean(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        # std_k = torch.std(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        std_k = std_cal(torch.from_numpy(k_test_re)).type(torch.float).to(device)
        # calculate u reference solution

        u_means_1 = np.load("mix//u15k9//PI-GEA//1000//u_mean_error.npy")[-n:-1]
        k_means_1 = np.load("mix//u15k9//PI-GEA//1000//k_mean_error.npy")[-n:-1]
        u_stds_1 = np.load("mix//u15k9//PI-GEA//1000//u_std_error.npy")[-n:-1]
        k_stds_1 = np.load("mix//u15k9//PI-GEA//1000//k_std_error.npy")[-n:-1]

        u_means_2 = np.load("mix//u15k9//PI-WGAN//PI_WGAN_u_mean_error.npy")[-n:-1]  ##30
        k_means_2 = np.load("mix//u15k9//PI-WGAN//PI_WGAN_k_mean_error.npy")[-n:-1]
        u_stds_2 = np.load("mix//u15k9//PI-WGAN//PI_WGAN_u_std_error.npy")[-n:-1]
        k_stds_2 = np.load("mix//u15k9//PI-WGAN//PI_WGAN_k_std_error.npy")[-n:-1]

        u_means_3 = np.load("mix//u15k9//PI-VEGAN//u_mean_error.npy")[-n:-1]  ##30
        k_means_3 = np.load("mix//u15k9//PI-VEGAN//k_mean_error.npy")[-n:-1]
        u_stds_3 = np.load("mix//u15k9//PI-VEGAN//u_std_error.npy")[-n:-1]
        k_stds_3 = np.load("mix//u15k9//PI-VEGAN//k_std_error.npy")[-n:-1]

        u_means_4 = np.load("mix//u15k9//PI-GEA//2000//u_mean_error.npy")[-n:-1]  ##30
        k_means_4 = np.load("mix//u15k9//PI-GEA//2000//k_mean_error.npy")[-n:-1]
        u_stds_4 = np.load("mix//u15k9//PI-GEA//2000//u_std_error.npy")[-n:-1]
        k_stds_4 = np.load("mix//u15k9//PI-GEA//2000//k_std_error.npy")[-n:-1]

        u_means_5 = np.load("mix//u15k9//PI-VAE//vae_u_mean_error.npy")[-n:-1]  ##30
        k_means_5 = np.load("mix//u15k9//PI-VAE//vae_k_mean_error.npy")[-n:-1]
        u_stds_5 = np.load("mix//u15k9//PI-VAE//vae_u_std_error.npy")[-n:-1]
        k_stds_5 = np.load("mix//u15k9//PI-VAE//vae_k_std_error.npy")[-n:-1]


        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))
        k_mean_1 = torch.mean(torch.from_numpy(k_means_1), dim=0)
        k_mean_error_1 = torch.std(torch.from_numpy(k_means_1))
        k_std_1 = torch.mean(torch.from_numpy(k_stds_1), dim=0)
        k_std_error_1 = torch.std(torch.from_numpy(k_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))
        k_mean_2 = torch.mean(torch.from_numpy(k_means_2), dim=0)
        k_mean_error_2 = torch.std(torch.from_numpy(k_means_2))
        k_std_2 = torch.mean(torch.from_numpy(k_stds_2), dim=0)
        k_std_error_2 = torch.std(torch.from_numpy(k_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))
        k_mean_3 = torch.mean(torch.from_numpy(k_means_3), dim=0)
        k_mean_error_3 = torch.std(torch.from_numpy(k_means_3))
        k_std_3 = torch.mean(torch.from_numpy(k_stds_3), dim=0)
        k_std_error_3 = torch.std(torch.from_numpy(k_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))
        k_mean_4 = torch.mean(torch.from_numpy(k_means_4), dim=0)
        k_mean_error_4 = torch.std(torch.from_numpy(k_means_4))
        k_std_4 = torch.mean(torch.from_numpy(k_stds_4), dim=0)
        k_std_error_4 = torch.std(torch.from_numpy(k_stds_4))

        u_mean_5 = torch.mean(torch.from_numpy(u_means_5), dim=0)
        u_mean_error_5 = torch.std(torch.from_numpy(u_means_5))
        u_std_5 = torch.mean(torch.from_numpy(u_stds_5), dim=0)
        u_std_error_5 = torch.std(torch.from_numpy(u_stds_5))
        k_mean_5 = torch.mean(torch.from_numpy(k_means_5), dim=0)
        k_mean_error_5 = torch.std(torch.from_numpy(k_means_5))
        k_std_5 = torch.mean(torch.from_numpy(k_stds_5), dim=0)
        k_std_error_5 = torch.std(torch.from_numpy(k_stds_5))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
        k_mean_ref = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
        k_std_ref = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 30 * bar_width, 5 * bar_width)
        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_5, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_5, u_std_ref]
        k_mean = [k_mean_1, k_mean_2, k_mean_3, k_mean_4, k_mean_5, k_mean_ref]
        k_std = [k_std_1, k_std_2, k_std_3, k_std_4, k_std_5, k_std_ref]
        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, u_mean_error_5, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, u_std_error_5, 0]
        k_mean_err = [k_mean_error_1, k_mean_error_2, k_mean_error_3, k_mean_error_4, k_mean_error_5, 0]
        k_std_err = [k_std_error_1, k_std_error_2, k_std_error_3, k_std_error_4, k_std_error_5, 0]
        tick_label = ['PI-GEA-1', 'PI-WGAN', 'PI-VEGAN', 'PI-GEA-2', 'PI-VAE', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        plt.figure(figsize=(16, 6))
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, k_mean, bar_width, color=['orange'], yerr=k_mean_err, error_kw=error_params1, label='k mean')
        plt.bar(x + bar_width, k_std, bar_width, color=['pink'], yerr=k_std_err, error_kw=error_params2, label='k std')
        plt.bar(x + 2 * bar_width, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + 3 * bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 1.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('mix_error_u15k9.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    if type == 5:

        u_data = np.load(file=r'../database/SDE/u_ODE.npy')[0:data_size]
        k_data = np.load(file=r'../database/SDE/k_ODE.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_ODE_ref.npy')[0:data_size]
        k_data_re = np.load(file=r'../database/SDE/k_ODE_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        k_test = k_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        k_test_re = k_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)  ##201
        true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        # std_u = torch.std(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)
        mean_k = torch.mean(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        # std_k = torch.std(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        std_k = std_cal(torch.from_numpy(k_test_re)).type(torch.float).to(device)
        # calculate u reference solution

        u_means_1 = np.load("mix//u9k15//PI-GEA//1000//u_mean_error.npy")[-n:-1]
        k_means_1 = np.load("mix//u9k15//PI-GEA//1000//k_mean_error.npy")[-n:-1]
        u_stds_1 = np.load("mix//u9k15//PI-GEA//1000//u_std_error.npy")[-n:-1]
        k_stds_1 = np.load("mix//u9k15//PI-GEA//1000//k_std_error.npy")[-n:-1]

        u_means_2 = np.load("mix//u9k15//PI-WGAN//PI_WGAN_u_mean_error.npy")[-n:-1]  ##30
        k_means_2 = np.load("mix//u9k15//PI-WGAN//PI_WGAN_k_mean_error.npy")[-n:-1]
        u_stds_2 = np.load("mix//u9k15//PI-WGAN//PI_WGAN_u_std_error.npy")[-n:-1]
        k_stds_2 = np.load("mix//u9k15//PI-WGAN//PI_WGAN_k_std_error.npy")[-n:-1]

        u_means_3 = np.load("mix//u9k15//PI-VEGAN//u_mean_error.npy")[-n:-1]  ##30
        k_means_3 = np.load("mix//u9k15//PI-VEGAN//k_mean_error.npy")[-n:-1]
        u_stds_3 = np.load("mix//u9k15//PI-VEGAN//u_std_error.npy")[-n:-1]
        k_stds_3 = np.load("mix//u9k15//PI-VEGAN//k_std_error.npy")[-n:-1]

        u_means_4 = np.load("mix//u9k15//PI-GEA//2000//u_mean_error.npy")[-n:-1]  ##30
        k_means_4 = np.load("mix//u9k15//PI-GEA//2000//k_mean_error.npy")[-n:-1]
        u_stds_4 = np.load("mix//u9k15//PI-GEA//2000//u_std_error.npy")[-n:-1]
        k_stds_4 = np.load("mix//u9k15//PI-GEA//2000//k_std_error.npy")[-n:-1]

        u_means_5 = np.load("mix//u9k15//PI-VAE//vae_u_mean_error.npy")[-n:-1]  ##30
        k_means_5 = np.load("mix//u9k15//PI-VAE//vae_k_mean_error.npy")[-n:-1]
        u_stds_5 = np.load("mix//u9k15//PI-VAE//vae_u_std_error.npy")[-n:-1]
        k_stds_5 = np.load("mix//u9k15//PI-VAE//vae_k_std_error.npy")[-n:-1]


        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))
        k_mean_1 = torch.mean(torch.from_numpy(k_means_1), dim=0)
        k_mean_error_1 = torch.std(torch.from_numpy(k_means_1))
        k_std_1 = torch.mean(torch.from_numpy(k_stds_1), dim=0)
        k_std_error_1 = torch.std(torch.from_numpy(k_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))
        k_mean_2 = torch.mean(torch.from_numpy(k_means_2), dim=0)
        k_mean_error_2 = torch.std(torch.from_numpy(k_means_2))
        k_std_2 = torch.mean(torch.from_numpy(k_stds_2), dim=0)
        k_std_error_2 = torch.std(torch.from_numpy(k_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))
        k_mean_3 = torch.mean(torch.from_numpy(k_means_3), dim=0)
        k_mean_error_3 = torch.std(torch.from_numpy(k_means_3))
        k_std_3 = torch.mean(torch.from_numpy(k_stds_3), dim=0)
        k_std_error_3 = torch.std(torch.from_numpy(k_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))
        k_mean_4 = torch.mean(torch.from_numpy(k_means_4), dim=0)
        k_mean_error_4 = torch.std(torch.from_numpy(k_means_4))
        k_std_4 = torch.mean(torch.from_numpy(k_stds_4), dim=0)
        k_std_error_4 = torch.std(torch.from_numpy(k_stds_4))

        u_mean_5 = torch.mean(torch.from_numpy(u_means_5), dim=0)
        u_mean_error_5 = torch.std(torch.from_numpy(u_means_5))
        u_std_5 = torch.mean(torch.from_numpy(u_stds_5), dim=0)
        u_std_error_5 = torch.std(torch.from_numpy(u_stds_5))
        k_mean_5 = torch.mean(torch.from_numpy(k_means_5), dim=0)
        k_mean_error_5 = torch.std(torch.from_numpy(k_means_5))
        k_std_5 = torch.mean(torch.from_numpy(k_stds_5), dim=0)
        k_std_error_5 = torch.std(torch.from_numpy(k_stds_5))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
        k_mean_ref = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
        k_std_ref = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 30 * bar_width, 5 * bar_width)
        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_5, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_5, u_std_ref]
        k_mean = [k_mean_1, k_mean_2, k_mean_3, k_mean_4, k_mean_5, k_mean_ref]
        k_std = [k_std_1, k_std_2, k_std_3, k_std_4, k_std_5, k_std_ref]
        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, u_mean_error_5, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, u_std_error_5, 0]
        k_mean_err = [k_mean_error_1, k_mean_error_2, k_mean_error_3, k_mean_error_4, k_mean_error_5, 0]
        k_std_err = [k_std_error_1, k_std_error_2, k_std_error_3, k_std_error_4, k_std_error_5, 0]
        tick_label = ['PI-GEA-1', 'PI-WGAN', 'PI-VEGAN', 'PI-GEA-2', 'PI-VAE', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        plt.figure(figsize=(16, 6))
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, k_mean, bar_width, color=['orange'], yerr=k_mean_err, error_kw=error_params1, label='k mean')
        plt.bar(x + bar_width, k_std, bar_width, color=['pink'], yerr=k_std_err, error_kw=error_params2, label='k std')
        plt.bar(x + 2 * bar_width, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + 3 * bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 1.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('mix_error_u9k15.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    if type == 6:
        u_data = np.load(file=r'../database/SDE/u_0.08_5000.npy')[0:data_size]
        k_data = np.load(file=r'../database/SDE/k_0.08_5000.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_0.08_ref.npy')[0:data_size]
        k_data_re = np.load(file=r'../database/SDE/k_0.08_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        k_test = k_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        k_test_re = k_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)  ##201
        true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)
        mean_k = torch.mean(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        std_k = std_cal(torch.from_numpy(k_test_re)).type(torch.float).to(device)
        # calculate u reference solution


        u_means_1 = np.load("forward//a=0.08//PI-GEA//u_mean_error.npy")[-n:-1]##forward//a=0.08//
        k_means_1 = np.load("forward//a=0.08//PI-GEA//k_mean_error.npy")[-n:-1]
        u_stds_1 = np.load("forward//a=0.08//PI-GEA//u_std_error.npy")[-n:-1]
        k_stds_1 = np.load("forward//a=0.08//PI-GEA//k_std_error.npy")[-n:-1]##PI_WGAN_

        u_means_2 = np.load("forward//a=0.08//PI-WGAN//PI_WGAN_u_mean_error.npy")[-n:-1]##30
        k_means_2 = np.load("forward//a=0.08//PI-WGAN//PI_WGAN_k_mean_error.npy")[-n:-1]
        u_stds_2 = np.load("forward//a=0.08//PI-WGAN//PI_WGAN_u_std_error.npy")[-n:-1]
        k_stds_2 = np.load("forward//a=0.08//PI-WGAN//PI_WGAN_k_std_error.npy")[-n:-1]


        u_means_3 = np.load("forward//a=0.08//PI-VEGAN//VEGAN_u_mean_error.npy")[-n:-1]##30
        k_means_3 = np.load("forward//a=0.08//PI-VEGAN//VEGAN_k_mean_error.npy")[-n:-1]
        u_stds_3 = np.load("forward//a=0.08//PI-VEGAN//VEGAN_u_std_error.npy")[-n:-1]
        k_stds_3 = np.load("forward//a=0.08//PI-VEGAN//VEGAN_k_std_error.npy")[-n:-1]

        u_means_4 = np.load("forward//a=0.08//PI-VAE//vae_u_mean_error.npy")[-n:-1]##30
        k_means_4 = np.load("forward//a=0.08//PI-VAE//vae_k_mean_error.npy")[-n:-1]
        u_stds_4 = np.load("forward//a=0.08//PI-VAE//vae_u_std_error.npy")[-n:-1]
        k_stds_4 = np.load("forward//a=0.08//PI-VAE//vae_k_std_error.npy")[-n:-1]


        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))
        k_mean_1 = torch.mean(torch.from_numpy(k_means_1), dim=0)
        k_mean_error_1 = torch.std(torch.from_numpy(k_means_1))
        k_std_1 = torch.mean(torch.from_numpy(k_stds_1), dim=0)
        k_std_error_1 = torch.std(torch.from_numpy(k_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))
        k_mean_2 = torch.mean(torch.from_numpy(k_means_2), dim=0)
        k_mean_error_2 = torch.std(torch.from_numpy(k_means_2))
        k_std_2 = torch.mean(torch.from_numpy(k_stds_2), dim=0)
        k_std_error_2 = torch.std(torch.from_numpy(k_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))
        k_mean_3 = torch.mean(torch.from_numpy(k_means_3), dim=0)
        k_mean_error_3 = torch.std(torch.from_numpy(k_means_3))
        k_std_3 = torch.mean(torch.from_numpy(k_stds_3), dim=0)
        k_std_error_3 = torch.std(torch.from_numpy(k_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))
        k_mean_4 = torch.mean(torch.from_numpy(k_means_4), dim=0)
        k_mean_error_4 = torch.std(torch.from_numpy(k_means_4))
        k_std_4 = torch.mean(torch.from_numpy(k_stds_4), dim=0)
        k_std_error_4 = torch.std(torch.from_numpy(k_stds_4))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
        k_mean_ref = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
        k_std_ref = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 15 * bar_width, 3 * bar_width)

        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_ref]

        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, 0]
        tick_label = ['PI-GEA', 'PI-WGAN', 'PI-VEGAN', 'PI-VAE', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 0.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('a=0.08.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    if type == 7:
        u_data = np.load(file=r'../database/SDE/u_0.02_5000.npy')[0:data_size]
        k_data = np.load(file=r'../database/SDE/k_0.02_5000.npy')[0:data_size]
        u_data_re = np.load(file=r'../database/SDE/u_0.02_ref.npy')[0:data_size]
        k_data_re = np.load(file=r'../database/SDE/k_0.02_ref.npy')[0:data_size]
        # calculate ground true for comparison
        n_validate = 101  # number of validation points
        test_coor = np.floor(np.linspace(0, 1, n_validate) * 100).astype(int)
        u_test = u_data[:, test_coor]
        k_test = k_data[:, test_coor]
        u_test_re = u_data_re[:, test_coor]
        k_test_re = k_data_re[:, test_coor]
        true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
        true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
        true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)  ##201
        true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
        mean_u = torch.mean(torch.from_numpy(u_test_re), dim=0).type(torch.float).to(device)
        std_u = std_cal(torch.from_numpy(u_test_re)).type(torch.float).to(device)
        mean_k = torch.mean(torch.from_numpy(k_test_re), dim=0).type(torch.float).to(device)
        std_k = std_cal(torch.from_numpy(k_test_re)).type(torch.float).to(device)
        # calculate u reference solution


        u_means_1 = np.load("forward//a=0.02//PI-GEA//u_mean_error.npy")[-n:-1]##forward//a=0.08//
        k_means_1 = np.load("forward//a=0.02//PI-GEA//k_mean_error.npy")[-n:-1]
        u_stds_1 = np.load("forward//a=0.02//PI-GEA//u_std_error.npy")[-n:-1]
        k_stds_1 = np.load("forward//a=0.02//PI-GEA//k_std_error.npy")[-n:-1]##PI_WGAN_

        u_means_2 = np.load("forward//a=0.02//PI-WGAN//PI_WGAN_u_mean_error.npy")[-n:-1]##30
        k_means_2 = np.load("forward//a=0.02//PI-WGAN//PI_WGAN_k_mean_error.npy")[-n:-1]
        u_stds_2 = np.load("forward//a=0.02//PI-WGAN//PI_WGAN_u_std_error.npy")[-n:-1]
        k_stds_2 = np.load("forward//a=0.02//PI-WGAN//PI_WGAN_k_std_error.npy")[-n:-1]


        u_means_3 = np.load("forward//a=0.02//PI-VEGAN//VEGAN_u_mean_error.npy")[-n:-1]##30
        k_means_3 = np.load("forward//a=0.02//PI-VEGAN//VEGAN_k_mean_error.npy")[-n:-1]
        u_stds_3 = np.load("forward//a=0.02//PI-VEGAN//VEGAN_u_std_error.npy")[-n:-1]
        k_stds_3 = np.load("forward//a=0.02//PI-VEGAN//VEGAN_k_std_error.npy")[-n:-1]

        u_means_4 = np.load("forward//a=0.02//PI-VAE//vae_u_mean_error.npy")[-n:-1]##30
        k_means_4 = np.load("forward//a=0.02//PI-VAE//vae_k_mean_error.npy")[-n:-1]
        u_stds_4 = np.load("forward//a=0.02//PI-VAE//vae_u_std_error.npy")[-n:-1]
        k_stds_4 = np.load("forward//a=0.02//PI-VAE//vae_k_std_error.npy")[-n:-1]


        u_mean_1 = torch.mean(torch.from_numpy(u_means_1), dim=0)
        u_mean_error_1 = torch.std(torch.from_numpy(u_means_1))
        u_std_1 = torch.mean(torch.from_numpy(u_stds_1), dim=0)
        u_std_error_1 = torch.std(torch.from_numpy(u_stds_1))
        k_mean_1 = torch.mean(torch.from_numpy(k_means_1), dim=0)
        k_mean_error_1 = torch.std(torch.from_numpy(k_means_1))
        k_std_1 = torch.mean(torch.from_numpy(k_stds_1), dim=0)
        k_std_error_1 = torch.std(torch.from_numpy(k_stds_1))

        u_mean_2 = torch.mean(torch.from_numpy(u_means_2), dim=0)
        u_mean_error_2 = torch.std(torch.from_numpy(u_means_2))
        u_std_2 = torch.mean(torch.from_numpy(u_stds_2), dim=0)
        u_std_error_2 = torch.std(torch.from_numpy(u_stds_2))
        k_mean_2 = torch.mean(torch.from_numpy(k_means_2), dim=0)
        k_mean_error_2 = torch.std(torch.from_numpy(k_means_2))
        k_std_2 = torch.mean(torch.from_numpy(k_stds_2), dim=0)
        k_std_error_2 = torch.std(torch.from_numpy(k_stds_2))

        u_mean_3 = torch.mean(torch.from_numpy(u_means_3), dim=0)
        u_mean_error_3 = torch.std(torch.from_numpy(u_means_3))
        u_std_3 = torch.mean(torch.from_numpy(u_stds_3), dim=0)
        u_std_error_3 = torch.std(torch.from_numpy(u_stds_3))
        k_mean_3 = torch.mean(torch.from_numpy(k_means_3), dim=0)
        k_mean_error_3 = torch.std(torch.from_numpy(k_means_3))
        k_std_3 = torch.mean(torch.from_numpy(k_stds_3), dim=0)
        k_std_error_3 = torch.std(torch.from_numpy(k_stds_3))

        u_mean_4 = torch.mean(torch.from_numpy(u_means_4), dim=0)
        u_mean_error_4 = torch.std(torch.from_numpy(u_means_4))
        u_std_4 = torch.mean(torch.from_numpy(u_stds_4), dim=0)
        u_std_error_4 = torch.std(torch.from_numpy(u_stds_4))
        k_mean_4 = torch.mean(torch.from_numpy(k_means_4), dim=0)
        k_mean_error_4 = torch.std(torch.from_numpy(k_means_4))
        k_std_4 = torch.mean(torch.from_numpy(k_stds_4), dim=0)
        k_std_error_4 = torch.std(torch.from_numpy(k_stds_4))

        u_mean_ref = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
        u_std_ref = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
        k_mean_ref = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
        k_std_ref = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()

        bar_width = 0.5
        x = np.arange(0, 15 * bar_width, 3 * bar_width)

        # 数据集
        u_mean = [u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_ref]
        u_std = [u_std_1, u_std_2, u_std_3, u_std_4, u_std_ref]

        # 误差列表
        u_mean_err = [u_mean_error_1, u_mean_error_2, u_mean_error_3, u_mean_error_4, 0]
        u_std_err = [u_std_error_1, u_std_error_2, u_std_error_3, u_std_error_4, 0]
        tick_label = ['PI-GEA', 'PI-WGAN', 'PI-VEGAN', 'PI-VAE', 'Reference']

        error_params1 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        error_params2 = dict(elinewidth=1, ecolor='black', capsize=1)  # 设置误差标记参数
        # 绘制柱状图，设置误差标记以及柱状图标签
        plt.bar(x, u_mean, bar_width, color=['cyan'], yerr=u_mean_err, error_kw=error_params1,
                label='u mean')
        plt.bar(x + bar_width, u_std, bar_width, color=['gray'], yerr=u_std_err, error_kw=error_params2,
                label='u std')

        plt.xticks(x + 0.5 * bar_width, tick_label)  # 设置x轴的标签
        # 设置网格
        plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
        # 显示图例
        plt.legend()
        plt.savefig('a=0.02.eps', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

##可视化采样路径
def samples_plot(type):
    if type == "sp":
        flag_sensor = 6
        data_1 = np.load(file=r'../sp/square_exp/6sensor,l=1.npy')
        data_2 = np.load(file=r'../sp/square_exp/6sensor,l=0.5.npy')
        data_3 = np.load(file=r'../sp/square_exp/6sensor,l=0.2.npy')
        plt.figure(figsize=(20,4))


        plt.subplot(1, 3, 1)
        x = np.linspace(-1, 1, len(data_1[0, :]))
        sensor_position = np.linspace(-1, 1, flag_sensor)
        lower_bound = np.min(np.min(np.array(data_1)))
        upper_bound = np.max(np.max(np.array(data_1)))
        for i in range(20):
            plt.plot(x, data_1[i, :], linewidth = 1)
            for k in range(flag_sensor):
                plt.vlines(sensor_position[k], lower_bound, upper_bound, colors="k", linestyles="dashed")
        plt.title('$l$ = 1.0')


        plt.subplot(1, 3, 2)
        x = np.linspace(-1, 1, len(data_2[0, :]))
        sensor_position = np.linspace(-1, 1, flag_sensor)
        lower_bound = np.min(np.min(np.array(data_2)))
        upper_bound = np.max(np.max(np.array(data_2)))
        for i in range(20):
            plt.plot(x, data_2[i, :], linewidth=1)
            for k in range(flag_sensor):
                plt.vlines(sensor_position[k], lower_bound, upper_bound, colors="k", linestyles="dashed")
        plt.title('$l$ = 0.5')

        plt.subplot(1, 3, 3)
        x = np.linspace(-1, 1, len(data_3[0, :]))
        sensor_position = np.linspace(-1, 1, flag_sensor)
        lower_bound = np.min(np.min(np.array(data_3)))
        upper_bound = np.max(np.max(np.array(data_3)))
        for i in range(20):
            plt.plot(x, data_3[i, :], linewidth=1)
            for k in range(flag_sensor):
                plt.vlines(sensor_position[k], lower_bound, upper_bound, colors="k", linestyles="dashed")
        plt.title('$l$ = 0.2')

        plt.savefig(f'sp_sample.eps', bbox_inches='tight', dpi=300)
        plt.show()

    if type == "sde":
        data_1 = np.load(file=r'../database/SDE/k_ODE.npy')
        data_2 = np.load(file=r'../database/SDE/u_ODE.npy')
        data_3 = np.load(file=r'../database/SDE/f_ODE.npy')
        plt.figure(figsize=(20,4))


        plt.subplot(1, 3, 1)
        x = np.linspace(-1, 1, len(data_1[0, :]))
        for i in range(100):
            plt.plot(x, data_1[i, :], linewidth = 1)
        plt.grid(True, axis='x', linestyle='solid')
        plt.title('sample paths of $k(x; \omega)$')


        plt.subplot(1, 3, 2)
        x = np.linspace(-1, 1, len(data_2[0, :]))
        for i in range(100):
            plt.plot(x, data_2[i, :], linewidth=1)
        plt.grid(True, axis='x', linestyle='solid')
        plt.title('sample paths of $u(x; \omega)$')

        plt.subplot(1, 3, 3)
        x = np.linspace(-1, 1, len(data_3[0, :]))
        for i in range(100):
            plt.plot(x, data_3[i, :], linewidth=1)
        plt.grid(True, axis='x', linestyle='solid')
        plt.title('sample paths of $f(x; \omega)$')

        plt.savefig(f'sde_sample.eps', bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == '__main__':
    cl = 0.2
    data_size = 1000
    batch_val = 1000
    # latent_dim = 4
    n_blocks = 4
    width = 128
    setup_seed(4)
    # error(type=7)##type:1、正向问题对比 2、正向问题 3、逆问题 4、u15k9 5、u9k15  6、a = 0.08  7、a = 0.02
    # loss()
    # error_curve()
    # distribution()
    # memory_complexity()
    samples_plot(type = "sde")  ### type: "sp" or "sde"