__all__  = ['MMD_loss', 'SinkhornDistance', 'VAE', 'Generator', 'Discriminator', 'PIVAE_SDE',\
             'PIGAN_Generator', 'PIGAN_Discriminator', 'PIVAE_SDE_multigroup', 'PIVAE_SPDE']
import torch
import torch.nn as nn
from torch.autograd import Variable

class plainBlock(nn.Module):
    'description'
    # blocks of plain neural network

    def __init__(self, width):
        super(plainBlock, self).__init__()
        self.fc1 = nn.Linear(width,width)
        # self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        out = self.fc1(x)
        # out = self.dropout(out)
        out = torch.tanh(out)
        return out
class plainBlock_res(nn.Module):
    'description'
    # blocks blocks of NN with shortcuts(ResNN) 

    def __init__(self, width):
        super(plainBlock_res, self).__init__()
        self.fc1 = nn.Linear(width,width)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.tanh(out)
        return out + x
# new network
class Encoder(nn.Module):
    def __init__(self, latent_dim, u_data_dim, k_data_dim, f_data_dim, n_blocks, width, device):
        super(Encoder, self).__init__()

        self.device = device

        self.u_data_dim = u_data_dim
        self.k_data_dim = k_data_dim
        self.f_data_dim = f_data_dim
        self.latent_dim = latent_dim

        self.encoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.encoder = nn.Sequential(nn.Linear(u_data_dim + k_data_dim + f_data_dim, width), self.encoder_blocks).to(device)
        self.encoder_mu = nn.Sequential(nn.Linear(width, latent_dim), nn.Tanh()).to(device)
        self.encoder_var = nn.Sequential(nn.Linear(width, latent_dim), nn.Sigmoid()).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std).to(self.device)
        return mu + std * eps

    def forward(self, u, k, f):  ##u([1000, 2])  k([1000, 13])  f([1000, 21])
        com = torch.cat((u, k, f), dim=1)
        out = self.encoder(com)
        mu = self.encoder_mu(out)
        logvar = self.encoder_var(out)
        z = self.reparameterize(mu, logvar)  ##(1000, 4)
        return z, mu, logvar


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                nn.init.constant_(m.bias, val=0)
                # m.weight.data.normal_(0, 0.02)
                # m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, lat_dim, udata_dim, kdata_dim, fdata_dim, width, n_blocks, device):
        super(Generator, self).__init__()
        self.device = device

        self.u_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.u_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.u_blocks, nn.Linear(width, 1)).to(device)

        self.k_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.k_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.k_blocks, nn.Linear(width, 1)).to(device)

        self.udata_dim = udata_dim
        self.kdata_dim = kdata_dim
        self.fdata_dim = fdata_dim
        self.lat_dim = lat_dim

    def combine_xz(self, x, z):
        x_new = x.view(-1, 1).to(self.device)
        z_new = torch.repeat_interleave(z, x.size(1), dim=0).to(self.device)  # .view(-1,self.latent_dim)
        return torch.cat((x_new, z_new), 1)

    def reconstruct(self, z, ucoor, kcoor):
        x_u = self.combine_xz(ucoor, z)
        urecon = self.u_gen(x_u).view(-1, ucoor.size(1))
        x_k = self.combine_xz(kcoor, z)
        krecon = self.k_gen(x_k).view(-1, kcoor.size(1))
        return urecon, krecon

    def f_recontruct(self, z, fcoor):
        x = Variable(fcoor.view(-1, 1).type(torch.FloatTensor), requires_grad=True).to(self.device)
        z_uk = torch.repeat_interleave(z, fcoor.size(1), dim=0)

        x_PDE = torch.cat((x, z_uk), 1)
        u_PDE = self.u_gen(x_PDE)
        k_PDE = self.k_gen(x_PDE)

        # calculate derivative
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(self.device),
                                       create_graph=True, only_inputs=True)[0]
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        f_recon = -0.1 * (k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx).view(-1, fcoor.size(1))
        return f_recon

    def forward(self, z, ucoor, kcoor, fcoor):
        urecon, krecon = self.reconstruct(z, ucoor, kcoor)
        f_recon = self.f_recontruct(z, fcoor)
        return urecon, krecon, f_recon



    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                nn.init.constant_(m.bias, val=0)
                # m.weight.data.normal_(0, 0.02)
                # m.bias.data.zero_()

class VEGAN_Encoder(nn.Module):
    def __init__(self, latent_dim, u_data_dim, k_data_dim, f_data_dim, n_blocks, width, device):
        super(VEGAN_Encoder, self).__init__()

        self.device = device

        self.u_data_dim = u_data_dim
        self.k_data_dim = k_data_dim
        self.f_data_dim = f_data_dim
        self.latent_dim = latent_dim

        self.encoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.encoder = nn.Sequential(nn.Linear(u_data_dim + k_data_dim + f_data_dim, width), self.encoder_blocks)#.to(device)
        self.encoder_mu = nn.Sequential(nn.Linear(width, latent_dim), nn.Tanh())#.to(device)
        self.encoder_var = nn.Sequential(nn.Linear(width, latent_dim), nn.Sigmoid())#.to(device)

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std)#.to(self.device)
        return mu + std * eps

    # def forward(self, u, k, f):  ##u([1000, 2])  k([1000, 13])  f([1000, 21])
    #     com = torch.cat((u, k, f), dim=1)
    #     out = self.encoder(com)
    #     mu = self.encoder_mu(out)
    #     logvar = self.encoder_var(out)
    #     z = self.reparameterize(mu, logvar)  ##(1000, 4)
    #     return z, mu, logvar
    def forward(self, z):  ##u([1000, 2])  k([1000, 13])  f([1000, 21])
        out = self.encoder(z)
        mu = self.encoder_mu(out)
        logvar = self.encoder_var(out)
        z = self.reparameterize(mu, logvar)  ##(1000, 4)
        return z

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                nn.init.constant_(m.bias, val=0)

class VEGAN_Decoder(nn.Module):
    def __init__(self, latent_dim, n_blocks, width, device):
        super(VEGAN_Decoder, self).__init__()

        self.device = device
        self.latent_dim = latent_dim

        self.u_decoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.u_decoder = nn.Sequential(nn.Linear(latent_dim + 1, width), nn.Tanh(), self.u_decoder_blocks,
                                       nn.Linear(width, 1))#.to(device)

        self.k_decoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.k_decoder = nn.Sequential(nn.Linear(latent_dim + 1, width), nn.Tanh(), self.k_decoder_blocks,
                                       nn.Linear(width, 1))#.to(device)

    def combine_xz(self, x, z):
        x_new = x.view(-1, 1)
        z_new = torch.repeat_interleave(z, x.size(-1), dim=0)  #.view(-1,self.latent_dim)
        return torch.cat((x_new, z_new), 1)

    def funval_cal(self, z, u_coor, k_coor):
        u_recon = self.u_decoder(self.combine_xz(u_coor, z)).view(-1, u_coor.size(1)).to(self.device)
        k_recon = self.k_decoder(self.combine_xz(k_coor, z)).view(-1, k_coor.size(1)).to(self.device)
        return u_recon, k_recon

    def PDE_check(self, z, f_coor):  ##f_coor(1000,21)
        x = Variable(f_coor.view(-1, 1).type(torch.FloatTensor), requires_grad=True).to(self.device)
        z_uk = torch.repeat_interleave(z, f_coor.size(1), dim=0)

        val_u = torch.cat((x, z_uk), 1)
        val_k = torch.cat((x, z_uk), 1)

        u_PDE = self.u_decoder(val_u)
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(self.device),
                                       create_graph=True, only_inputs=True)[0]
        k_PDE = self.k_decoder(val_k)
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        f_recon = -0.1 * (k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx).view(-1, f_coor.size(1))
        return f_recon

    # def forward(self, z, u_coor, k_coor, f_coor):
    #     u_recon, k_recon = self.funval_cal(z, u_coor, k_coor)
    #     f_recon = self.PDE_check(z, f_coor)
    #     return u_recon, k_recon, f_recon
    def forward(self, z):
        f_recon = self.u_decoder(z)
        return f_recon
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                nn.init.constant_(m.bias, val=0)

class VEGAN_Discriminator(nn.Module):
    def __init__(self, in_dim, width, device):
        super(VEGAN_Discriminator, self).__init__()
        self.device = device

        self.fc1 = nn.Linear(in_dim, width)#.to(device)
        self.fc2 = nn.Linear(width, width)#.to(device)
        self.fc3 = nn.Linear(width, width)#.to(device)
        self.fc4 = nn.Linear(width, width)#.to(device)
        self.fc5 = nn.Linear(width, 1)#.to(device)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))#.to(self.device)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                nn.init.constant_(m.bias, val=0)


# network of PIGANs
class PIGAN_Generator(nn.Module):
    def __init__(self, lat_dim, udata_dim, kdata_dim, fdata_dim, width, n_blocks, device):
        super(PIGAN_Generator, self).__init__()
        self.device = device

        self.u_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.u_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.u_blocks, nn.Linear(width, 1))#.to(device)

        self.k_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.k_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.k_blocks, nn.Linear(width, 1))#.to(device)

        self.udata_dim = udata_dim
        self.kdata_dim = kdata_dim
        self.fdata_dim = fdata_dim
        self.lat_dim = lat_dim

    def combine_xz(self, x, z):
        x_new = x.view(-1, 1).to(self.device)
        z_new = torch.repeat_interleave(z, x.size(1), dim=0).to(self.device)  # .view(-1,self.latent_dim)
        return torch.cat((x_new, z_new), 1)

    def reconstruct(self, z, ucoor, kcoor):
        x_u = self.combine_xz(ucoor, z)
        urecon = self.u_gen(x_u).view(-1, ucoor.size(1))
        x_k = self.combine_xz(kcoor, z)
        krecon = self.k_gen(x_k).view(-1, kcoor.size(1))
        return urecon, krecon

    def f_recontruct(self, z, fcoor):
        x = Variable(fcoor.view(-1, 1).type(torch.FloatTensor), requires_grad=True).to(self.device)
        z_uk = torch.repeat_interleave(z, fcoor.size(1), dim=0)

        x_PDE = torch.cat((x, z_uk), 1)
        u_PDE = self.u_gen(x_PDE)
        k_PDE = self.k_gen(x_PDE)

        # calculate derivative
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(self.device),
                                       create_graph=True, only_inputs=True)[0]
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        f_recon = -0.1 * (k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx).view(-1, fcoor.size(1))
        return f_recon

    # def forward(self, z, ucoor, kcoor, fcoor):
    #     urecon, krecon = self.reconstruct(z, ucoor, kcoor)
    #     f_recon = self.f_recontruct(z, fcoor)
    #     return urecon, krecon, f_recon
    def forward(self, z):
        f_recon = self.u_gen(z)
        return f_recon


class PIGAN_Discriminator(nn.Module):
    def __init__(self, in_dim, width, device):
        super(PIGAN_Discriminator, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(in_dim, width).to(device)
        self.fc2 = nn.Linear(width, width).to(device)
        self.fc3 = nn.Linear(width, width).to(device)
        self.fc4 = nn.Linear(width, width).to(device)
        self.fc5 = nn.Linear(width, 1).to(device)

    def forward(self, x):
        x = torch.tanh(self.fc1(x)).to(self.device)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class PIVAE_SDE(nn.Module):
    def __init__(self, latent_dim, u_data_dim, k_data_dim, f_data_dim, n_blocks, width, device):
        super(PIVAE_SDE, self).__init__()

        self.device = device

        self.u_data_dim = u_data_dim
        self.k_data_dim = k_data_dim
        self.f_data_dim = f_data_dim
        self.latent_dim = latent_dim

        self.encoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.encoder = nn.Sequential(nn.Linear(u_data_dim + k_data_dim + f_data_dim, width), self.encoder_blocks)
        self.encoder_mu = nn.Sequential(nn.Linear(width, latent_dim), nn.Tanh())
        self.encoder_var = nn.Sequential(nn.Linear(width, latent_dim), nn.Sigmoid())

        self.u_decoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.u_decoder = nn.Sequential(nn.Linear(latent_dim + 1, width), nn.Tanh(), self.u_decoder_blocks,
                                       nn.Linear(width, 1))

        self.k_decoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.k_decoder = nn.Sequential(nn.Linear(latent_dim + 1, width), nn.Tanh(), self.k_decoder_blocks,
                                       nn.Linear(width, 1))

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std).to(self.device)
        return mu + std * eps

    def combine_xz(self, x, z):  # x是一个n行矩阵，z是一个列向量
        x_new = x.view(-1, 1)
        z_new = torch.repeat_interleave(z, x.size(-1), dim=0)  # .view(-1,self.latent_dim)
        return torch.cat((x_new, z_new), 1)

    def encode(self, u, k, f):
        com = torch.cat((u, k, f), dim=1)
        out = self.encoder(com)
        mu = self.encoder_mu(out)
        logvar = self.encoder_var(out)
        Z = self.reparameterize(mu, logvar)

        return Z

    def funval_cal(self, z, u_coor, k_coor):
        u_recon = self.u_decoder(self.combine_xz(u_coor, z)).view(-1, u_coor.size(1))
        k_recon = self.k_decoder(self.combine_xz(k_coor, z)).view(-1, k_coor.size(1))

        return u_recon, k_recon

    def PDE_check(self, z, f_coor, device):
        x = Variable(f_coor.view(-1, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        z_uk = torch.repeat_interleave(z, f_coor.size(1), dim=0)

        val_u = torch.cat((x, z_uk), 1)
        val_k = torch.cat((x, z_uk), 1)

        u_PDE = self.u_decoder(val_u)
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(device),
                                      create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(device),
                                       create_graph=True, only_inputs=True)[0]
        k_PDE = self.k_decoder(val_k)
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(device),
                                      create_graph=True, only_inputs=True)[0]
        f_recon = -0.1 * (k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx).view(-1, f_coor.size(1))

        return f_recon

    def forward(self, u, k, f, u_coor, k_coor, f_coor):
        Z = self.encode(u, k, f)
        u_recon, k_recon = self.funval_cal(Z, u_coor, k_coor)
        f_recon = self.PDE_check(Z, f_coor, self.device)

        return u_recon, k_recon, f_recon, Z

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