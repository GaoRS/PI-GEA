
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

class plainBlock(nn.Module):
    'description'
    # blocks of plain neural network

    def __init__(self, width):
        super(plainBlock, self).__init__()
        self.fc1 = nn.Linear(width ,width)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.tanh(out)
        return out


class Encoder(nn.Module):
    def __init__(self, latent_dim, f_data_dim, n_blocks, width, device):
        super(Encoder, self).__init__()

        self.device = device
        self.f_data_dim = f_data_dim
        self.latent_dim = latent_dim

        self.encoder_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.encoder = nn.Sequential(nn.Linear(f_data_dim, width), self.encoder_blocks).to(device)
        self.encoder_mu = nn.Sequential(nn.Linear(width, latent_dim), nn.Tanh()).to(device)
        self.encoder_var = nn.Sequential(nn.Linear(width, latent_dim), nn.Sigmoid()).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std).to(self.device)
        return mu + std * eps

    def forward(self, f):  ##u([1000, 2])  k([1000, 13])  f([1000, 21])
        out = self.encoder(f)
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
    def __init__(self, lat_dim, fdata_dim, width, n_blocks, device):
        super(Generator, self).__init__()
        self.device = device

        self.blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.blocks, nn.Linear(width, 1)).to(device)

        self.fdata_dim = fdata_dim
        self.lat_dim = lat_dim

    def combine_xz(self, x, z):
        x_new = x.view(-1, 1).to(self.device)
        z_new = torch.repeat_interleave(z, x.size(1), dim=0).to(self.device)  # .view(-1,self.latent_dim)
        return torch.cat((x_new, z_new), 1)

    def forward(self, z, coor):
        z_ = self.combine_xz(coor, z)
        f_recon = self.gen(z_).view(-1, coor.size(1))
        return f_recon

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                nn.init.constant_(m.bias, val=0)
                # m.weight.data.normal_(0, 0.02)
                # m.bias.data.zero_()




class VEGAN(nn.Module):
    def __init__(self, latent_dim, f_data_dim, n_blocks, width, device):
        super(VEGAN, self).__init__()

        self.device = device

        self.f_data_dim = f_data_dim
        self.latent_dim = latent_dim

        self.encoder_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])).to(device)
        self.encoder = nn.Sequential(nn.Linear(f_data_dim, width), self.encoder_blocks).to(device)
        self.encoder_mu = nn.Sequential( nn.Linear(width, latent_dim),  nn.Tanh()).to(device)
        self.encoder_var = nn.Sequential( nn.Linear(width, latent_dim),  nn.Sigmoid()).to(device)

        self.decoder_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])).to(device)
        self.decoder = nn.Sequential(nn.Linear(latent_dim +1, width),  nn.Tanh(), self.decoder_blocks, nn.Linear(width, 1)).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std).to(self.device)
        return mu + std * eps

    def combine_xz(self, x, z):    # x是一个n行矩阵，z是一个列向量
        x_new = x.view(-1 ,1)
        z_new = torch.repeat_interleave(z, x.size(-1), dim=0) # .view(-1,self.latent_dim)
        return torch.cat((x_new ,z_new) ,1)

    def encode(self, f):
        out = self.encoder(f)
        mu = self.encoder_mu(out)
        logvar = self.encoder_var(out)
        Z = self.reparameterize(mu, logvar)

        return Z, mu, logvar

    def decode(self, z, f_coor):
        f_recon = self.decoder(self.combine_xz(f_coor, z)).view(-1 ,f_coor.size(1))

        return f_recon

class VEGAN_Discriminator(nn.Module):
    def __init__(self, in_dim, width=64):
        super(VEGAN_Discriminator, self).__init__()

        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

class PI_WGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(5, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def combination(self, x, z):
        x_new = x.view(-1, 1)
        z_new = torch.repeat_interleave(z, x.size(1), dim=0)
        return torch.cat((x_new, z_new), 1)

    def forward(self, z, F_loc):
        z_ = self.combination(F_loc, z)
        F_recon = self.model(z_).view(-1, F_loc.size(1))
        return F_recon

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, val=0)

class PI_WGAN_Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(PI_WGAN_Discriminator, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, val=0)



