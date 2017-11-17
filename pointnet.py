from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


# STN1 transform on raw input data 
# spatial transformer network
class STN3d(nn.Module):
    def __init__(self, num_points):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv2d(1, 64, (3, 1))
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.view(batchsize, 1, 3, self.num_points)
        x = self.conv1(x)
        x = x.view(batchsize, 64, self.num_points)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # identity transform
        # bz x 9
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# STN2 64 x 64 transform
class Feats_STN3d(nn.Module):
    def __init__(self, num_points):    
        super(Feats_STN3d, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4096) # 64*64=4096

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # 64 x n -> 64 x n
        x = F.relu(self.bn2(self.conv2(x))) # 64 x n -> 128 x n
        x = F.relu(self.bn3(self.conv3(x))) # 128 x n -> 1024 x n
        x = self.mp1(x) # 1024 x n -> 1024 x 1
        x = x.view(-1, 1024) # 1 x 1024

        x = F.relu(self.bn4(self.fc1(x))) # 512
        x = F.relu(self.bn5(self.fc2(x))) # 256
        x = self.fc3(x) # 256 -> 256 x (64*64)
        # identity transform
        # bz x (64*64)
        iden = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).view(1,64*64).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 64, 64)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points, global_feat):
        super(PointNetfeat, self).__init__()
        self.stn1 = STN3d(num_points = num_points)
        self.stn2 = Feats_STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv2d(1, 64, (3, 1))
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points, return_indices=True)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        # regressing the space transforming parameters using STN1
        trans1 = self.stn1(x)
        x = x.transpose(2, 1) # 3 x n -> n x 3
        x = torch.bmm(x, trans1) # (n x 3) * (3 x 3)
        x = x.transpose(2, 1).contiguous() # n x 3 -> 3 x n
        # 3 x n -> 1 x 3 x n
        x = x.view(batchsize, 1, 3, self.num_points)
        # conv1 1 x 3 x n -> 1 x 64 x n
        x = self.conv1(x)
        # 1 x 64 x n -> 64 x n
        x = x.view(batchsize, 64, self.num_points)
        x = F.relu(self.bn1(x))
        # conv2 64 x n -> 64 x n
        x = F.relu(self.bn2(self.conv2(x)))
        # regressing the feature transforming parameters using STN2
        trans2 = self.stn2(x)
        x = x.transpose(2, 1) # 64 x n -> n x 64
        x = torch.bmm(x, trans2) # (n x 64) * (64 x 64)
        x = x.transpose(2, 1) # n x 64 -> 64 x n
        # conv3 64 x n -> 64 x n
        x = F.relu(self.bn3(self.conv3(x)))
        pointfeat = x
        # conv4 64 x n -> 128 x n
        x = F.relu(self.bn4(self.conv4(x)))
        # conv5 128 x n -> 1024 x n
        x = self.bn5(self.conv5(x))
        # pooling b x 1024 x n -> b x 1024 x 1
        x, pool_indices = self.mp1(x)
        # view b x 1024 x 1 -> b x 1024
        x = x.view(-1, 1024)
        if self.global_feat:    # using global feats for classification
            return x, trans1, trans2, pool_indices
        else:                   # for segmentation
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans1, trans2

# classification
class PointNetCls(nn.Module):
    def __init__(self, num_points, k):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
    def forward(self, x):
        x, trans1, trans2, pool_indices = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x) # bz x 40
        return F.log_softmax(x), trans1, trans2

class PointNetfeat_Encode(nn.Module):
    def __init__(self, num_points):
        super(PointNetfeat_Encode, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points, return_indices=True)
        # bn
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
    def forward(self, x, ):
        # conv1 b x 3 x n -> b x 64 x n
        x = F.relu(self.bn1(self.conv1(x)))
        # conv2 b x 64 x n -> b x 128 x n
        x = F.relu(self.bn2(self.conv2(x)))
        # conv3 b x 128 x n -> b x 1024 x n
        x = F.relu(self.bn3(self.conv3(x)))
        # pooling b x 1024 x n -> b x 1024 x 1
        x, pool_indices = self.mp1(x)
        # view b x 1024 x 1 -> b x 1024
        x = x.squeeze()
        return x, pool_indices

class PointNetfeat_Decode(nn.Module):
    def __init__(self, num_points):
        super(PointNetfeat_Decode, self).__init__()
        self.num_points = num_points
        self.mup1 = torch.nn.MaxUnpool1d(num_points)
        self.dconv3 = torch.nn.ConvTranspose1d(1024, 128, 1)
        self.dconv2 = torch.nn.ConvTranspose1d(128, 64, 1)
        self.dconv1 = torch.nn.ConvTranspose1d(64, 3, 1)
        # bn
        self.bn3 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(3)
    def forward(self, x, pool_indices):
        x = x.unsqueeze(-1)
        # unpooling b x 1024 x 1 -> b x 1024 x n
        x = self.mup1(x, pool_indices)
        # dconv3 b x 1024 x n -> b x 128 x n
        x = F.relu(self.bn3(self.dconv3(x)))
        # dconv2 b x 128 x n -> b x 64 x n
        x = F.relu(self.bn2(self.dconv2(x)))
        # dconv1 b x 64 x n -> b x 3 x n
        x = F.relu(self.bn1(self.dconv1(x)))
        return x

# PointNet VAE
class PointNetVAE(nn.Module):
    def __init__(self, num_points, use_pointnet_feat):
        super(PointNetVAE, self).__init__()
        self.num_points = num_points
        self.use_pointnet_feat = use_pointnet_feat
        # encoder
        if self.use_pointnet_feat:
            self.enc_feat = PointNetfeat(num_points, global_feat=True)
        else:
            self.enc_feat = PointNetfeat_Encode(num_points)
        self.enc_fc1 = nn.Linear(1024, 512)
        self.enc_fc2_1 = nn.Linear(512, 256)
        self.enc_fc2_2 = nn.Linear(512, 256)
        # decoder
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_fc1 = nn.Linear(512, 1024)
        self.dec_feat = PointNetfeat_Decode(num_points)
        # bn
        self.enc_bn512 = nn.BatchNorm1d(512)
        self.enc_bn256_1 = nn.BatchNorm1d(256)
        self.enc_bn256_2 = nn.BatchNorm1d(256)
        self.dec_bn512 = nn.BatchNorm1d(512)
        self.dec_bn1024 = nn.BatchNorm1d(1024)
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def forward(self, x):
        # encode
        if self.use_pointnet_feat:
            x, trans1, trans2, pool_indices = self.enc_feat(x)
        else:
            x, pool_indices = self.enc_feat(x)
        x = F.relu(self.enc_bn512(self.enc_fc1(x)))
        # reparametrize
        mu = F.relu(self.enc_bn256_1(self.enc_fc2_1(x)))
        logvar = F.relu(self.enc_bn256_2(self.enc_fc2_2(x)))
        z = self.reparametrize(mu, logvar)
        # decode
        x = F.relu(self.dec_bn512(self.dec_fc2(z)))
        x = F.relu(self.dec_bn1024(self.dec_fc1(x)))
        x = self.dec_feat(x, pool_indices)
        points = x
        if self.use_pointnet_feat:
            return z, points, trans1, trans2, mu, logvar
        else:
            return z, points, mu, logvar

# regular segmentation
class PointNetDenseCls(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans1, trans2 = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans1, trans2


if __name__ == '__main__':
    n = 2500
    c = 40
    sim_data = Variable(torch.rand(32, 3, n))

    trans = STN3d(num_points=n)
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(num_points=n, global_feat=True)
    out, trans1, trans2, pool_indices = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(num_points=n, global_feat=False)
    out, trans1, trans2 = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_points=n, k = c)
    out, trans1, trans2 = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_points=n, k = c)
    out, trans1, trans2 = seg(sim_data)
    print('seg', out.size())

    vae = PointNetVAE(num_points=n)
    feature, points, trans1, trans2 = vae(sim_data)
    print('vae', feature.size(), points.size())
