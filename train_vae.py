from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets import PartDataset
from modelnet40_pcl_datasets import Modelnet40_PCL_Dataset
from pointnet import PointNetVAE
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default = 8, help='input batch size')
parser.add_argument('--num_points', type=int, default = None, help='input batch size')
parser.add_argument('--workers', type=int, default = 4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default = 250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default = 'vae',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--cuda', type=bool, default = True, help='run with cuda')
parser.add_argument('--start_epoch', type=int, default = 0, help='start epoch index')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = 'modelnet40_pcl'
if dataset == 'partnno':
    opt.num_points = 2500
    train_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',
                                classification = True, train = True, npoints = opt.num_points)
    test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',
                               classification = True, train = False, npoints = opt.num_points)
elif dataset == 'modelnet40_pcl':
    opt.num_points = 2048
    train_dataset = Modelnet40_PCL_Dataset(data_dir = 'modelnet40_ply_hdf5_2048',
                                           train = True, npoints = opt.num_points)
    test_dataset = Modelnet40_PCL_Dataset(data_dir = 'modelnet40_ply_hdf5_2048',
                                          train = False, npoints = opt.num_points)
else:
    assert 0

traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

print('train:', len(train_dataset), 'test:', len(test_dataset))

if not os.path.exists(opt.outf):
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
LOG_FOUT = open(os.path.join(opt.outf, 'log_train.txt'), 'w')
LOG_FOUT.write(str(opt)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

autoencoder = PointNetVAE(num_points = opt.num_points)
optimizer = optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.9)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

if opt.cuda:
    autoencoder.cuda()

def prepare_one_batch_input(points, target):
    if dataset == 'partnno':
        points, target = Variable(points), Variable(target[:,0])
    elif dataset == 'modelnet40_pcl':
        points, target = Variable(points), Variable(target[:])
    # b x np x 3 -> b x 3 x np
    points = points.transpose(2,1)
    if opt.cuda:
        return points.cuda(), target.cuda()
    return points, target
def get_trans_loss(trans2):
    eye64 = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).repeat(bsize, 1)
    eye64 = eye64.view(bsize, 64, 64)
    regression_weight = Variable(torch.FloatTensor([0.001]))
    if trans2.is_cuda:
        eye64 = eye64.cuda()
        regression_weight = regression_weight.cuda()
    trans2 = torch.bmm(trans2, trans2.transpose(2, 1))
    trans2 = trans2 - eye64
    trans_loss = torch.mul(torch.norm(trans2, 2), regression_weight)
    return trans_loss
def get_pc_loss(out_points, points, trans1):
    out_points = out_points.clone()
    trans1 = trans1.clone()
    for i in range(trans1.size()[0]):
        trans1[i] = torch.inverse(trans1[i])
    out_points = out_points.transpose(2, 1) # 3 x n -> n x 3
    out_points = torch.bmm(out_points, trans1)
    out_points = out_points.transpose(2, 1).contiguous() # n x 3 -> 3 x n
    #
    pc_loss = torch.max(torch.abs(out_points - points))
    return pc_loss
def get_KL_loss(mu, logvar):
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return KLD

num_batch = len(train_dataset) / opt.batchSize + 1
se = opt.start_epoch
for epoch in range(opt.nepoch):
    i = 0
    # train
    for points, target in traindataloader:
        i += 1
        bsize = len(target)
        points, target = prepare_one_batch_input(points, target)
        optimizer.zero_grad()
        # forward
        feature, out_points, trans1, trans2, mu, logvar = autoencoder(points)
        # get loss
        trloss = get_trans_loss(trans2)
        pcloss = get_pc_loss(out_points, points, trans1)
        klloss = get_KL_loss(mu, logvar)
        loss = trloss + pcloss + klloss
        # backward
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_string('[%d: %d/%d] train loss: %f (%f %f %f)' %
                   (se+epoch, i, num_batch, loss.data[0],
                   trloss.data[0], pcloss.data[0], klloss.data[0]))
    # test per epoch
    j, loss = 0, 0
    for points, target in testdataloader:
        j += 1
        bsize = len(target)
        points, target = prepare_one_batch_input(points, target)
        # forward
        feature, out_points, trans1, trans2, mu, logvar = autoencoder(points)
        # get loss
        trloss = get_trans_loss(trans2)
        pcloss = get_pc_loss(out_points, points, trans1)
        klloss = get_KL_loss(mu, logvar)
        all_loss = trloss + pcloss + klloss
        loss += all_loss.data[0]
    # print test result
    log_string('[%d: %d/%d] test loss: %f' % (se+epoch, i, num_batch, loss/j))
    # save per epoch
    torch.save(autoencoder.state_dict(), '%s/vae_model_%d.pth' % (opt.outf, se+epoch))
print('Done.')
