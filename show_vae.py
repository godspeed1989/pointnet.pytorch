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
from ply_file_utils import write_ply

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default = 10, help='input batch size')
parser.add_argument('--outf', type=str, default = 'vae',  help='output folder')

opt = parser.parse_args()
print (opt)

if not os.path.exists(opt.outf):
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

dataset = 'modelnet40_pcl'
if dataset == 'partnno':
    test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',
                                classification = True, train = False, npoints = 2500)
    autoencoder = PointNetVAE(num_points = 2500)
elif dataset == 'modelnet40_pcl':
    test_dataset = Modelnet40_PCL_Dataset(data_dir = 'modelnet40_ply_hdf5_2048', train = False, npoints = 2048)
    autoencoder = PointNetVAE(num_points = 2048)
else:
    assert 0
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle = False)

autoencoder.cuda()
autoencoder.load_state_dict(torch.load(opt.model))
autoencoder.eval()

i = 0
for points, target in testdataloader:
    bsize = len(target)
    if dataset == 'partnno':
        points = Variable(points)
    elif dataset == 'modelnet40_pcl':
        points = Variable(points)
    points = points.transpose(2, 1)
    points = points.cuda()
    feature, out_points, trans1, trans2, mu, logvar = autoencoder(points)
    print('%d' % (i))
    i += 1
    if i % 50 == 0:
        for j in range(0, bsize, 5):
            pc = points[j].cpu().data.numpy().copy().transpose()
            pc_out = out_points[j].cpu().data.numpy().copy().transpose()
            pc_out += 2.0
            pc = np.vstack((pc, pc_out))
            fname = os.path.join(opt.outf, '%d_%d.ply' % (i, j))
            write_ply(pc, fname)
