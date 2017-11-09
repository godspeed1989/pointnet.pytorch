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
from pointnet import PointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default = 10, help='input batch size')

opt = parser.parse_args()
print (opt)

dataset = 'modelnet40_pcl'
if dataset == 'partnno':
    test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',
                                classification = True, train = False, npoints = 2500)
    classifier = PointNetCls(k = len(test_dataset.classes), num_points = 2500)
elif dataset == 'modelnet40_pcl':
    test_dataset = Modelnet40_PCL_Dataset(data_dir = 'modelnet40_ply_hdf5_2048', train = False, npoints = 2048)
    classifier = PointNetCls(k = 40, num_points = 2048)
else:
    assert 0
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle = False)

classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

# F.nll_loss()
criterion = nn.CrossEntropyLoss()

for points, target in testdataloader:
    bsize = len(target)
    if dataset == 'partnno':
        points, target = Variable(points), Variable(target[:,0])
    elif dataset == 'modelnet40_pcl':
        points, target = Variable(points), Variable(target[:])
    points = points.transpose(2,1)
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = criterion(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('loss: %f accuracy: %f' %(loss.data[0], correct/float(bsize)))
