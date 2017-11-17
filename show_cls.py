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
import h5py

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default = 5, help='input batch size')
parser.add_argument('--dump', type=str, default = 'feature.h5', help='feature dump file')

opt = parser.parse_args()
print (opt)

dataset = 'SHREC'
if dataset == 'partnno':
    num_classes = len(test_dataset.classes)
    test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',
                                classification = True, train = False, npoints = 2500)
    classifier = PointNetCls(k = num_classes, num_points = 2500)
elif dataset == 'modelnet40_pcl':
    num_classes = 40
    test_dataset = Modelnet40_PCL_Dataset(data_dir = 'modelnet40_ply_hdf5_2048', train = False, npoints = 2048)
    classifier = PointNetCls(k = num_classes, num_points = 2048)
elif dataset == 'SHREC':
    num_classes = 55
    test_dataset = Modelnet40_PCL_Dataset(data_dir = 'shrec2017_4096', train = False, npoints = 4096)
    classifier = PointNetCls(k = num_classes, num_points = 4096)
else:
    assert 0

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle = False)

classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

# F.nll_loss()
criterion = nn.CrossEntropyLoss()

dump_feature_file = h5py.File(opt.dump, 'w')
dump_feature_file.clear()
f1 = dump_feature_file.create_dataset('pred', shape=[len(test_dataset), num_classes], dtype=float)
f2 = dump_feature_file.create_dataset('feature', shape=[len(test_dataset), 1024], dtype=float)
idx = 0

for points, target in testdataloader:
    bsize = len(target)
    if dataset == 'partnno':
        points, target = Variable(points), Variable(target[:,0])
    elif dataset == 'modelnet40_pcl' or dataset == 'SHREC':
        points, target = Variable(points), Variable(target[:])
    points = points.transpose(2,1)
    points, target = points.cuda(), target.cuda()
    pred, _, _, feature = classifier(points)
    loss = criterion(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('loss: %f accuracy: %f' %(loss.data[0], correct/float(bsize)))
    # dump feature to file
    pred = pred.data.cpu().numpy()
    feature = feature.data.cpu().numpy()
    for i in range(bsize):
        f1[idx + i] = pred[i]
        f2[idx + i] = feature[i]
    idx += bsize

dump_feature_file.close()
