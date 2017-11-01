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
from pointnet import PointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default = 16, help='input batch size')

opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0' , train = False, classification = True)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle = False)


classifier = PointNetCls(k = len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

for points, target in testdataloader:
    points, target = Variable(points), Variable(target[:,0])
    points = points.transpose(2,1)
    points, target = points.cuda(), target.cuda()
    pred, _ = classifier(points)
    loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('loss: %f accuracy: %f' %(loss.data[0], correct/float(opt.batchSize)))