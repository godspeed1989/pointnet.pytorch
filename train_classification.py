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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default = 12, help='input batch size')
parser.add_argument('--num_points', type=int, default = 2500, help='input batch size')
parser.add_argument('--workers', type=int, default = 4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default = 50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default = 'cls',  help='output folder')
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
num_classes = len(train_dataset.classes)
print('classes', num_classes)

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

classifier = PointNetCls(k = num_classes, num_points = opt.num_points)
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
# F.nll_loss()
criterion = nn.CrossEntropyLoss()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

if opt.cuda:
    classifier.cuda()
    criterion.cuda()

num_batch = len(train_dataset) / opt.batchSize + 1

se = opt.start_epoch
for epoch in range(opt.nepoch):
    i = 0
    # train
    for points, target in traindataloader:
        i += 1
        bsize = len(target)
        if dataset == 'partnno':
            points, target = Variable(points), Variable(target[:,0])
        elif dataset == 'modelnet40_pcl':
            points, target = Variable(points), Variable(target[:])
        points = points.transpose(2,1)
        if opt.cuda:
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, _, trans2 = classifier(points)
        # get loss
        eye64 = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).repeat(bsize,1)
        if trans2.is_cuda:
            eye64 = eye64.cuda()
        trans2 = trans2 - eye64
        loss = criterion(pred, target) + torch.norm(trans2, 2)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        log_string('[%d: %d/%d] train loss: %f accuracy: %f' %
                    (se+epoch, i, num_batch, loss.data[0], correct/float(bsize)))
    # test per epoch
    j, loss, correct = 0, 0, 0
    for points, target in testdataloader:
        j += 1
        bsize = len(target)
        if dataset == 'partnno':
            points, target = Variable(points), Variable(target[:,0])
        elif dataset == 'modelnet40_pcl':
            points, target = Variable(points), Variable(target[:])
        points = points.transpose(2,1)
        if opt.cuda:
            points, target = points.cuda(), target.cuda()
        pred, _, _ = classifier(points)
        loss += F.nll_loss(pred, target).data[0]
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum() / float(bsize)
    log_string('[%d: %d/%d] test loss: %f accuracy: %f' %
                (se+epoch, i, num_batch, loss/j, correct/j))
    # save per epoch
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, se+epoch))
