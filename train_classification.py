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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default = 12, help='input batch size')
parser.add_argument('--num_points', type=int, default = 2500, help='input batch size')
parser.add_argument('--workers', type=int, default = 4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default = 25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default = 'cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--cuda', type=bool, default = True, help='run with cuda')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print('train:', len(dataset), 'test:', len(test_dataset))
num_classes = len(dataset.classes)
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

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if opt.cuda:
    classifier.cuda()

num_batch = len(dataset) / opt.batchSize + 1

for epoch in range(opt.nepoch):
    i = 0
    # train
    for points, target in dataloader:
        i += 1
        bsize = len(target)
        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1)
        if opt.cuda:
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, _, trans2 = classifier(points)
        # get loss
        eye64 = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).view(1,64*64).repeat(bsize,1)
        if trans2.is_cuda:
            eye64 = eye64.cuda()
        trans2 = trans2 - eye64
        loss = F.nll_loss(pred, target) + torch.norm(trans2, 2)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        log_string('[%d: %d/%d] train loss: %f accuracy: %f' %
                    (epoch, i, num_batch, loss.data[0], correct/float(bsize)))
    # test per epoch
    j, loss, correct = 0, 0, 0
    for points, target in testdataloader:
        j += 1
        bsize = len(target)
        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1)
        if opt.cuda:
            points, target = points.cuda(), target.cuda()
        pred, _, _ = classifier(points)
        loss += F.nll_loss(pred, target).data[0]
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum() / float(bsize)
    log_string('[%d: %d/%d] test loss: %f accuracy: %f' %
                (epoch, i, num_batch, loss/j, correct/j))
    # save per epoch
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
