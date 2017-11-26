import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
From Paper <Deep Learning with Sets and Point Clouds>
'''

'''
B x K x N -> B x K' x N
    B batch size
    K,K' in/out feature dimension
    N set size
'''
class permute_invariant_layer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(permute_invariant_layer, self).__init__()
        self.conv = nn.Conv1d(in_feat, out_feat, 1)
    def forward(self, x):
        batchsize = x.size()[0]
        featsize = x.size()[1]
        setsize = x.size()[2]
        x = x.clone()
        for i in range(batchsize):
            maxv, _ = torch.max(x[i], dim=1)
            maxvn = maxv.repeat(setsize).view(setsize, featsize).transpose(0,1)
            x[i] = x[i] - maxvn
        return self.conv(x)

class PointSetCls(nn.Module):
    def __init__(self, num_points, k):
        super(PointSetCls, self).__init__()
        self.num_points = num_points
        self.pi1 = permute_invariant_layer(3, 64)
        self.pi2 = permute_invariant_layer(64, 128)
        self.pi3 = permute_invariant_layer(128, 256)
        self.mp = nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k)
    def forward(self, x):
        x = F.tanh(self.pi1(x))
        x = F.tanh(self.pi2(x))
        x = F.tanh(self.pi3(x))
        x = F.dropout(self.mp(x), p = 0.5)
        x = x.view(-1, 256)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return F.log_softmax(x)

if __name__ == '__main__':
    n = 2500
    c = 40
    sim_data = Variable(torch.rand(32, 3, n))
    print('test')

    trans = permute_invariant_layer(3, 64)
    out = trans(sim_data)
    print('permute_invariant_layer', out.size())

    cls = PointSetCls(num_points = n, k = c)
    out = cls(sim_data)
    print('PointSetCls', out.size())
