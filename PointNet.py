import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch .nn.functional as F


class T_Net(nn.Module):
    def __init__(self):
        super(T_Net,self).__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self,x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1,9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)  # 输出为Batch*3*3的张量
        return x

class PointNetEncoder(nn.Module):
    def __init__(self,global_feat=True):
        super(PointNetEncoder, self).__init__()
        self.Tnet = T_Net()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.relu = nn.ReLU()

    def forward(self,x):
        n_pts = x.size()[2]
        trans = self.Tnet.forward(x)
        x = x.transpose(2,1)
        x = torch.bmm(x,trans)
        x = x.transpose(2,1)
        x = self.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x_skip = self.conv2(x)
        x = self.relu(self.bn2(x_skip))
        x = self.bn3(self.conv3(x))
        x = torch.max(x,2,keepdim=True)[0]
        x = x.view(-1,1024)
        if self.global_feat:
            return x,trans
        else:
            x = x.view(-1,1024,1).repeat(1,1,n_pts)
            return torch.cat([x,pointfeat],1),trans

class PointNetPartSeg(nn.Module):
    def __init__(self, num_class):
        super(PointNetPartSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x,trans


