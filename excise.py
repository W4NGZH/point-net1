import DataLoad
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

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

        return x

liver_dataset = DataLoad.AllDataset("D:/data/excise_data/data0")
dataloaders = DataLoader(liver_dataset, batch_size=1, drop_last=True, shuffle=False, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T_Net().to(device)

for x, y in dataloaders:

    inputs = torch.tensor(np.array(x)).reshape(14, 3, 1).float().to(device)
    labels = torch.tensor(np.array(y)).reshape(1, 3, 14).float().to(device)
    res = model(inputs)
    print(inputs)
    print('res:',res.shape)

