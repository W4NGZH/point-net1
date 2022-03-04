import numpy as np
import torch
from torch.utils.data import DataLoader
import DataLoad
import PointNet
from sklearn.metrics import mean_squared_error


model_file = 'D:/model/point/model1.pth'

test_dataset = DataLoad.AllDataset("D:/data/excise_data/data0")
dataloaders = DataLoader(test_dataset, batch_size=1, drop_last=True,shuffle=False, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet.PointNetPartSeg(3).to(device)
model.load_state_dict(torch.load(model_file))
model.eval()

mse = 0
for x,y in dataloaders:
    inputs = torch.tensor(np.asarray(x)).reshape(1, 3, -1).float().to(device)
    labels = np.asarray(y)
    ouputs = model(inputs)
    data_out = [yy.cpu().detach().numpy() for yy in ouputs][0]
    err = mean_squared_error(data_out,labels)
    mse += err

print(mse)