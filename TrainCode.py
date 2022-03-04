import numpy as np
import torch
from torch.optim import *
import random
import DataLoad
import PointNet
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# x_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
# y_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
# y_transforms = transforms.ToTensor()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 20
epochs = 1e4
lr_rate = 1e-3
step = 500
gamma_size = 0.99
model_path = 'D:/model/point'

setup_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet.PointNetPartSeg(3).to(device)
batch_size = 1
criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma_size)

def train():
    train_dataset = DataLoad.AllDataset("D:/data/excise_data/data0")
    # liver_dataset = DataLoad.AllDataset("D:/data/excise_data", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,shuffle=False, num_workers=0)
    for epoch in range(int(epochs)):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        epoch_loss = 0
        step = 0
        for x, y in dataloaders:
            step += 1
            inputs = torch.tensor(np.asarray(x)).reshape(1, 3, -1).float().to(device)
            labels = torch.tensor(np.asarray(y)).reshape(1, -1, 3).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        print('学习率：', optimizer.state_dict()['param_groups'][0]['lr'])

        # if epoch > 10:
        #     torch.save(model.state_dict(), model_path + '/' + str(epoch) + 'model1.pth')
    torch.save(model.state_dict(), '{0}/model1.pth'.format(model_path))


if __name__ == '__main__':
    train()