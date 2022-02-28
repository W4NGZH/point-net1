import os
import torch.utils.data as data
import numpy as np

def make_dataset(root):
    paths = []
    n = len(os.listdir(root))
    for i in range(n):
        path = os.path.join(root, "%04d.txt" % i)
        paths.append(path)
    return paths

class AllDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        inPuts = make_dataset(root)
        self.inPuts = inPuts
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.inPuts[index]
        data_x,data_y = [],[]
        with open(x_path, 'r') as f:
            data_origin = f.readlines()
            for data_str in data_origin:
                data_float = list(map(float, data_str.split()))
                data_x.append(data_float[:3])
                data_y.append(data_float[:3])
        # data_x,data_y = np.array(data_x).T,np.array(data_y).T
        if self.transform is not None:
            data_x = self.transform(data_x)
        if self.target_transform is not None:
            data_y = self.target_transform(data_y)
        return data_x, data_y

    def __len__(self):
        return len(self.inPuts)



