import numpy as np

data = []
with open("D:/data/excise_data/data1/0000.txt",'r') as f1:
    data_origin = f1.readlines()
    for data_str in data_origin:
        data_float = list(map(float,data_str.split()))
        data.append(data_float[:6])

data = np.asarray(data)
data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
print(data_norm.mean(axis=0),data_norm.std(axis=0))
print(data[0])
print(len(data),len(data_norm))
