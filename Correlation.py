# -*- coding:utf-8 -*-
import numpy as np
import collections
from scipy.stats import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data = []
# data_class = []
#
# with open("D:/data/excise_data/927-txt/500_landslide.txt",'r') as f1:
#     data_origin1 = f1.readlines()
#     for data_str1 in data_origin1:
#         data_float1 = list(map(float,data_str1.split()))
#         data.append(data_float1)
#
# with open("D:/data/excise_data/927-txt/500.txt",'r') as f2:
#     data_origin2 = f2.readlines()
#     for data_str2 in data_origin2:
#         data_float2 = list(map(float,data_str2.split()))
#         data_class.append(data_float2)
#
# DataLenth = len(data_class)
#
# ClassDict = collections.defaultdict(dict)
# for i in range(DataLenth):
#     ClassDict[data_class[i][0]][data_class[i][1]] = 0
# for j in range(len(data)):
#     ClassDict[data[j][0]][data[j][1]] = 1
#
# for k in range(DataLenth):
#     data_class[k].append(ClassDict[data_class[k][0]][data_class[k][1]])
#
# with open("D:/data/excise_data/927-txt/500_class.txt",'w') as f_write:
#     for data_s in data_class:
#         data_s = list(map(str, data_s))
#         for s in data_s:
#             f_write.write(s)
#             f_write.write(' ')
#         f_write.write('\n')
#
# ClassNumpy = np.asarray(data_class).T
#
# df = pd.DataFrame({'x':ClassNumpy[0],'y':ClassNumpy[1],'z':ClassNumpy[2],'r':ClassNumpy[3],'g':ClassNumpy[4],'b':ClassNumpy[5],'flag':ClassNumpy[6],'num':ClassNumpy[7],'class':ClassNumpy[8]})
# df.to_csv("D:/data/excise_data/927-txt/500_class.csv")


#皮尔森相关系数
AllData = pd.read_csv("D:/data/excise_data/927-txt/500_class.csv")

plt.figure(figsize = (10, 5))
sns.pairplot(vars=['x'],hue='class',markers=['+',"s"],data=AllData)
# sns.scatterplot(x='x',y='y',hue='class',size='class',sizes=(30, 30),data=AllData)
# ax.figure.set_size_inches(12,6)
plt.show()
# 计算全部变量的相关性矩阵
print(AllData.corr()['class'])
r, p_value = stats.pearsonr(AllData['r'], AllData['class'])
print('相关系数为{:.3f},p值为{:.5f}'.format(r, p_value))
# 绘制相关系数的热力图
r_pearson = AllData.corr()
sns.heatmap(data=r_pearson, cmap="YlGnBu")  # cmap设置色系
# plt.show()

# plt.plot(AllData['x'],AllData['class'])
# plt.show()






