# -*- coding:utf-8 -*-
import numpy as np

data = []
with open("D:/data/excise_data/data1/0000.txt",'r') as f1:
    data_origin = f1.readlines()
    for data_str in data_origin:
        data_float = list(map(float,data_str.split()))
        data.append(data_float[:6])

data = np.asarray(data)
data_norm = data - data.mean(axis=0) / data.std(axis=0)

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit_center(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    x = data
    k_means = K_Means(k=2)
    k_means.fit_center(x)
    print(k_means.centers_)
    data_landslide = []
    for i in range(len(data_norm)):
        class_res = k_means.predict(data_norm[i])
        if class_res:
            data_landslide.append(data[i])

    with open("D:/data/excise_data/data_process/0000.txt",'w') as f2:
        for data_s in data_landslide:
            data_s = list(map(str,data_s))
            for s in data_s:
                f2.write(s)
                f2.write(' ')
            f2.write('\n')
