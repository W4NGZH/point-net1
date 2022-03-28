import numpy as np

data = []
with open("D:/data/excise_data/data1/0000.txt",'r') as f1:
    data_origin = f1.readlines()
    for data_str in data_origin:
        data_float = list(map(float,data_str.split()))
        data.append(data_float[:6])

data = np.asarray(data)
data_norm = (data - data.mean(axis=0)) / data.std(axis=0)

class KMeans(object):
    def __init__(self, n_cluster, epochs=100):
        self.n_cluster = n_cluster
        self.epochs = epochs
        pass
    def init_centers(self, X):
        idx = np.random.randint(len(X), size=(self.n_cluster,))
        centers = X[idx,:]
        return centers

    def calculate_distance(self,arr1,arr2):
        # L2 distance.
        distance = np.mean(np.sqrt((arr1-arr2)**2))
        return distance
    def update_centers(self, X):
        predict_class = self.predict(X)
        # update centers
        centers = self.centers
        for ct in range(len(centers)):
            idx, = np.where(predict_class == ct)

            samples = X[idx, :]
            assert len(samples)>0
            centers[ct] = np.mean(samples,axis=0)
        self.centers = centers
        return self.centers

    def fit(self, X, y=None):
        self.centers = self.init_centers(X)
        for epoch in range(self.epochs):
            self.centers = self.update_centers(X)
        return self.centers
    def predict(self,X):
        predict_class = np.zeros(shape=(len(X),))
        centers = self.centers
        for n_sample,arr in enumerate(X):
            min_distance = float("inf")
            p_class = 0
            for ct in range(len(centers)):
                distance = self.calculate_distance(arr,centers[ct])
                if distance < min_distance:
                    min_distance = distance
                    p_class = ct
            predict_class[n_sample] = p_class
        return predict_class
    def score(self, X):
        pass


X = data_norm
print(X.shape)
kmeans = KMeans(2)
centers = kmeans.fit(X)
print(centers)

class1 = kmeans.predict(X)

data_landslide = []
noise1 = []

for i in range(len(class1)):
    if class1[i] == 0:
        noise1.append(data[i])
    # elif class_res == 1:
    #     noise2.append(data[i])
    else:
        data_landslide.append(data[i])

print(len(noise1),len(data_landslide))

with open("D:/data/excise_data/kmeans/landslide.pts", 'w') as f2:
    for data_s in data_landslide:
        data_s = list(map(str, data_s))
        for s in data_s:
            f2.write(s)
            f2.write(' ')
        f2.write('\n')
with open("D:/data/excise_data/kmeans/noise1.pts", 'w') as f2:
    for data_s in noise1:
        data_s = list(map(str, data_s))
        for s in data_s:
            f2.write(s)
            f2.write(' ')
        f2.write('\n')