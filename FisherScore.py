import pandas as pd
import numpy as np


def binary_fisher_score(sample,label):

    if len(sample) != len(label):
        print('Sample does not match label')
        exit()
    df1 = pd.DataFrame(sample)
    df2 = pd.DataFrame(label, columns=['label'])
    data = pd.concat([df1, df2], axis=1)  # 合并成为一个dataframe

    data0 = data[data.label == 0]#对标签分类，分成包含0和1的两个dataframe
    data1 = data[data.label == 1]
    n = len(label)#标签长度
    n1 = sum(label)#1类标签的个数
    n0 = n - n1#0类标签的个数
    lst = []#用于返回的列表
    features_list = list(data.columns)[:-1]
    for feature in features_list:

        # 算关于data0
        m0_feature_mean = data0[feature].mean()  # 0类标签在第m维上的均值
        # 0类在第m维上的sw
        m0_SW=sum((data0[feature] -m0_feature_mean )**2)
        # 算关于data1
        m1_feature_mean = data1[feature].mean()  # 1类标签在第m维上的均值
        # 1类在第m维上的sw
        m1_SW=sum((data1[feature] -m1_feature_mean )**2)
        # 算关于data
        m_all_feature_mean = data[feature].mean()  # 所有类标签在第m维上的均值

        m0_SB = n0 / n * (m0_feature_mean - m_all_feature_mean) ** 2
        m1_SB = n1 / n * (m1_feature_mean - m_all_feature_mean) ** 2
        #计算SB
        m_SB = m1_SB + m0_SB
        #计算SW
        m_SW = (m0_SW + m1_SW) / n
        if m_SW == 0:
            # 0/0类型也是返回nan
            m_fisher_score = np.nan
        else:
            # 计算Fisher score
            m_fisher_score = m_SB / m_SW
        #Fisher score值添加进列表
        lst.append(m_fisher_score)

    return lst

data_feature = []
data_class = []
with open("D:/data/excise_data/927-txt/500_class.txt",'r') as f1:
    data_origin1 = f1.readlines()
    for data_str1 in data_origin1:
        data_float1 = list(map(float,data_str1.split()))
        cla = data_float1.pop()
        data_class.append(int(cla))
        data_feature.append(data_float1)

print(binary_fisher_score(data_feature,data_class))





#
# from scipy.sparse import *
# from sklearn.metrics.pairwise import pairwise_distances
#
#
# def construct_W(X, **kwargs):
#     if 'metric' not in kwargs.keys():
#         kwargs['metric'] = 'cosine'
#
#     # default neighbor mode is 'knn' and default neighbor size is 5
#     if 'neighbor_mode' not in kwargs.keys():
#         kwargs['neighbor_mode'] = 'knn'
#     if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
#         kwargs['k'] = 5
#     if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
#         kwargs['k'] = 5
#     if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
#         print ('Warning: label is required in the supervised neighborMode!!!')
#         exit(0)
#
#     # default weight mode is 'binary', default t in heat kernel mode is 1
#     if 'weight_mode' not in kwargs.keys():
#         kwargs['weight_mode'] = 'binary'
#     if kwargs['weight_mode'] == 'heat_kernel':
#         if kwargs['metric'] != 'euclidean':
#             kwargs['metric'] = 'euclidean'
#         if 't' not in kwargs.keys():
#             kwargs['t'] = 1
#     elif kwargs['weight_mode'] == 'cosine':
#         if kwargs['metric'] != 'cosine':
#             kwargs['metric'] = 'cosine'
#
#     # default fisher_score and reliefF mode are 'false'
#     if 'fisher_score' not in kwargs.keys():
#         kwargs['fisher_score'] = False
#     if 'reliefF' not in kwargs.keys():
#         kwargs['reliefF'] = False
#
#     n_samples, n_features = np.shape(X)
#
#     # choose 'knn' neighbor mode
#     if kwargs['neighbor_mode'] == 'knn':
#         k = kwargs['k']
#         if kwargs['weight_mode'] == 'binary':
#             if kwargs['metric'] == 'euclidean':
#                 # compute pairwise euclidean distances
#                 D = pairwise_distances(X)
#                 D **= 2
#                 # sort the distance matrix D in ascending order
#                 dump = np.sort(D, axis=1)
#                 idx = np.argsort(D, axis=1)
#                 # choose the k-nearest neighbors for each instance
#                 idx_new = idx[:, 0:k+1]
#                 G = np.zeros((n_samples*(k+1), 3))
#                 G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
#                 G[:, 1] = np.ravel(idx_new, order='F')
#                 G[:, 2] = 1
#                 # build the sparse affinity matrix W
#                 W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#                 bigger = np.transpose(W) > W
#                 W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#                 return W
#
#             elif kwargs['metric'] == 'cosine':
#                 # normalize the data first
#                 X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
#                 for i in range(n_samples):
#                     X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
#                 # compute pairwise cosine distances
#                 D_cosine = np.dot(X, np.transpose(X))
#                 # sort the distance matrix D in descending order
#                 dump = np.sort(-D_cosine, axis=1)
#                 idx = np.argsort(-D_cosine, axis=1)
#                 idx_new = idx[:, 0:k+1]
#                 G = np.zeros((n_samples*(k+1), 3))
#                 G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
#                 G[:, 1] = np.ravel(idx_new, order='F')
#                 G[:, 2] = 1
#                 # build the sparse affinity matrix W
#                 W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#                 bigger = np.transpose(W) > W
#                 W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#                 return W
#
#         elif kwargs['weight_mode'] == 'heat_kernel':
#             t = kwargs['t']
#             # compute pairwise euclidean distances
#             D = pairwise_distances(X)
#             D **= 2
#             # sort the distance matrix D in ascending order
#             dump = np.sort(D, axis=1)
#             idx = np.argsort(D, axis=1)
#             idx_new = idx[:, 0:k+1]
#             dump_new = dump[:, 0:k+1]
#             # compute the pairwise heat kernel distances
#             dump_heat_kernel = np.exp(-dump_new/(2*t*t))
#             G = np.zeros((n_samples*(k+1), 3))
#             G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
#             G[:, 1] = np.ravel(idx_new, order='F')
#             G[:, 2] = np.ravel(dump_heat_kernel, order='F')
#             # build the sparse affinity matrix W
#             W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#             bigger = np.transpose(W) > W
#             W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#             return W
#
#         elif kwargs['weight_mode'] == 'cosine':
#             # normalize the data first
#             X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
#             for i in range(n_samples):
#                     X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
#             # compute pairwise cosine distances
#             D_cosine = np.dot(X, np.transpose(X))
#             # sort the distance matrix D in ascending order
#             dump = np.sort(-D_cosine, axis=1)
#             idx = np.argsort(-D_cosine, axis=1)
#             idx_new = idx[:, 0:k+1]
#             dump_new = -dump[:, 0:k+1]
#             G = np.zeros((n_samples*(k+1), 3))
#             G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
#             G[:, 1] = np.ravel(idx_new, order='F')
#             G[:, 2] = np.ravel(dump_new, order='F')
#             # build the sparse affinity matrix W
#             W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#             bigger = np.transpose(W) > W
#             W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#             return W
#
#     # choose supervised neighborMode
#     elif kwargs['neighbor_mode'] == 'supervised':
#         k = kwargs['k']
#         # get true labels and the number of classes
#         y = kwargs['y']
#         label = np.unique(y)
#         n_classes = np.unique(y).size
#         # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
#         if kwargs['fisher_score'] is True:
#             W = lil_matrix((n_samples, n_samples))
#             for i in range(n_classes):
#                 class_idx = (y == label[i])
#                 class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
#                 W[class_idx_all] = 1.0/np.sum(np.sum(class_idx))
#             return W
#
#         # construct the weight matrix W in a reliefF way, NH(x) and NM(x,y) denotes a set of k nearest
#         # points to x with the same class as x, a different class (the class y), respectively. W_ij = 1 if i = j;
#         # W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y)
#         if kwargs['reliefF'] is True:
#             # when xj in NH(xi)
#             G = np.zeros((n_samples*(k+1), 3))
#             id_now = 0
#             for i in range(n_classes):
#                 class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
#                 D = pairwise_distances(X[class_idx, :])
#                 D **= 2
#                 idx = np.argsort(D, axis=1)
#                 idx_new = idx[:, 0:k+1]
#                 n_smp_class = (class_idx[idx_new[:]]).size
#                 if len(class_idx) <= k:
#                     k = len(class_idx) - 1
#                 G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
#                 G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
#                 G[id_now:n_smp_class+id_now, 2] = 1.0/k
#                 id_now += n_smp_class
#             W1 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#             # when i = j, W_ij = 1
#             for i in range(n_samples):
#                 W1[i, i] = 1
#             # when x_j in NM(x_i, y)
#             G = np.zeros((n_samples*k*(n_classes - 1), 3))
#             id_now = 0
#             for i in range(n_classes):
#                 class_idx1 = np.column_stack(np.where(y == label[i]))[:, 0]
#                 X1 = X[class_idx1, :]
#                 for j in range(n_classes):
#                     if label[j] != label[i]:
#                         class_idx2 = np.column_stack(np.where(y == label[j]))[:, 0]
#                         X2 = X[class_idx2, :]
#                         D = pairwise_distances(X1, X2)
#                         idx = np.argsort(D, axis=1)
#                         idx_new = idx[:, 0:k]
#                         n_smp_class = len(class_idx1)*k
#                         G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx1, (k, 1)).reshape(-1)
#                         G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx2[idx_new[:]], order='F')
#                         G[id_now:n_smp_class+id_now, 2] = -1.0/((n_classes-1)*k)
#                         id_now += n_smp_class
#             W2 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#             bigger = np.transpose(W2) > W2
#             W2 = W2 - W2.multiply(bigger) + np.transpose(W2).multiply(bigger)
#             W = W1 + W2
#             return W
#
#         if kwargs['weight_mode'] == 'binary':
#             if kwargs['metric'] == 'euclidean':
#                 G = np.zeros((n_samples*(k+1), 3))
#                 id_now = 0
#                 for i in range(n_classes):
#                     class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
#                     # compute pairwise euclidean distances for instances in class i
#                     D = pairwise_distances(X[class_idx, :])
#                     D **= 2
#                     # sort the distance matrix D in ascending order for instances in class i
#                     idx = np.argsort(D, axis=1)
#                     idx_new = idx[:, 0:k+1]
#                     n_smp_class = len(class_idx)*(k+1)
#                     G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
#                     G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
#                     G[id_now:n_smp_class+id_now, 2] = 1
#                     id_now += n_smp_class
#                 # build the sparse affinity matrix W
#                 W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#                 bigger = np.transpose(W) > W
#                 W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#                 return W
#
#             if kwargs['metric'] == 'cosine':
#                 # normalize the data first
#                 X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
#                 for i in range(n_samples):
#                     X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
#                 G = np.zeros((n_samples*(k+1), 3))
#                 id_now = 0
#                 for i in range(n_classes):
#                     class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
#                     # compute pairwise cosine distances for instances in class i
#                     D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
#                     # sort the distance matrix D in descending order for instances in class i
#                     idx = np.argsort(-D_cosine, axis=1)
#                     idx_new = idx[:, 0:k+1]
#                     n_smp_class = len(class_idx)*(k+1)
#                     G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
#                     G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
#                     G[id_now:n_smp_class+id_now, 2] = 1
#                     id_now += n_smp_class
#                 # build the sparse affinity matrix W
#                 W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#                 bigger = np.transpose(W) > W
#                 W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#                 return W
#
#         elif kwargs['weight_mode'] == 'heat_kernel':
#             G = np.zeros((n_samples*(k+1), 3))
#             id_now = 0
#             for i in range(n_classes):
#                 class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
#                 # compute pairwise cosine distances for instances in class i
#                 D = pairwise_distances(X[class_idx, :])
#                 D **= 2
#                 # sort the distance matrix D in ascending order for instances in class i
#                 dump = np.sort(D, axis=1)
#                 idx = np.argsort(D, axis=1)
#                 idx_new = idx[:, 0:k+1]
#                 dump_new = dump[:, 0:k+1]
#                 t = kwargs['t']
#                 # compute pairwise heat kernel distances for instances in class i
#                 dump_heat_kernel = np.exp(-dump_new/(2*t*t))
#                 n_smp_class = len(class_idx)*(k+1)
#                 G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
#                 G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
#                 G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_heat_kernel, order='F')
#                 id_now += n_smp_class
#             # build the sparse affinity matrix W
#             W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#             bigger = np.transpose(W) > W
#             W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#             return W
#
#         elif kwargs['weight_mode'] == 'cosine':
#             # normalize the data first
#             X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
#             for i in range(n_samples):
#                 X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
#             G = np.zeros((n_samples*(k+1), 3))
#             id_now = 0
#             for i in range(n_classes):
#                 class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
#                 # compute pairwise cosine distances for instances in class i
#                 D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
#                 # sort the distance matrix D in descending order for instances in class i
#                 dump = np.sort(-D_cosine, axis=1)
#                 idx = np.argsort(-D_cosine, axis=1)
#                 idx_new = idx[:, 0:k+1]
#                 dump_new = -dump[:, 0:k+1]
#                 n_smp_class = len(class_idx)*(k+1)
#                 G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
#                 G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
#                 G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_new, order='F')
#                 id_now += n_smp_class
#             # build the sparse affinity matrix W
#             W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
#             bigger = np.transpose(W) > W
#             W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
#             return W
#
#
# def fisher_score(X, y):
#
#     kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
#     W = construct_W(X, **kwargs)
#
#     # build the diagonal D matrix from affinity matrix W
#     D = np.array(W.sum(axis=1))
#     L = W
#     tmp = np.dot(np.transpose(D), X)
#     D = diags(np.transpose(D), [0])
#     Xt = np.transpose(X)
#     t1 = np.transpose(np.dot(Xt, D.todense()))
#     t2 = np.transpose(np.dot(Xt, L.todense()))
#     # compute the numerator of Lr
#     D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
#     # compute the denominator of Lr
#     L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
#     # avoid the denominator of Lr to be 0
#     D_prime[D_prime < 1e-12] = 10000
#     lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
#
#     # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
#     score = 1.0/lap_score - 1
#     return np.transpose(score)
#
#
# def feature_ranking(score):
#     """
#     Rank features in descending order according to fisher score, the larger the fisher score, the more important the
#     feature is
#     """
#     idx = np.argsort(score, 0)
#     return idx[::-1]