import numpy as np
from matplotlib.pylab import plt

class KMeans:
    def __init__(self, n_clusters, init = 'kmeans++', max_iter = 1000, random_state = 0):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = None

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        if self.init == 'kmeans++':
            c = X[np.random.randint(0, len(X), 1)]
            c = c.reshape((1, -1))
            # distance to the closest center
            distance = np.sum(np.square(X - c), axis=1)
            centers = [c]
            for i in range(1, self.n_clusters):
                # random sampling according to the distance
                c_idx = np.random.choice(len(X), 1,
                                         p=distance / distance.sum())
                c = X[c_idx]
                c = c.reshape((1, -1))
                centers.append(c)
                # new distance to the new center
                new_distance = np.sum(np.square(X - c), axis=1)
                # update the minimum distance
                distance = np.minimum(distance, new_distance)
            centers = np.concatenate(centers, axis=0)
        else:
            # random sampling
            centers = X[np.random.choice(len(X), self.n_clusters)]

        old_labels = np.empty(N)
        for step in range(self.max_iter):
            dist_matrix = []

            # update the distance to each center
            for i in range(self.n_clusters):
                distance = np.sum(np.square(X - centers[i, :]), axis=1)
                dist_matrix.append(distance[:, None])
            dist_matrix = np.concatenate(dist_matrix, axis=1)

            # update labels
            labels = np.argmin(dist_matrix, axis=1)

            if np.all(labels == old_labels):
                break
            # update centers
            for i in range(self.n_clusters):
                # the ith cluster
                mask = labels == i
                masked_data = X[mask, :]
                centers[i, :] = np.mean(masked_data, axis=0)
        self.centers = centers

    def predict(self, X):
        dist_matrix = np.linalg.norm(X[:,None,:] - self.centers[None, ...], ord=2, axis=-1)
        pred = np.argmin(dist_matrix, axis=-1)
        return pred







if __name__ == '__main__':
    # 生成三簇数据，用于测试聚类结果
    cluster1 = np.random.normal(0, 2, size=(100, 2)) + np.asarray(
        [[-3.0, -5.0]])
    label1 = np.ones(100) * 0

    cluster2 = np.random.normal(0, 3, size=(100, 2)) + np.asarray([[2.0, 8.0]])
    label2 = np.ones(100) * 1

    cluster3 = np.random.normal(0, 5, size=(100, 2)) + np.asarray([[10.0, 1.0]])
    label3 = np.ones(100) * 2
    data = np.concatenate([cluster1, cluster2, cluster3], axis=0)
    label = np.concatenate([label1, label2, label3], axis=0)
    # 打乱数据
    shuffle = np.random.permutation(len(data))
    data = data[shuffle]
    label = label[shuffle]

    # 使用kmeans聚类
    kmeans = KMeans(3)
    kmeans.fit(data)
    pred = kmeans.predict(data)

    # 绘制数据散点图
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('Ground Truth')
    # 更具真实label绘制散点图，不同类使用不同颜色表示
    plt.scatter(data[:, 0], data[:, 1], c=label / label.max())

    plt.subplot(122)
    plt.title('K Means Prediction')
    # 更具kmeans聚类结果绘制散点图，不同类使用不同颜色表示
    plt.scatter(data[:, 0], data[:, 1], c=pred / pred.max())
    plt.show()
