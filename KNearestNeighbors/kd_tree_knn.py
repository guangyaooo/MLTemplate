import numpy as np
from sklearn.neighbors import KDTree


class KDTreeKNN:
    def __init__(self, k):
        '''K Nearest Neighbors Classifier

        This algorithm implement K neighbors with KDTree.
        Args:
            k: number of neighbors used in prediction stage
        '''
        self.k = k

    def fit(self, X, y):
        self.kdtree = KDTree(X)
        self.labels = y

    def predict(self, X):
        '''
        :param X: N x M
        :return:
        '''
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        _, indices = self.kdtree.query(X, self.k)
        nei_labels = self.labels[indices]
        pred_y = np.asarray([np.argmax(np.bincount(l)) for l in nei_labels])
        assert len(pred_y) == len(X)
        return pred_y


if __name__ == '__main__':
    from sklearn import datasets

    data, labels = datasets.load_digits(return_X_y=True)
    # data = np.load('../data/mnist/mnist_data.npy')
    # labels = np.load('../data/mnist/mnist_labels.npy')

    n_sample = len(data)
    shuffle = np.random.permutation(n_sample)
    data = data[shuffle]
    labels = labels[shuffle]
    split = int(n_sample * 0.8)
    train_data, test_data = data[:split], data[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    model = KDTreeKNN(25)
    model.fit(train_data, train_labels)
    test_pred = model.predict(test_data)
    from utils.metric import accuracy

    print('ScratchKNN Acc: %.4g' % accuracy(test_labels, test_pred))
