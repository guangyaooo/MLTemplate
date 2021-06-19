import numpy as np
from tqdm import tqdm


class ScratchKNN:
    def __init__(self, k):
        '''K Nearest Neighbors Classifier
        This algorithm implement K neighbors classifier from scratch.
        Args:
            k: number of neighbors used in prediction stage
        '''
        self.k = k

    def fit(self, X, y):
        '''

        Args:
            X: N, D
            y: N,

        Returns:

        '''
        self.train_data = X
        self.labels = y

    def predict(self, X):
        '''

        Args:
            X: N, D

        Returns:
            predict labels
        '''
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = X[:, np.newaxis, :]
        pred_y = []
        for X_batch in tqdm(X):
            dist = np.linalg.norm(X_batch - self.train_data, axis=1, ord=2)
            partition = np.argpartition(dist, self.k)[:self.k]
            part_labels = self.labels[partition]
            pred_y.append(np.argmax(np.bincount(part_labels)))
        pred_y = np.asarray(pred_y)
        assert len(pred_y) == len(X)
        return pred_y


if __name__ == '__main__':
    from sklearn import datasets
    data, labels = datasets.load_digits(return_X_y=True)
    # from sklearn import datasets
    # data, labels = datasets.load_digits(return_X_y=True)

    # data = np.load('../data/mnist/mnist_data.npy')
    # labels = np.load('../data/mnist/mnist_labels.npy')

    n_sample = len(data)
    shuffle = np.random.permutation(n_sample)
    data = data[shuffle]
    labels = labels[shuffle]
    split = int(n_sample * 0.8)
    train_data, test_data = data[:split], data[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    model = ScratchKNN(25)
    model.fit(train_data, train_labels)
    test_pred = model.predict(test_data)
    from utils.metric import accuracy

    print('ScratchKNN Acc: %.4g' % accuracy(test_labels, test_pred))
