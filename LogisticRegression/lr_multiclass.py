import numpy as np
from utils.common import onehot_encode, softmax


class MultinomialLR:
    def __init__(self,max_iter=1000,learning_rate=0.01):
        '''Multinomial Logistic Regression

        Args:
            max_iter:
            learning_rate:
        '''
        self.w = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self,X,Y):
        '''
        :param X: N x D,
        :param Y: N, range in 0, 1, 2, ..., n_classes - 1
        :return:
        '''
        if len(Y.shape)==1:
            Y = onehot_encode(Y)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.n_classes = Y.shape[1]
        W = np.zeros((np.size(X, 1), self.n_classes))
        for _ in range(self.max_iter):
            W_prev = np.copy(W)
            Y_hat = softmax(X @ W)
            grad = X.T @ (Y_hat - Y)
            W -= self.learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.w = W

    def predict_prob(self,X):
        '''

        Args:
            X: (N, D)

        Returns:

        '''
        if self.w is None:
            print('Please call model.fit(X,y) to train the model before predicting.')
            return
        X = np.hstack([X,np.ones((X.shape[0],1))])
        pred_y = softmax(X @ self.w)
        return pred_y

    def predict(self,X):
        return np.argmax(self.predict_prob(X),axis=1)

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

    model = MultinomialLR(max_iter=2000)
    model.fit(train_data, train_labels)
    test_pred = model.predict(test_data)
    from utils.metric import accuracy
    print('Multinomial Acc: %.4g' % accuracy(test_labels, test_pred))
