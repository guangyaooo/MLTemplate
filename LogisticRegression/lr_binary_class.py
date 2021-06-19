import numpy as np


class LogisticRegression:
    def __init__(self, max_iter=100, learning_rate=0.01, penalty=0.0,
                 use_bias=True, w_init=None, solver='gd'):
        '''Logistic Regression for binary classification.

        This algorithm implement two optimize method, gradient descent and
        newton's method.
        Args:
            max_iter: maximum number of iterations
            learning_rate: learning rate.
            penalty: regularization strength, must be a positive float
            w_init: initialize weight if none w will be init by zeros.
            use_bias: whether to use bias.
            solver: {'newton', 'gd'}
                Algorithm to use in the optimization problem.
        '''
        self.max_iter = max_iter
        self.w = w_init
        self.penalty = penalty
        self.solver = solver
        self.learning_rate = learning_rate
        self.use_bias = use_bias

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        '''

        Args:
            X: N x D, independent variable
            y: N, dependent variable

        Returns:

        '''
        if self.use_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        penalty_matrix = np.eye(X.shape[1]) * self.penalty
        if self.use_bias:
            penalty_matrix[-1, -1] = 0

        w = np.zeros(X.shape[1]) if self.w is None else self.w

        for _ in range(self.max_iter):
            w_prev = np.copy(w)
            y_hat = self.sigmoid(X @ w)
            grad = X.T @ (y_hat - y) + penalty_matrix @ w
            if self.solver == 'newton':
                hessian = (X.T @ np.diag(
                    y_hat * (1 - y_hat)) @ X) + penalty_matrix
                hessian_inv = np.linalg.pinv(hessian)
                w = w - hessian_inv @ (grad)
            else:
                w = w - self.learning_rate * grad

            if np.allclose(w, w_prev):
                break
        self.w = w

    def predict_prob(self, X):
        if self.use_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        prob = self.sigmoid(X @ self.w)
        return prob

    def predict(self, X):
        return (self.predict_prob(X) > 0.5).astype(np.int)



if __name__ == '__main__':
    from utils import metric
    data = np.loadtxt('../data/binary_classification_data_100rows.txt',
                      np.float, delimiter='\t', skiprows=1)
    X = data[:, 1:] # 100 x 45, no bias
    y = data[:, 0] # 100, binary classification

    # 1. test gradient descent
    model = LogisticRegression(learning_rate=0.1,
                               solver='gd')
    model.fit(X, y)
    y_pred = model.predict(X)
    print('Gradient descent acc:%.4g' % metric.accuracy(y, y_pred))

    # 2. test gradient descent with penalty
    model = LogisticRegression(learning_rate=0.1,
                               penalty=1,
                               solver='gd')
    model.fit(X, y)
    y_pred = model.predict(X)
    print('Gradient descent with penalty acc:%.4g' % metric.accuracy(y, y_pred))

    # 3. test newton's method
    model = LogisticRegression(solver='newton')
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Newton's descent acc:%.4g" % metric.accuracy(y, y_pred))

    # 4. test newton's method
    model = LogisticRegression(solver='newton', penalty=1)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Newton's descent with penalty acc:%.4g" % metric.accuracy(y, y_pred))
