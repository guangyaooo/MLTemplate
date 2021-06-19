import numpy as np


class LinearRegression:
    def __init__(self, penalty=0):
        '''Linear Regression

        This algorithm implements the linear regression with penalty
        Args:
            penalty: regularization strength.
        '''
        self.w = None
        self.penalty = penalty

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X[:, None]
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        I = np.identity(X.shape[1])
        I[-1, -1] = 0
        self.w = np.linalg.inv(X.T @ X + self.penalty * I) @ X.T @ y

    def predict(self, X):
        if self.w is None:
            print('Please call model.fit(X,y) to train the model before predicting.')
        if len(X.shape) == 1:
            X = X[:, None]
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        pred_y = X @ self.w
        return pred_y

if __name__ == '__main__':
    fake_x = np.linspace(1, 3, 100)
    fake_y = -3 * fake_x + 2 + np.random.rand(100)

    model = LinearRegression(penalty=1)
    model.fit(fake_x, fake_y)
    y_pred = model.predict(fake_x)
    from matplotlib.pylab import plt

    plt.scatter(fake_x, fake_y, marker='o', c='c', label='real data')
    plt.plot(fake_x, y_pred, '-r', label='least square')
    plt.legend()
    plt.show()