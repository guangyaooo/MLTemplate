import numpy as np


class LinearRegression:
    def __init__(self):
        '''Linear Regression

        This algorithm implements the linear regression without penalty
        '''
        self.w = None

    def fit(self,X,y):
        '''
        :param X: N x M
        :param y: N
        :return:
        '''
        if len(X.shape) == 1:
            X = X.reshape((-1,1))
        X = np.hstack([X,np.ones((X.shape[0],1))])
        self.w = np.linalg.pinv(X) @ y

    def predict(self,X):
        if self.w is None:
            print('Please call model.fit(X,y) to train the model before predicting.')
        if len(X.shape) == 1:
            X = X.reshape((-1,1))
        X = np.hstack([X,np.ones((X.shape[0],1))])
        pred_y = X @ self.w
        return pred_y


if __name__ == '__main__':
    fake_x = np.linspace(1, 3, 100)
    fake_y = -3 * fake_x + 2 + np.random.rand(100)

    model = LinearRegression()
    model.fit(fake_x, fake_y)
    y_pred = model.predict(fake_x)
    from matplotlib.pylab import plt
    plt.scatter(fake_x, fake_y,marker='o', c='c',  label='real data')
    plt.plot(fake_x, y_pred, '-r',label='least square')
    plt.legend()
    plt.show()
