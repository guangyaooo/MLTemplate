import numpy as np
from utils import common
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.spatial.distance as ssd


def RBF_kernel(sigma):
    def kernel(x, y):
        '''
        :param x: N x d
        :param y: M x d
        :return: N x M
        '''
        dist = ssd.cdist(x, y, 'euclidean')
        return np.exp(-np.square(dist) * sigma)

    return kernel


def auto_scale(X):
    X_var = np.var(X,ddof=1)
    sigma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
    return sigma


def linear_kernel():
    def kernel(x, y):
        '''
        :param x: N x d
        :param y: M x d
        :return: N x M
        '''
        return x @ y.T

    return kernel


class SVM:
    def __init__(self, C, kernel, max_iter=500,show_fitting_bar=True):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.biary_encoder = common.BinaryEncoder()
        self.tqdm = tqdm if show_fitting_bar else lambda x:x

    def error(self, index):
        pred = (self.alpha * self._label).T @ self.Q[:, index] + self.b
        return pred - self._label[index]


    def fit(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-3):
        '''
        :param X: N x d
        :param y: N
        :param tol: scalar
        :return:
        '''
        N, d = X.shape
        self._X = X
        self._label = self.biary_encoder.fit_transform(y)
        self.Q = self.kernel(X, X)
        self.alpha = np.zeros(N)
        self.b = 0.0
        satisfied_count = 0

        for iter_count in self.tqdm(range(self.max_iter)):
            alpha_changed_count = 0
            for i in range(N):
                Ei = self.error(i)
                alphai_old = self.alpha[i].copy()
                yi = self._label[i]
                # select alpha i witch violates the KKT condition
                if (yi * Ei < -tol and alphai_old < self.C) or (yi * Ei > tol and alphai_old > 0):
                    # random select j
                    j = np.random.randint(0, N)
                    while j == i:
                        j = np.random.randint(0, N)
                    yj = self._label[j]
                    Ej = self.error(j)
                    alphaj_old = self.alpha[j].copy()

                    # boundary L and H.
                    if yi == yj:
                        L = max(0.0, alphai_old + alphaj_old - self.C)
                        H = min(self.C, alphai_old + alphaj_old)
                    else:
                        L = max(0.0, alphaj_old - alphai_old)
                        H = min(self.C, self.C + alphaj_old - alphai_old)

                    # eta
                    eta = 2.0 * self.Q[i, j] - self.Q[i, i] - self.Q[j, j]
                    if eta >= 0:
                        continue

                    # update alpha j
                    self.alpha[j] =alphaj_old - (yj * (Ei - Ej) / eta)

                    # clip alpha_j
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # update alpha_i
                    delta_j = alphaj_old - self.alpha[j]
                    if np.abs(delta_j) < tol:
                        continue
                    self.alpha[i] =alphai_old + yi * yj * delta_j

                    # update b
                    bi = self.b - Ei \
                         - yi * (self.alpha[i] - alphai_old) * self.Q[i, i] \
                         - yj * (self.alpha[j] - alphaj_old) * self.Q[i, j]
                    bj = self.b - Ej \
                         - yi * (self.alpha[i] - alphai_old) * self.Q[i, j] \
                         - yj * (self.alpha[j] - alphaj_old) * self.Q[j, j]
                    if 0.0+tol < self.alpha[i] < self.C-tol:
                        self.b = bi
                    elif 0.0+tol < self.alpha[j] < self.C-tol:
                        self.b = bj
                    else:
                        self.b = (bi + bj) / 2.0

                    alpha_changed_count += 1
            if alpha_changed_count == 0:
                satisfied_count += 1
            else:
                satisfied_count = 0
            if satisfied_count >= 5:
                break


        mask = self.alpha > tol
        self.support_vectors = mask
        self.X_sup = X[mask]
        self.Y_sup = self._label[mask]
        self.alpha_sup = self.alpha[mask]

        del self._X
        del self._label
        del self.alpha
        del self.Q

    def predict_proba(self, X):
        sigma = self.kernel(X, self.X_sup)
        sigma = self.Y_sup[np.newaxis, :] * self.alpha_sup[np.newaxis, :] * sigma
        y_hat = np.sum(sigma, axis=1) + self.b
        return y_hat

    def predict(self, X):
        r = self.predict_proba(X)
        r[r>0]=1
        r[r<=0]=-1
        r = self.biary_encoder.inverse_transform(r)
        return r



def plot_boundary():

    def generate_data(n_sample=50):
        x1 = np.random.rand(n_sample, 1) * 6 - 2
        y1 = x1 + np.random.normal(0, 0.7, (n_sample, 1))
        x2 = np.random.rand(n_sample, 1) * 6 - 3
        y2 = x2 + np.random.normal(0, 0.7, (n_sample, 1)) + 3
        c1 = np.hstack([x1, y1, np.ones_like(x1) * -1])
        c2 = np.hstack([x2, y2, np.ones_like(x2)])
        data_with_label = np.vstack([c1, c2])
        data, label = data_with_label[:, :-1], data_with_label[:, -1]

        return data, label

    import matplotlib.pyplot as plt
    np.random.seed(1)
    X,y = generate_data(50)
    kernels = [RBF_kernel(1.0),linear_kernel()]
    plt.figure(figsize=(10,5))
    for i,kernel in enumerate(kernels):
        plt.subplot(1,2,i+1)
        model = SVM(C=1, kernel=kernel,max_iter=1000)
        model.fit(X, y, tol=1e-5)
        sv = model.support_vectors

        plt.scatter(X[sv, 0], X[sv, 1], c=y[sv], marker='*',
                    linewidths=0.5, edgecolors=(0, 0, 0, 1))
        plt.scatter(X[~sv, 0], X[~sv, 1], c=y[~sv],
                    linewidths=0.5, edgecolors=(0, 0, 0, 1))
        xvals = np.linspace(-3, 6, 200)
        yvals = np.linspace(-3, 6, 200)
        xx, yy = np.meshgrid(xvals, yvals)

        pred = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        zz = np.reshape(pred, xx.shape)
        plt.pcolormesh(xx, yy, zz, zorder=0)
        plt.contour(xx, yy, zz, levels=(-1, 0, 1), colors='w', linewidths=1.5, zorder=1, linestyles='solid')

        plt.xlim([-3, 6])
        plt.ylim([-3, 6])
    plt.show()


if __name__ == '__main__':
    plot_boundary()
    from sklearn.datasets import load_breast_cancer
    data, label = load_breast_cancer(return_X_y=True)

    sigma = auto_scale(data)
    svm = SVM(1.0, kernel=RBF_kernel(sigma))
    svm.fit(data, label)
    y_pred = svm.predict(data)
    acc = np.sum(y_pred == label) / len(y_pred)
    print('SVM Test Acc %.4f' % acc)
