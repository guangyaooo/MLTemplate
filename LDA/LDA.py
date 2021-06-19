import numpy as np
import utils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class LDA:
    def __init__(self,n_component):
        self.n_component = n_component


    def fit_transform(self,X,y):
        '''
        :param X: N x d
        :param y: N
        :return: N x n_component
        '''

        N,d = X.shape
        y_set = np.sort(np.unique(y))
        K = len(y_set)
        upper_bound = min(d,K-1)
        assert self.n_component <= upper_bound,\
            'Expect n_component <= %d, got %d' % (upper_bound,self.n_component)

        inner_cov = np.zeros((K,d,d))
        inter_cov = np.zeros((K,d,d))
        mean = np.mean(X,axis=0,keepdims=True)
        for i,y_id in enumerate(y_set):
            mask = y == y_id
            X_sub = X[mask]
            mean_sub = np.mean(X_sub, axis=0, keepdims=True)
            X_sub -= mean_sub
            cov = X_sub.T @ X_sub
            inner_cov[i,:,:] = cov[:,:]
            inter_cov[i] = len(X_sub)*(mean_sub - mean).T @ (mean_sub - mean)

        Sb = np.mean(inter_cov,axis=0)
        Sw = np.mean(inner_cov, axis=0)
        try:
            Sw_inv = np.linalg.inv(Sw)
        except np.linalg.LinAlgError:
            # Singular matrix is a singular matrix,
            Sw += 1e-5 * np.identity(len(Sw))
            Sw_inv = np.linalg.inv(Sw)

        matrix = Sw_inv @ Sb
        eig_value, eig_vector = np.linalg.eig(matrix)
        index = np.argsort(eig_value)[::-1]  # descending order
        self.eig_value = eig_value[index].astype(np.float)
        self.eig_vector = eig_vector[..., index].astype(np.float)
        self.P = eig_vector[..., :self.n_component]

        Y = X @ self.P
        return Y

if __name__ == '__main__':
    specified_labels = [2, 3, 5]
    mnist_root = '../data/raw_mnist'
    from utils import common
    data,label,_,_ = common.load_raw_mnist(mnist_root)
    data,label = common.data_filter(data,label,specified_labels)
    lda = LDA(2)
    lda_data = lda.fit_transform(data,label)
    common.scatter(lda_data, label,'LDA Visualization.')