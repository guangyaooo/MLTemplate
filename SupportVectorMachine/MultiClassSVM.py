import numpy as np
from utils import common
from itertools import combinations
from SupportVectorMachine.SVM import RBF_kernel,linear_kernel,SVM,auto_scale


class MultiClassSVM:
    def __init__(self,C,kernel,classes=None,tol=1e-3):
        self.C = C
        self.kernel = kernel

        self.classes = classes
        self.tol=tol
        self.svms = []
        self.class_num = len(classes) if classes is not None else 0



    def fit(self,X,y):
        '''
        :param X: N x d
        :param y: N
        :return:
        '''
        if self.classes is None:
            self.classes = np.sort(np.unique(y))
            self.class_num = len(self.classes)
        for i,specified in enumerate(combinations(self.classes,2)):
            print('SVM: %d %d' % specified)
            data,label = common.data_filter(X,y,specified)
            if self.kernel=='rbf':
                sigma = auto_scale(data)
                kernel = RBF_kernel(sigma)
            elif self.kernel=='linear':
                kernel = linear_kernel()
            else:
                raise NotImplemented()

            svm = SVM(self.C, kernel, show_fitting_bar=True, max_iter=1000)
            svm.fit(data,label,tol=self.tol)
            self.svms.append(svm)



    def predict(self,X):
        '''
        :param X: N x d
        :return: N
        '''

        vote_res = []
        for svm in self.svms:
            vote_res.append(svm.predict(X).reshape((-1,1)))
        vote_res = np.concatenate(vote_res,axis=1).astype(np.int)
        pred = []
        for row in vote_res:
            pred.append(np.argmax(np.bincount(row)))
        pred = np.asarray(pred)
        return pred




if __name__ == '__main__':
    np.random.seed(1)
    from sklearn.datasets import load_iris

    data, label = load_iris(return_X_y=True)
    svm = MultiClassSVM(1.0, kernel='rbf')
    svm.fit(data, label)
    y_pred = svm.predict(data)
    acc = np.sum(y_pred == label) / len(y_pred)
    print('MultiClassSVM Test Acc %.4f' % acc)