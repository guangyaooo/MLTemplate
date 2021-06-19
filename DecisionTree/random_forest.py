import pandas as pd
import numpy as np
from DecisionTree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier(object):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 d=None,
                 random_state=0):
        '''
        Args:
            n_estimators: decision tree numbers
            criterion: {"gini", "entropy", "error"}, default="gini"
                The function to measure the quality of a split. Supported
                criteria are "gini" for the Gini impurity , "entropy" for
                the information gain and "error" for the classification error.
            max_depth: int, default=None
                The maximum depth of the tree. If None, then nodes are expanded
                until all leaves are pure.
            d: int, default=None
                if m is not None, the algorithm will randomly select d features
                without replacement
            random_state: init random state
        '''
        self.estimators = [DecisionTreeClassifier(criterion, max_depth, d,
                                                  random_state + i) for i in
                           range(n_estimators)]
        self.random_state = random_state

    def fit(self, X, y):
        '''
        fit model.
        Args:
            X: pandas.DataFrame or numpy.ndarray
                N x M, training data
            y: pandas.DataFrame or numpy.ndarray
                N, training label

        Returns:

        '''
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        rgen = np.random.RandomState(self.random_state)
        N, _ = X.shape
        indices = np.arange(N)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.squeeze(y.values)
        for estimator in self.estimators:
            sampled_indices = rgen.choice(indices, size=N, replace=True)
            sampled_X = X.iloc[sampled_indices]
            sampled_y = y[sampled_indices]
            estimator.fit(sampled_X, sampled_y)

    def predict(self, X):
        '''
        Returns predicted categories of `X`
        Args:
            X: pandas.DataFrame of numpy.ndarray
                input data

        Returns:
            pred_y: predicted categories of `X`
        '''
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict(X))
        preds = np.asarray(preds)
        preds = np.split(preds, preds.shape[1], axis=1)
        predict = []
        for p in preds:
            label, count = np.unique(p.squeeze(), return_counts=True)
            predict.append(label[np.argmax(count)])
        predict = np.asarray(predict)
        return predict


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=1)
    model.fit(X, y)
    pred = model.predict(X)
    acc = np.mean(pred == y)
    print('IRIS Test acc %.4f' % acc)
