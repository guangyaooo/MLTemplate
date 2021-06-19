import numpy as np
import utils
from collections import OrderedDict
from itertools import product


class NaiveBayes:
    '''
    This naive bayes only support discrete X
    '''

    def fit(self, data, label):
        var_range = OrderedDict()
        N, K = data.shape

        for var in range(K):
            var_range[var] = sorted(list(set(data[:, var])))
        label_range = sorted(list(set(label)))

        cond_prob_table = OrderedDict()
        for var in range(K):
            prob_list = []
            for y, x in product(label_range, var_range[var]):
                y_mask = label == y
                x_mask = data[:, var] == x
                xy_mask = x_mask & y_mask
                prob = np.sum(xy_mask) / np.sum(y_mask)

                # Laplace smooth
                # prob = (np.sum(xy_mask) + 1)/ (np.sum(y_mask) + len(var_range[var]))

                # m-estimate
                # m = 0.1
                # p = np.sum(x_mask) / len(train_set)
                # prob = (np.sum(xy_mask)+m*p) / (np.sum(y_mask)+m)

                prob_list.append(((x, y), prob))


            cond_prob_table[var] = OrderedDict(prob_list)

        self.cond_prob_table = cond_prob_table
        self.var_range = var_range
        self.label_range = label_range


    def predict(self, data):
        preds = []
        for sample_idx, sample in enumerate(data):
            pred_y = self.label_range[0]
            max_log_prob_acc = -float('inf')

            for y_idx, y in enumerate(self.label_range):
                log_prob_acc = 0
                for var in self.var_range.keys():
                    x = sample[var]
                    prob = self.cond_prob_table[var][(x, y)]
                    if prob == 0:
                        log_prob_acc = -float('inf')
                        break
                    log_prob = np.log(prob)
                    log_prob_acc += log_prob
                if log_prob_acc > max_log_prob_acc:
                    pred_y = y
                    max_log_prob_acc = log_prob_acc
            preds.append(pred_y)
        preds = np.asarray(preds)
        return preds



if __name__ == '__main__':
    import pandas as pd
    from utils import common, metric

    data = pd.read_excel('../data/category_data_with_discrete_x.xlsx',
                         skiprows=1)
    data = data.to_numpy()
    data, label = data[:, :-1], data[:, -1]
    train_data, train_label, val_data, val_label = common.train_val_split(data,
                                                                          label,
                                                                          0.8)
    dim = train_data.shape[1]
    model = NaiveBayes()
    model.fit(train_data, train_label)
    val_pred = model.predict(val_data)
    print('Discrete Naive Bayes acc: %.4g' % metric.accuracy(val_label, val_pred))
