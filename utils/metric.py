import numpy as np


def accuracy(y_true:np.ndarray, y_pred:np.ndarray):
    '''
    Args:
        y_true: N, ground truth labels
        y_pred: N, predict labels

    Returns:
        accuracy
    '''
    acc = np.mean(y_true == y_pred)
    return acc

def macro_average_precision_score(y_true, y_pred):
    '''
    计算AP
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 平均精度
    '''
    label_set = sorted(list(set(y_true)))
    p = []
    for label in label_set:
        pos_mask_true = y_true == label
        neg_mask_true = ~pos_mask_true

        pos_mask_pred = y_pred == label
        neg_mask_pred = ~pos_mask_pred

        tp = np.sum(y_pred[pos_mask_true] == label) # True positive
        # tn = np.sum(y_pred[neg_mask_true] != label) # True negtive
        fp = np.sum(y_true[pos_mask_pred] != label) # False positive
        # fn = np.sum(y_true[neg_mask_pred] == label) # False negtive
        p.append(tp / (tp + fp) if (tp + fp)>0 else 0)
    average_p = sum(p) / len(p)
    return average_p