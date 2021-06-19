import pandas as pd
from collections import OrderedDict
from itertools import product
import numpy as np


def evaluate(data, cond_prob_table, var_range, label_column):
    '''
    使用给定的cond_prob_table，评估'data'
    :param data: 要评估的数据
    :param cond_prob_table: 条件概率表
    :param var_range: 随机变量取值范围
    :param label_column: label所在列的列名
    :return: 准确率，概率矩阵
    '''
    label = data[label_column].to_numpy()
    preds = []
    #用来给定样本时的概率矩阵，logP(y_i|sample_j) = log_prob_matrix[j,i]/C_i,C_i为归一化系数
    log_prob_matrix = np.zeros((len(data), len(var_range[label_column])))

    # 遍历所有的sample
    for sample_idx, sample in data.iterrows():
        pred_y = var_range[label_column][0]
        max_log_prob_acc = -float('inf')

        # 枚举所有label的取值可能
        for y_idx, y in enumerate(var_range[label_column]):
            log_prob_acc = 0

            # 计算log 概率的累积和
            for var in var_range.keys():
                if var == label_column:
                    continue
                x = sample[var] # 当前样本的var特征对应的值
                if var_range[var] == 'continuous':
                    # 连续变量条件概率，使用高斯分布分布计算
                    mean, std = cond_prob_table[var][y]
                    # prob = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-np.square((x - mean)) / (2 * np.square(std)))
                    log_prob = -np.square((x - mean)) / (2 * np.square(std)) - np.log(std * np.sqrt(2 * np.pi))
                else:
                    # 离散变量条件概率，直接从概率表获得
                    prob = cond_prob_table[var][(x, y)]
                    if prob == 0:  # 等于0时直接退出循环
                        log_prob_acc = -float('inf')
                        break
                    log_prob = np.log(prob)
                log_prob_acc += log_prob # 计算log概率值，防止连乘时数值溢出
            # 当前的log概率大于最大的log概率，更新预测
            if log_prob_acc > max_log_prob_acc:
                pred_y = y
                max_log_prob_acc = log_prob_acc
            log_prob_matrix[sample_idx, y_idx] = log_prob_acc
        preds.append(pred_y)
    preds = np.asarray(preds)
    acc = np.sum(preds == label) / len(preds)
    return acc, np.exp(log_prob_matrix)


def gaussian_nbc(train_set: pd.DataFrame, test_set: pd.DataFrame):
    label_column = 'label' # label 对应的列
    var_range = OrderedDict()

    # 计算每个随机变量的可取值
    for var in train_set.columns.values:
        var_set = set(train_set[var])
        if len(var_set) > 0.5 * len(train_set):
            # 连续变量表示为'continuous'，方便后面判断
            var_range[var] = 'continuous'
        else:
            # 离散变量获得所有不同的可取值
            var_range[var] = sorted(list(var_set))

    cond_prob_table = OrderedDict()
    # 计算条件概率表，条件概率表的格式如下
    for var in train_set.columns.values:
        if var == label_column:
            continue
        prob_list = []
        if var_range[var] == 'continuous':
            # 如果是连续变量，则计算给定y时的，x的均值和标准差
            for y in var_range[label_column]:
                y_mask = train_set[label_column] == y
                x_data = train_set[var][y_mask]
                mean = x_data.mean()
                std = x_data.std()
                prob_list.append((y, (mean, std)))
        else:
            # 如果是离散变量，则计算给定y时的，x所有可能取值下条件概率，也就是P(x|y)
            for y, x in product(var_range[label_column], var_range[var]):
                y_mask = train_set[label_column] == y # bool索引，train_set[label_column]为y的地方为True，否则为False
                x_mask = train_set[var] == x
                xy_mask = x_mask & y_mask # 逐位进行并运算，也就是说train_set[label_column]为y且train_set[var]为x的地方为True，否则为False
                # Original 计算条件概率，原始方法
                prob = np.sum(xy_mask) / np.sum(y_mask)

                # Laplace 平滑
                # prob = (np.sum(xy_mask) + 1)/ (np.sum(y_mask) + len(var_range[var]))

                # m-estimate
                # m = 0.1
                # p = np.sum(x_mask) / len(train_set)
                # prob = (np.sum(xy_mask)+m*p) / (np.sum(y_mask)+m)

                prob_list.append(((x, y), prob))

        # 将list转换为OrderedDict，list中每个元素是一个二元tuple，
        # 每个tuple包含两个元素，第一是字典的key，第二为value
        cond_prob_table[var] = OrderedDict(prob_list)

    # 评估
    train_acc, _ = evaluate(train_set, cond_prob_table, var_range, label_column)
    test_acc, test_prob_matrix = evaluate(test_set, cond_prob_table, var_range, label_column)
    # Part 1
    print('The accuracy on training data is %.04f' % train_acc)
    print('The accuracy on testing data is %.04f' % test_acc)

    # Part 2
    test_data = test_set.drop(label_column,1)
    for i, test_prob_row in enumerate(test_prob_matrix[:5]):
        cond_prob_sum = test_prob_row.sum()
        for cond_p, y in zip(test_prob_row, var_range[label_column]):
            cond = test_data.iloc[i].to_list()
            cond = ','.join(map(str, cond))
            print('P(%s|%s)=%.04f' % (y, cond, cond_p / cond_prob_sum))

    # Part 3
    for var in cond_prob_table.keys():
        if var_range[var]=='continuous':
            for y in var_range[label_column]:
                print('%s,label=%s: mean=%.04f,std=%.04f'%(var,str(y),cond_prob_table[var][y][0],cond_prob_table[var][y][1]))
        else:
            for y, x in product(var_range[label_column], var_range[var]):

                print('P(%s=%s|%s)=%.04f' % (var,str(x), str(y), cond_prob_table[var][(x, y)]))


if __name__ == '__main__':
    train_set = pd.read_csv('training sample.csv')
    test_set = pd.read_csv('testing sample.csv')
    gaussian_nbc(train_set, test_set)
