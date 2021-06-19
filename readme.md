# 基础机器学习算法实现

----------------------
## 简介
通过实现机器学习算法能够更加深刻的理解算法的本质，以及运行过程中会出现的问题。
本仓库为笔者在学习机器学习算法过程中实现的一些算法集合，
实现过程主要参考了《统计学习方法》和《模式识别与机器学习》这两本书。
为了方便使用和阅读，大多数算法按照如下模板实现，此外所有模型文件最后都提供了一个简单的测试数据集，直接`run`对应的文件即可运行。
```python
class ModuleName:
    def __init__(self, *args, **kwargs):
        # 初始化模型，保存模型的超参
        pass
    
    def fit(self, X, y, *args, **kwargs):
        # 模型拟合
        pass
    
    def predict(self, X)->y:
        # 预测
        pass
```

-------------------------------
## 目录

1. 聚类算法 
    - [高斯混合模型(Gaussian Mixture Model, GMM)](/clustering/gmm.py)
    - [K均值聚类(K Means Clustering)](/clustering/k_means.py)
    
2. 分类算法
    - [决策树(Decision Tree)](/DecisionTree/decision_tree.py)
    - [随机森林(Random Fores)](/DecisionTree/random_forest.py)
    - [K近邻(K Nearest Neighbors, KNN)](/KNearestNeighbors/scratch_knn.py)
    - [线性判别分析(Linear Discriminant Analysis, LDA)](/LDA/LDA.py)
    - [二分类逻辑斯蒂回归(Logistic Regression)](/LogisticRegression/lr_binary_class.py)
    - [多分类逻辑斯蒂回归(Logistic Regression)](/LogisticRegression/lr_multiclass.py)
    - [离散朴素贝叶斯(Naive Bayes)](/NaiveBayes/discrete_nb.py)
    - [连续朴素贝叶斯(Naive Bayes)](/NaiveBayes/gaussian_nb.py)
    - [多层感知机(Multi Layer Perceptron, MLP)](/NeuralNetwork/MLPClassifier.py)
    - [二分类支持向量机(Support Vector Machine, SVM)](/SupportVectorMachine/SVM.py)
    - [多分类支持向量机(one vs one)](/SupportVectorMachine/MultiClassSVM.py)
    
3. 回归算法
    - [多层感知机(Multi Layer Perceptron, MLP)](/NeuralNetwork/MLPRegressor.py)
    - [线性回归(Liner Regression)](/LeastSquare/linear_regression_with_penalty.py)
    
4. 其它
    - [主成分分析(Principal Component Analysis, PCA)](/PCA/pca.py)
    - 隐马尔可夫模型
        - [前向算法(Forward)](/HiddenMarkovModel/hmm.py)
        - [后向算法(Backward)](/HiddenMarkovModel/hmm.py)
        - [维特比算法(Viterbi)](/HiddenMarkovModel/hmm.py)
        - [BaumWelch](/HiddenMarkovModel/hmm.py)
