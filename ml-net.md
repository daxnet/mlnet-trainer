# ML.NET机器学习实践
## 机器学习
机器学习是一种对算法和统计数据模型进行科学学习的方式，通过使用这种方式，计算机系统能够有效地基于模式与推断，而非遵循特定的指令序列来完成一项特定的任务。机器学习是人工智能科学的一个分支，属于人工智能范畴。
[https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)

## 分类
机器学习可以分成如下几类：
- 监督学习（Supervised Learning）
- 无监督学习（Unsupervised Learning）
- 半监督学习（Semi-supervised Learning）
- 增强学习（Reinforcement Learning）

### 监督学习
从给定的训练数据集中学习出一种算法，当的数据到来时，可以根据这个函数预测结果。监督学习的训练集要求是包括输入和输出，也可以说是特征和目标。训练集中的目标是由人标注的。常见算法分为：**统计分类**（Classification，根据训练模型，通过给定的特征属性，预测目标属性属于哪个分类）和**回归分析**（Regression，根据训练模型，通过给定的特征属性，预测目标属性的取值）。

### 无监督学习
与监督学习相比，训练集没有人为标注的结果（没有人会对训练数据集中的某个属性进行标注，标注其为哪个分类，或者取值是多少）。常见算法有**聚类**（Clustering）、**无监督异常情况检测**（Unsupervised Anomaly Detection）等。

### 半监督学习
介于监督学习与无监督学习之间，根据部分已被标记的数据来推断未标注数据的标注信息，并实现预测。

### 增强学习
机器为了达成目标，随着环境的变动，而逐步调整其行为，并评估每一个行动之后所到的回馈是正向的或负向的。

## 机器学习算法衡量标准

### 统计分类（Classification）算法衡量标准
- Classification Accuracy
- Logarithmic Loss
- Area Under ROC Curve
- Confusion Matrix
- Classification Report

### 回归分析（Regression）算法衡量标准
- Mean Absolute Error
- Mean Squared Error
- R^2 (R-Squared)

以上参考：[Metrics To Evaluate Machine Learning Algorithms in Python](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/)

## ML.NET支持的机器学习任务
ML.NET支持以下机器学习任务：
- 二元分类（Binary Classification）：预测目标属性的取值是“真”或“假”（0或1，Yes或No）
- 多类分类（Multiclassification）：预测目标属性的取值属于哪一分类
- 回归（Regression）：预测目标属性的结果值（一般是得到一个浮点数）
- 聚类分析（Clustering）：根据数据集中数据的某种特性，将数据进行分组
- 异常情况检测（Anomaly Detection）：根据数据集的数据，识别出小部分的“特殊化”数据（异常数据）
- 排名（Ranking）：机器为了达成目标，随着环境的变动，而逐步调整其行为，并评估每一个行动之后所到的回馈是正向的或负向的
- 建议（Recommendation）：支持生成推荐产品或服务的列表，例如，你为用户提供历史电影评级数据，并希望向他们推荐接下来可能观看的其他电影

以上参考：[https://docs.microsoft.com/zh-cn/dotnet/machine-learning/resources/tasks](https://docs.microsoft.com/zh-cn/dotnet/machine-learning/resources/tasks)

## ML.NET机器学习实践步骤
1. 数据预处理与规整化
2. 确定问题类型（是分类、回归还是聚合等）
3. 确定特征属性与目标属性
4. 使用相应分类下的不同算法，基于训练数据集进行模型训练
5. 根据算法衡量标准对模型进行评估，选择合适算法
6. 基于最优算法生成并发布模型
7. 构建API
8. 部署API
