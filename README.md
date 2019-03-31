# mlnet-trainer
一个基于ML.NET监督式（Supervised）机器学习进行学生成绩预测的案例程序。本案例训练数据来自于[https://archive.ics.uci.edu/ml/datasets/Student+Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)，其中包含了来自两个葡萄牙学校的学生情况调查数据以及他们的综合学习成绩。

本案例代码包含以下功能：
- 针对ML.NET所支持的各种回归式训练算法，基于给定的输入数据进行模型训练
- 基于给定的测试样本，对每种训练算法进行评估，得到最优算法
- 根据最优算法生成训练模型，并将模型保存并发布到本地或者Azure Blob Storage

由本案例保存的训练模型将被`mlnet-webapi`项目所使用，用以提供学生成绩预测服务的RESTful API接口。

## 数据文件
本案例包含两个数据文件：
- `student-mat.txt`：学生成绩训练样本，用以训练机器学习模型
- `student-mat-test.txt`：学生成绩测试样本，用于模型评估

数据文件都是以TAB为分隔的文本文件（Tab Separated Values，TSV）。各字段名称、类型以及说明可以参考以上数据来源页面的详细注解。
