using Microsoft.ML;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers;
using mlnet_model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace mlnet_trainer
{
    class Program
    {
        const string ModelFileName = "student_perf_model.zip";
        static void Main(string[] args)
        {
            // 创建ML.NET执行上下文
            var mlContext = new MLContext();

            // 初始化训练算法
            var trainers = new List<ITrainerEstimator<ISingleFeaturePredictionTransformer<ModelParametersBase<float>>, ModelParametersBase<float>>>()
            {
                mlContext.Regression.Trainers.FastTree(),
                mlContext.Regression.Trainers.FastForest(),
                mlContext.Regression.Trainers.FastTreeTweedie(),
                mlContext.Regression.Trainers.GeneralizedAdditiveModels(),
                mlContext.Regression.Trainers.OnlineGradientDescent(),
                mlContext.Regression.Trainers.PoissonRegression(),
                mlContext.Regression.Trainers.StochasticDualCoordinateAscent()
            };

            // 创建训练任务
            var session = new LearningSession(mlContext, trainers);

            // 读入训练数据集以及测试数据集
            var trainingDataView = session.LoadDataView("student-mat.txt");
            var testingDataView = session.LoadDataView("student-mat-test.txt");
            Console.WriteLine(">>> 开始测试并评估...");
            Console.WriteLine();

            // 基于训练数据集进行训练，并基于测试数据集进行评估，然后输出评估结果
            var regressionMetrics = session.TrainAndEvaluate(trainingDataView, testingDataView);
            foreach (var item in regressionMetrics)
            {
                LearningSession.OutputRegressionMetrics(item.Key, item.Value);
            }

            // 找到RMS最小的算法，作为最优算法
            var winnerAlgorithmName = regressionMetrics.OrderBy(x => x.Value.Rms).First().Key;
            Console.WriteLine($"最优算法为：{winnerAlgorithmName}");
            Console.WriteLine();

            // 使用最优算法进行预测
            Console.WriteLine("以下是基于测试样本数据的预测结果");
            Console.WriteLine("==============================");
            var winnerModel = session.GetTrainedModel(winnerAlgorithmName);
            var samples = ReadPredictionSamples();
            foreach (var sample in samples)
            {
                var prediction = session.Predict(winnerModel, sample);
                Console.WriteLine($"测试样本G3: {sample.G3}，预测值：{prediction.PredictedG3}");
            }
            Console.WriteLine();

            // 保存模型
            Console.WriteLine("正在保存模型...");
            using (var fileStream = new FileStream(ModelFileName, FileMode.Create, FileAccess.Write))
            {
                mlContext.Model.Save(winnerModel, fileStream);
            }
            Console.WriteLine("任务成功完成.");
        }

        static IEnumerable<StudentTrainingModel> ReadPredictionSamples()
        {
            object ConvertValue(string val, Type toType)
            {
                if (toType == typeof(float))
                {
                    return Convert.ToSingle(val);
                }

                return val;
            }

            var predictionSamples = new List<StudentTrainingModel>();
            using (var fileStream = new FileStream("student-mat-test.txt", FileMode.Open, FileAccess.Read))
            {
                using (var textReader = new StreamReader(fileStream))
                {
                    var lineNumber = 0;
                    var line = string.Empty;
                    var columns = new List<string>();
                    while (!textReader.EndOfStream)
                    {
                        line = textReader.ReadLine();
                        if (lineNumber == 0)
                        {
                            columns.AddRange(line.Split('\t'));
                        }
                        else
                        {
                            var values = line.Split('\t');
                            var sample = new StudentTrainingModel();
                            for (var idx = 0; idx < values.Length; idx++)
                            {
                                var column = columns[idx];
                                var fieldInfo = typeof(StudentTrainingModel)
                                    .GetFields()
                                    .FirstOrDefault(x => string.Equals(x.Name, column, StringComparison.InvariantCultureIgnoreCase));

                                if (fieldInfo != null)
                                {
                                    fieldInfo.SetValue(sample, ConvertValue(values[idx], fieldInfo.FieldType));
                                }
                            }

                            predictionSamples.Add(sample);
                        }
                        lineNumber++;
                    }
                }
            }

            return predictionSamples;
        }
    }
}
