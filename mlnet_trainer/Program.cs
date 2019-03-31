using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers;
using mlnet_trainer.Model;
using System;
using System.Collections.Generic;

namespace mlnet_trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var trainers = new List<ITrainerEstimator<ISingleFeaturePredictionTransformer<ModelParametersBase<float>>, ModelParametersBase<float>>>()
            {
                mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"),
                mlContext.Regression.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"),
                mlContext.Regression.Trainers.FastTreeTweedie(),
                mlContext.Regression.Trainers.GeneralizedAdditiveModels(),
                mlContext.Regression.Trainers.OnlineGradientDescent(),
                mlContext.Regression.Trainers.PoissonRegression(),
                mlContext.Regression.Trainers.StochasticDualCoordinateAscent()
            };

            var session = new LearningSession(mlContext, trainers);
            var trainingDataView = session.LoadDataView("student-mat.txt");
            var testingDataView = session.LoadDataView("student-mat-test.txt");
            session.TrainAndTest(trainingDataView, testingDataView);
            var regressionMetrics = session.RegressionMetrics;
            foreach (var item in regressionMetrics)
            {
                LearningSession.OutputRegressionMetrics(item.Key, item.Value);
            }

            var predictingModel = new StudentTrainingModel
            {
                School = "GP",
                Absences = 7,
                Age = 18,
                Traveltime = 1,
                Studytime = 3,
                Failures = 0,
                Guardian = "mother",
                Paid = "no",
                Famrel = 5,
                Famsize = "GT3",
                Higher = "yes",
                Sex = "F"
            };
            var predictions = session.Predict(predictingModel);
            foreach(var pred in predictions)
            {
                Console.WriteLine($"Trainer: {pred.TrainerName}, Predicted: {pred.PredictedG3}");
            }
            /*
            var mlContext = new MLContext();
            var textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                Columns = new TextLoader.Column[]
                {
                    new TextLoader.Column("School", DataKind.String, 0),
                    new TextLoader.Column("Sex", DataKind.String, 1),
                    new TextLoader.Column("Age", DataKind.Single, 2),
                    new TextLoader.Column("Famsize", DataKind.String, 4),
                    new TextLoader.Column("Guardian", DataKind.String, 11),
                    new TextLoader.Column("Traveltime", DataKind.Single, 12),
                    new TextLoader.Column("Studytime", DataKind.Single, 13),
                    new TextLoader.Column("Failures", DataKind.Single, 14),
                    new TextLoader.Column("Paid", DataKind.String, 17),
                    new TextLoader.Column("Higher", DataKind.String, 20),
                    new TextLoader.Column("Famrel", DataKind.Single, 23),
                    new TextLoader.Column("Absences", DataKind.Single, 29),
                    new TextLoader.Column("G3", DataKind.Single, 31)
                }
            });

            var trainingDataView = textLoader.Load("student-mat.txt");
            var pipeline = mlContext.Transforms.CopyColumns(inputColumnName: "G3", outputColumnName: "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("School"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Sex"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Age"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Famsize"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Guardian"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Traveltime"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Studytime"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Failures"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Paid"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Higher"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Famrel"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Absences"))
                .Append(mlContext.Transforms.Concatenate("Features", "School",
                    "Sex", "Age", "Famsize", "Guardian", "Traveltime", "Studytime", "Failures", "Paid", "Higher", "Famrel", "Absences"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.FastForest());

            var trainedModel = pipeline.Fit(trainingDataView);

            var testDataView = textLoader.Load("student-mat-test.txt");
            var predictions = trainedModel.Transform(testDataView);
            var regressionMetrics = mlContext.Regression.Evaluate(predictions);
            OutputRegressionMetrics(regressionMetrics);
            var predictionEngine = trainedModel.CreatePredictionEngine<StudentTrainingModel, StudentPredictionModel>(mlContext);
            var predictingModel = new StudentTrainingModel
            {
                School = "GP",
                Absences = 7,
                Age = 18,
                Traveltime = 1,
                Studytime = 3,
                Failures = 0,
                Guardian = "mother",
                Paid = "no",
                Famrel = 5,
                Famsize = "GT3",
                Higher = "yes",
                Sex = "F"
            };

            var prediction = predictionEngine.Predict(predictingModel);
            Console.WriteLine(prediction);*/
        }

        static void OutputRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"L1: {metrics.L1}, L2: {metrics.L2}, LossFn: {metrics.LossFn}, Rms: {metrics.Rms}, RSquared: {metrics.RSquared}");
        }
    }
}
