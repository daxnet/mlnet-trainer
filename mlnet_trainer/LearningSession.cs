using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers;
using mlnet_trainer.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace mlnet_trainer
{
    internal sealed class LearningSession
    {
        private readonly MLContext mlContext;
        private readonly TextLoader textLoader;
        private readonly List<ITrainerEstimator<ISingleFeaturePredictionTransformer<ModelParametersBase<float>>, ModelParametersBase<float>>> trainers =
            new List<ITrainerEstimator<ISingleFeaturePredictionTransformer<ModelParametersBase<float>>, ModelParametersBase<float>>>();
        private readonly Dictionary<string, ITransformer> trainedModels = new Dictionary<string, ITransformer>();
        private readonly Dictionary<string, RegressionMetrics> regressionMetrics = new Dictionary<string, RegressionMetrics>();

        public LearningSession(
            MLContext mlContext,
            IEnumerable<ITrainerEstimator<ISingleFeaturePredictionTransformer<ModelParametersBase<float>>, ModelParametersBase<float>>> regressionTrainers)
        {
            this.mlContext = mlContext;
            textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options
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

            trainers.AddRange(regressionTrainers);
        }

        public IDataView LoadDataView(string path)
            => textLoader.Load(path);

        public void TrainAndTest(IDataView trainingDataView, IDataView testDataView)
        {
            foreach(var trainer in this.trainers)
            {
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
                    .Append(mlContext.Transforms.Concatenate("Features",
                        "School",
                        "Sex",
                        "Age",
                        "Famsize",
                        "Guardian",
                        "Traveltime",
                        "Studytime",
                        "Failures",
                        "Paid",
                        "Higher",
                        "Famrel",
                        "Absences"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(trainer);

                var trainedModel = pipeline.Fit(trainingDataView);
                trainedModels.Add(trainer.GetType().Name, trainedModel);

                var predictionModel = trainedModel.Transform(testDataView);
                var regMetrics = mlContext.Regression.Evaluate(predictionModel);
                regressionMetrics.Add(trainer.GetType().Name, regMetrics);
            }
        }

        public IEnumerable<KeyValuePair<string, RegressionMetrics>> RegressionMetrics => regressionMetrics;

        public IEnumerable<StudentPredictionModel> Predict(StudentTrainingModel model)
        {
            foreach(var trainedModel in trainedModels)
            {
                var engine = trainedModel.Value.CreatePredictionEngine<StudentTrainingModel, StudentPredictionModel>(mlContext);
                var predictedModel = engine.Predict(model);
                predictedModel.TrainerName = trainedModel.Key;
                yield return predictedModel;
            }
        }

        public static void OutputRegressionMetrics(string trainer, RegressionMetrics regMetrics)
        {
            Console.WriteLine($"* {trainer}");
            Console.WriteLine("*************************************");
            Console.WriteLine($" L1: {regMetrics.L1}");
            Console.WriteLine($" L2: {regMetrics.L2}");
            Console.WriteLine($" LossFn: {regMetrics.LossFn}");
            Console.WriteLine($" Rms: {regMetrics.Rms}");
            Console.WriteLine($" RSquared: {regMetrics.RSquared}");
            Console.WriteLine();
        }
    }
}
