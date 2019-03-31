using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace mlnet_trainer.Model
{
    public class StudentPredictionModel
    {
        public string TrainerName;

        [ColumnName("Score")]
        public float PredictedG3;

        public override string ToString() => PredictedG3.ToString();
    }
}
