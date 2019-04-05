using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using mlnet_model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace mlnet_webapi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class StudentsController : ControllerBase
    {
        private readonly PredictionEngine<StudentTrainingModel, StudentPredictionModel> predictionEngine;

        public StudentsController(PredictionEngine<StudentTrainingModel, StudentPredictionModel> predictionEngine)
        {
            this.predictionEngine = predictionEngine;
        }

        [HttpPost("predict")]
        public IActionResult Predict([FromBody] StudentTrainingModel model)
        {
            var prediction = predictionEngine.Predict(model);
            return Ok(prediction);
        }
    }
}
