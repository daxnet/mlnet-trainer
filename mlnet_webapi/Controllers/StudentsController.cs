using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using mlnet_model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;

namespace mlnet_webapi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class StudentsController : ControllerBase
    {
        private readonly IConfiguration configuration;
        private readonly PredictionEngine<StudentTrainingModel, StudentPredictionModel> predictionEngine;

        public StudentsController(PredictionEngine<StudentTrainingModel, StudentPredictionModel> predictionEngine,
            IConfiguration configuration)
        {
            this.predictionEngine = predictionEngine;
            this.configuration = configuration;
        }

        [HttpPost("predict")]
        public IActionResult Predict([FromBody] StudentTrainingModel model)
        {
            var prediction = predictionEngine.Predict(model);
            return Ok(prediction);
        }

        [HttpGet("cred")]
        public IActionResult GetCredentialSettings()
            => Ok(this.configuration["BLOB_ACCOUNT_NAME"]);
    }
}
