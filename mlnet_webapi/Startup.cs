using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.HttpsPolicy;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Blob;
using mlnet_model;

namespace mlnet_webapi
{
    public class Startup
    {
        private const string BlobProtocolConfigName = "BLOB_DEFAULT_ENDPOINTS_PROTOCOL";
        private const string BlobAccountNameConfigName = "BLOB_ACCOUNT_NAME";
        private const string BlobAccountKeyConfigName = "BLOB_ACCOUNT_KEY";
        private const string BlobEndpointSuffixConfigName = "BLOB_ENDPOINT_SUFFIX";

        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddMvc().SetCompatibilityVersion(CompatibilityVersion.Version_2_2);

            var defaultEndpointsProtocol = Configuration[BlobProtocolConfigName];
            var accountName = Configuration[BlobAccountNameConfigName];
            var accountKey = Configuration[BlobAccountKeyConfigName];
            var endpointSuffix = Configuration[BlobEndpointSuffixConfigName];
            var connectionString = $"DefaultEndpointsProtocol={defaultEndpointsProtocol};AccountName={accountName};AccountKey={accountKey};EndpointSuffix={endpointSuffix}";
            var storageAccount = CloudStorageAccount.Parse(connectionString);
            var blobClient = storageAccount.CreateCloudBlobClient();
            var mlnetContainer = blobClient.GetContainerReference("mlnetmodel");
            var blob = mlnetContainer.GetBlobReference("student_perf_model.zip");
            using (var ms = new MemoryStream())
            {
                blob.DownloadToStream(ms);
                services.AddSingleton(new ModelData(ms.ToArray()));
            }

            // 注册MLContext对象
            services.AddScoped<MLContext>();
            services.AddScoped(serviceProvider =>
            {
                // 通过serviceProvider获取已注册的MLContext对象
                var mlContext = serviceProvider.GetRequiredService<MLContext>();
                var dataStream = serviceProvider.GetRequiredService<ModelData>().DataBytes;
                using (var modelStream = new MemoryStream(dataStream))
                {
                    var model = mlContext.Model.Load(modelStream);
                    return model.CreatePredictionEngine<StudentTrainingModel, StudentPredictionModel>(mlContext);
                }
            });
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IHostingEnvironment env, IApplicationLifetime lifetime)
        {
            lifetime.ApplicationStopping.Register(() =>
            {
                var modelDataStream = app.ApplicationServices.GetService<ModelData>();
                modelDataStream.Dispose();
            });

            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }
            else
            {
                // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
                app.UseHsts();
            }

            app.UseMvc();
        }
    }
}
