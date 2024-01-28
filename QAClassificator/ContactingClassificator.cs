using Microsoft.ML;
using Microsoft.ML.Data;

namespace QAClassificator
{
    public class Contacting
    {
        [LoadColumn(0)]
        public string Question { get; set; }

        [LoadColumn(1)]
        public string Respondent { get; set; }
    }

    public class ContactingPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Respondent { get; set; }
    }

    public class ContactingClassificator
    {
        private const string questionColumnName = "Question";

        private const string respondentColumnName = "Respondent";

        private const string predictionColumnName = "PredictedLabel";

        private string trainingDataPath = Path.Combine(AppContext.BaseDirectory, "dataForTrain.csv");

        private IDataView trainingDataView;

        private MLContext mlContext;

        private PredictionEngine<Contacting, ContactingPrediction> predictionEngine;

        private ITransformer trainedModel;

        public ContactingPrediction PredictRespondent(Contacting contacting)
        {
            if (predictionEngine == null)
                throw new InvalidOperationException("Prediction engine is null.");

            return predictionEngine.Predict(contacting);
        }

        public void TrainModel()
        {
            mlContext = new MLContext();
            trainingDataView = mlContext.Data.LoadFromTextFile<Contacting>(
                path: trainingDataPath,
                hasHeader: true,
                separatorChar: ';',
                trimWhitespace: true);

            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(trainingDataView, pipeline);
            Evaluate(trainingDataView.Schema);
        }

        private IEstimator<ITransformer> ProcessData()
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: respondentColumnName, outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Feature", inputColumnName: questionColumnName))
                .AppendCacheCheckpoint(mlContext);
            
            return pipeline;
        }

        private IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Feature"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(predictionColumnName));

            trainedModel = trainingPipeline.Fit(trainingDataView);
            predictionEngine = mlContext.Model.CreatePredictionEngine<Contacting, ContactingPrediction>(trainedModel);

            var contacting = new Contacting() { Question = "What is Git branch?" };
            var prediction = PredictRespondent(contacting);

            return trainingPipeline;
        }

        private void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = mlContext.Data.LoadFromTextFile<Contacting>(
                path: trainingDataPath,
                hasHeader: true,
                separatorChar: ';',
                trimWhitespace: true);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

            Console.WriteLine($"Classification model metrics: " +
                $"MicroAccuracy: {testMetrics.MicroAccuracy:0.###}; " +
                $"MacroAccuracy: {testMetrics.MacroAccuracy:0.###}; " +
                $"LogLoss: {testMetrics.LogLoss:#.###}; " +
                $"LogLossReduction: {testMetrics.LogLossReduction:#.###}.");
        }
    }
}
