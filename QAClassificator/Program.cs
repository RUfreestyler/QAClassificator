namespace QAClassificator
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var classificator = new ContactingClassificator();
            classificator.TrainModel();
            var contacting = new Contacting() { Question = "How do I delete repository?" };
            var prediction = classificator.PredictRespondent(contacting);
            Console.WriteLine("Question: {0}\nPossible respondent: {1}", contacting.Question, prediction.Respondent);
        }
    }
}
