using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace ML_App
{
    class Program
    {

        static void Main(string[] args)
        {
            int userId;

            Console.WriteLine("Enter a user ID:");

            while (!int.TryParse(Console.ReadLine(), out userId)){
                Console.WriteLine("Incorrect Input. Please enter a valid integer number.");
            }

            MLContext mlContext = new MLContext();

            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

            //EvaluateModel(mlContext, testDataView, model);

            ShowMenu(mlContext, model, userId);
            
            //SaveModel(mlContext, trainingDataView.Schema, model);

            Console.WriteLine("\nEnd of the program. Press any key to close.");
            Console.ReadKey();
        }





        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100,
                Quiet = true
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            //Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            //Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            //Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            //Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static void ShowMenu(MLContext mlContext, ITransformer model, int userId)
        {
            Console.WriteLine("\n========== MENU ==========");
            Console.WriteLine("1. Get top 5 recommendation movies for user '" + userId + "'.");
            Console.WriteLine("2. Predict if user '" + userId + "' would like a specific movie.");
            Console.WriteLine("3. EXIT.");
            int menuSelection = int.Parse(Console.ReadLine());

            switch (menuSelection)
            {
                case 1:
                    UseModelForPrediction(mlContext, model, userId);
                    break;
                case 2: UseModelForSinglePrediction(mlContext, model, userId);
                    break;
                default:
                    break;
            }
        }

        public static void UseModelForPrediction(MLContext mlContext, ITransformer model, int userId)
        {
            Console.WriteLine("\n=============== Top 20 recommended movies for user '" + userId +"' are: ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var movieList = File.ReadLines("Data/recommendation-movies.csv").Skip(1).Select(line => new MovieInfo(line)).ToList();
            var movieRatingsList = File.ReadLines("Data/recommendation-ratings-train.csv").Skip(1).Select(line => new MovieRatingInfo(line)).ToList();
            var alreadyRatedMovieIds = from movie in movieRatingsList where movie.userId == userId select movie.movieId;

            var topRecommendedMovies = (from movie in movieList.AsParallel().Where(movie => !alreadyRatedMovieIds.Contains(movie.movieId))
                        let prediction = predictionEngine.Predict(
                           new MovieRating()
                           {
                               userId = userId,
                               movieId = movie.movieId
                           })
                        orderby prediction.Score descending
                        select (MovieId: movie.movieId, Score: prediction.Score, Title: movie.movieTitle)).Take(20);

            foreach (var movie in topRecommendedMovies)
                Console.WriteLine($"  Movie: {movie.Title,-60}\tPredicted score: {movie.Score:N2}");
            Console.WriteLine();
        }

        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model, int userId)
        {

           Console.WriteLine("\nEnter a movieID");
           int movieID = int.Parse(Console.ReadLine());
           Console.WriteLine();

           var movieList = File.ReadLines("Data/recommendation-movies.csv").Skip(1).Select(line => new MovieInfo(line)).ToList();


            //Console.WriteLine("=============== Making a single prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var testInput = new MovieRating { userId = userId, movieId = movieID };
            var movieRatingPrediction = predictionEngine.Predict(testInput);
            string movieTitle = (from movie in movieList where movieID == movie.movieId select movie.movieTitle).FirstOrDefault();

            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine($"YES! Movie {testInput.movieId} - '{movieTitle}' is recommended for user {testInput.userId} with a score of: {movieRatingPrediction.Score:N1}");
            }
            else
            {
                Console.WriteLine($"NO! Movie {testInput.movieId} - '{movieTitle}' is not recommended for user '{testInput.userId}' with a score of: {movieRatingPrediction.Score:N1}");
            }
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }


    }
}
