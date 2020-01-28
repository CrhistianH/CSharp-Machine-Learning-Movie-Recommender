using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;

namespace ML_App
{
    class MovieInfo
    {
        public float movieId;
        public string movieTitle;

        public MovieInfo(string line)
        {
            using (StringReader csv_reader = new StringReader(line))
            {
                using (TextFieldParser csv_parser = new TextFieldParser(csv_reader))
                {
                    csv_parser.SetDelimiters(",");
                    csv_parser.HasFieldsEnclosedInQuotes = true;

                    var csv_array = csv_parser.ReadFields();

                    this.movieId = float.Parse(csv_array[0]);
                    this.movieTitle = csv_array[1];
                }
            }
        }
    }

    class MovieRatingInfo
    {
        public float userId;
        public float movieId;
        public float movieRating;

        public MovieRatingInfo(string line)
        {
            using (StringReader csv_reader = new StringReader(line))
            {
                using (TextFieldParser csv_parser = new TextFieldParser(csv_reader))
                {
                    csv_parser.SetDelimiters(",");
                    csv_parser.HasFieldsEnclosedInQuotes = false;

                    var csv_array = csv_parser.ReadFields();

                    this.userId = float.Parse(csv_array[0]);
                    this.movieId = float.Parse(csv_array[1]);
                    this.movieRating = float.Parse(csv_array[2]);
                }
            }
        }
    }
}
