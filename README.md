# Movie Recommendation System 🎬

This Movie Recommendation System leverages a similarity matrix built using various factors such as cast, genre, and IMDb ratings. By employing **TF-IDF** (Term Frequency-Inverse Document Frequency), the system returns the top 5 most related movies based on user input.

## Features
- **Similarity Matrix**: Based on the movie's cast, genre, and IMDb ratings.
- **TF-IDF Algorithm**: Used to calculate the similarity between movies and provide recommendations.
- **Top 5 Recommendations**: The system returns the five most related movies for any given input movie.

## How It Works
1. A dataset containing movies with relevant attributes (cast, genre, IMDb ratings) is provided.
2. A similarity matrix is generated using TF-IDF to analyze textual features such as the cast and genre of movies.
3. The system takes an input movie and calculates the similarity between it and other movies in the dataset.
4. The five most similar movies are recommended based on the calculated similarity scores.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
