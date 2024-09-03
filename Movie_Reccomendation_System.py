import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import requests
import streamlit as st

# Load the data
df = pd.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv")
features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
x = features['Movie_Genre'] + ' ' + features['Movie_Keywords'] + ' ' + features['Movie_Tagline'] + ' ' + features['Movie_Cast'] + ' ' + features['Movie_Director']

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(x)
Similarity_Score = cosine_similarity(x)

# Directly setting the API key for testing
API_KEY = "4be41a6b"

def fetch_movie_details(movie_title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={movie_title}"
    response = requests.get(url)
    data = response.json()
    poster_url = data.get('Poster', 'https://via.placeholder.com/150x225?text=Poster+Not+Found')
    plot = data.get('Plot', 'Plot not available')
    year = data.get('Year', 'Year not available')
    imdb_rating = data.get('imdbRating', 'Rating not available')
    return poster_url, plot, year, imdb_rating

# Streamlit app layout
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎥")

# Add a background image and set global text color
background_image_url = "https://media.licdn.com/dms/image/D5612AQGy6sM0SJAdxg/article-cover_image-shrink_720_1280/0/1693150322893?e=2147483647&v=beta&t=tmyCkhGahTKcBOOftyXZLhkLjtUIkqio94iGE3Y670E"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        color: white; /* Set default text color to white */
    }}
    .stButton > button {{
        color: white;
        background-color: red; /* Set button color to red */
    }}
    .stTextInput > input {{
        color: white; /* Set text input color to white */
    }}
    .stMarkdown {{
        color: white; /* Ensure markdown text color is white */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Set the title with white color
st.markdown("<h1 style='color:white;'>🎥 Movie Recommendation System 🎬</h1>", unsafe_allow_html=True)

with st.form(key='movie_form'):
    movie_name = st.text_input("Enter your favorite movie name:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if movie_name:
        list_of_all_titles = df['Movie_Title'].tolist()
        close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if not close_match:
            st.error("No close match found. Please try again with a different movie name.")
        else:
            match = close_match[0]
            index_of_movie = df[df.Movie_Title == match]['Movie_ID'].values[0]
            recommendation_score = list(enumerate(Similarity_Score[index_of_movie]))
            sorted_similar_movies = sorted(recommendation_score, key=lambda x: x[1], reverse=True)
            
            st.markdown(f"<h2 style='color:white;'>Top 5 similar movies to '{movie_name}':</h2>", unsafe_allow_html=True)
            
            for i, movie in enumerate(sorted_similar_movies[:5]):
                index = movie[0]
                title_from_index = df[df.Movie_ID == index]['Movie_Title'].values[0]
                poster_url, plot, year, imdb_rating = fetch_movie_details(title_from_index)
                
                col1, col2 = st.columns([1, 2])  # Adjusted column sizes for better layout
                with col1:
                    st.image(poster_url, use_column_width=True, width=120)  # Adjusted width for posters
                with col2:
                    st.markdown(f"<h3 style='color:white;'>{title_from_index}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:white;'>Year: {year}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:white;'>IMDB Rating: {imdb_rating}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:white;'>Plot: {plot}</p>", unsafe_allow_html=True)
    else:
        st.error("Please enter a movie name.")
