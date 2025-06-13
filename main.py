import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import psycopg2
from psycopg2 import extras
from fastapi import FastAPI, HTTPException, Query, Body, status, Path
from pydantic import BaseModel, Field
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import nltk
from collections import Counter 
import os

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="An API for getting movie data, searching, and recommendations.",
    version="1.0.0"
)

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Enter your connection string
DATABASE_URL = os.environ.get("DATABASE_URL", "connection_string")

def get_db_connection():
    """Establishes and returns a database connection using the DATABASE_URL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Global variables for DataFrame and cosine similarity matrix
df = None
cosine_sim = None
tfidf_vectorizer = None
unique_genres_cached: Set[str] = set() 
GENRE_MIN_MOVIE_THRESHOLD = 5 

# ThreadPoolExecutor for running synchronous DB operations in a separate thread
executor = ThreadPoolExecutor(max_workers=5)

def clean_text(text: str) -> str:
    """Cleans text by lowercasing and removing non-alphanumeric characters."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def load_model_data_from_db_sync():
    """
    Synchronously loads movie data from the PostgreSQL database, processes it,
    and calculates TF-IDF matrix and cosine similarity.
    This function is synchronous because psycopg2 is synchronous.
    Also, it now filters genres based on a minimum movie count.
    """
    global df, cosine_sim, tfidf_vectorizer, unique_genres_cached
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("Failed to establish database connection.")

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT movie_id, title, genre, plot_summary, release_year, imdb_rating, duration_minutes, poster_url, director FROM combined_movies")

        movies_data_raw = cur.fetchall()
        movies_data = [dict(row) for row in movies_data_raw]

        df_loaded = pd.DataFrame(movies_data)

        if df_loaded.empty:
            print("No data found in the 'combined_movies' table. Please ensure your table has data.")
            return None
        df_loaded.fillna('', inplace=True)

        df_loaded['genre'] = df_loaded['genre'].astype(str)
        df_loaded['plot_summary'] = df_loaded['plot_summary'].astype(str)
        df_loaded['director'] = df_loaded['director'].astype(str)

        # Include director in combined_features for TF-IDF
        df_loaded['combined_features'] = df_loaded['genre'] + ' ' + df_loaded['plot_summary'] + ' ' + df_loaded['director']
        df_loaded['combined_features'] = df_loaded['combined_features'].apply(clean_text)

        tfidf_vectorizer_loaded = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.85,
            max_features=10000
        )
        tfidf_matrix = tfidf_vectorizer_loaded.fit_transform(df_loaded['combined_features'])
        cosine_sim_loaded = cosine_similarity(tfidf_matrix, tfidf_matrix)

        df = df_loaded
        tfidf_vectorizer = tfidf_vectorizer_loaded
        cosine_sim = cosine_sim_loaded

        # Populate unique_genres_cached with filtering
        genre_counts = Counter()
        for genres_str in df['genre'].dropna():
            for genre in genres_str.split(','):
                cleaned_genre = genre.strip().title()
                if cleaned_genre:
                    genre_counts[cleaned_genre] += 1

        # Filter genres based on the defined threshold
        unique_genres_cached.clear() # Clear any previous cached genres
        for genre, count in genre_counts.items():
            if count >= GENRE_MIN_MOVIE_THRESHOLD:
                unique_genres_cached.add(genre)

        print(f"Model data loaded successfully from PostgreSQL! {len(unique_genres_cached)} main genres cached.")
        return True
    except Exception as e:
        print(f"An error occurred during model data loading from database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_recommendations_for_app(movie_title: str, cosine_sim_matrix: np.ndarray, df_data: pd.DataFrame, num_recommendations: int = 10) -> List[dict]:
    """Generates movie recommendations based on cosine similarity."""
    if df_data is None:
        return []

    cleaned_movie_title = clean_text(movie_title)

    matches = df_data[df_data['title'].astype(str).str.lower().apply(lambda x: clean_text(re.sub(r'^\d+\.\s*', '', x))).str.contains(cleaned_movie_title, case=False, na=False)]

    if matches.empty:
        matches = df_data[df_data['title'].astype(str).str.contains(movie_title, case=False, na=False)]

    if matches.empty:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []

    movie_index = matches.index[0]

    sim_scores = list(enumerate(cosine_sim_matrix[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    recommendations = []
    for i, score in sim_scores:
        row = df_data.iloc[i]
        clean_title = re.sub(r'^\d+\.\s*', '', str(row['title'])).strip()

        poster_url = row['poster_url'] if pd.notna(row['poster_url']) and row['poster_url'] != 'N/A' else f"https://placehold.co/150x225/A0A0A0/FFFFFF?text={clean_title[:15]}"

        movie = {
            'id': int(row['movie_id']),
            'title': clean_title,
            'year': int(row['release_year']) if pd.notna(row['release_year']) and row['release_year'] != '' else None,
            'rating': float(row['imdb_rating']) / 2 if pd.notna(row['imdb_rating']) and row['imdb_rating'] != '' else None,
            'genre': ', '.join([g.strip() for g in str(row['genre']).split(',')[:6]]) if pd.notna(row['genre']) and str(row['genre']).strip() else 'Unknown',
            'duration': int(row['duration_minutes']) if pd.notna(row['duration_minutes']) and row['duration_minutes'] != '' else 0,
            'overview': row['plot_summary'],
            'poster_url': poster_url,
            'director': row['director'] if pd.notna(row['director']) and row['director'] != '' else 'Unknown',
            'similarity_score': float(score) if pd.notna(score) else 0.0
        }
        recommendations.append(movie)
    return recommendations

def get_recommendations_by_genre(genre_name: str, df_data: pd.DataFrame, num_recommendations: int = 10) -> List[dict]:
    """Generates movie recommendations based on a given genre."""
    if df_data is None:
        return []

    cleaned_genre_name_lower = clean_text(genre_name)

    genre_movies = df_data[
        df_data['genre'].astype(str).str.lower().apply(lambda x: cleaned_genre_name_lower in x)
    ]

    if genre_movies.empty:
        return []

    genre_movies['imdb_rating_numeric'] = pd.to_numeric(genre_movies['imdb_rating'], errors='coerce')
    genre_movies['release_year_numeric'] = pd.to_numeric(genre_movies['release_year'], errors='coerce')

    genre_movies_sorted = genre_movies.sort_values(
        by=['imdb_rating_numeric', 'release_year_numeric'],
        ascending=[False, False]
    ).dropna(subset=['imdb_rating_numeric', 'release_year_numeric'])

    recommendations = []
    for _, row in genre_movies_sorted.head(num_recommendations).iterrows():
        clean_title = re.sub(r'^\d+\.\s*', '', str(row['title'])).strip()
        poster_url = row['poster_url'] if pd.notna(row['poster_url']) and row['poster_url'] != 'N/A' else f"https://placehold.co/150x225/A0A0A0/FFFFFF?text={clean_title[:15]}"

        movie = {
            'id': int(row['movie_id']),
            'title': clean_title,
            'year': int(row['release_year']) if pd.notna(row['release_year']) and row['release_year'] != '' else None,
            'rating': float(row['imdb_rating']) / 2 if pd.notna(row['imdb_rating']) and row['imdb_rating'] != '' else None,
            'genre': ', '.join([g.strip() for g in str(row['genre']).split(',')[:6]]) if pd.notna(row['genre']) and str(row['genre']).strip() else 'Unknown',
            'duration': int(row['duration_minutes']) if pd.notna(row['duration_minutes']) and row['duration_minutes'] != '' else 0,
            'overview': row['plot_summary'],
            'poster_url': poster_url,
            'director': row['director'] if pd.notna(row['director']) and row['director'] != '' else 'Unknown',
            'similarity_score': 1.0
        }
        recommendations.append(movie)
    return recommendations

# --- Pydantic Models for Request and Response ---
class MovieBase(BaseModel):
    id: int = Field(..., description="Unique movie ID")
    title: str = Field(..., example="The Lord of the Rings: The Return of the King")
    year: Optional[int] = Field(None, example=2003)
    rating: Optional[float] = Field(None, example=4.5, description="IMDb rating divided by 2 (out of 5 stars)")
    genre: str = Field(..., example="Action, Adventure, Fantasy")
    duration: int = Field(..., example=201, description="Duration in minutes")
    overview: str = Field(..., example="Gandalf and Aragorn lead the World of Men...")
    poster_url: str = Field(..., example="https://m.media-amazon.com/images/M/MV5B...jpg")
    director: Optional[str] = Field(None, example="Peter Jackson")

class MovieRecommendation(MovieBase):
    similarity_score: float = Field(..., example=0.85, description="Cosine similarity score with the input movie")

class MovieSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, example="Lord of the Rings")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of search results")

class MovieRecommendationRequest(BaseModel):
    title: str = Field(..., min_length=1, example="The Lord of the Rings: The Return of the King")
    num_recommendations: int = Field(10, ge=1, le=20, description="Number of recommendations to retrieve")

class FlexibleRecommendationRequest(BaseModel):
    query: str = Field(..., description="The movie title or genre name to search by.", example="Action" if "genre" else "Inception")
    search_by: str = Field(..., description="Specify 'movie_name' or 'genre'.", example="genre")
    num_recommendations: int = Field(10, ge=1, le=20, description="Number of recommendations to retrieve")


class MovieResponse(BaseModel):
    movies: List[MovieBase]

class RecommendationsResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    message: Optional[str] = None

class MessageResponse(BaseModel):
    message: str

class UserRatingRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user giving the rating")
    movie_id: int = Field(..., description="ID of the movie being rated")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating given by the user (0.5 to 5.0)")

class UserRatingResponse(BaseModel):
    rating_id: int
    user_id: int
    movie_id: int
    rating: float
    timestamp: str

class UserRatingsListResponse(BaseModel):
    ratings: List[UserRatingResponse]


# --- FastAPI Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """
    Load the movie model data from the database when the FastAPI application starts.
    This runs synchronously within a thread pool to avoid blocking the main event loop.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("NLTK 'punkt' tokenizer downloaded.")

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("NLTK 'punkt_tab' resource not found. Downloading...")
        nltk.download('punkt_tab')
        print("NLTK 'punkt_tab' resource downloaded.")

    print("Application startup: Loading movie data...")
    success = await asyncio.get_event_loop().run_in_executor(executor, load_model_data_from_db_sync)
    if not success:
        print("Failed to load model data. API endpoints relying on it may not function.")

@app.on_event("shutdown")
def shutdown_event():
    """Shutdown the ThreadPoolExecutor when the application closes."""
    executor.shutdown(wait=True)
    print("Application shutdown: ThreadPoolExecutor closed.")


# --- API Endpoints ---

@app.get("/movies", response_model=MovieResponse, summary="Get All Movies")
async def get_all_movies(limit: Optional[int] = Query(100, ge=1, description="Limit the number of movies returned. Set to a very large number or omit for all.")):
    """
    Retrieves a sample of all movies from the dataset.
    This is useful for initial display or Browse.
    """
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Movie data not loaded. Please try again later."
        )

    actual_limit = limit if limit is not None else len(df)

    movies_list = await asyncio.get_event_loop().run_in_executor(executor, lambda: [
        {
            'id': int(row['movie_id']),
            'title': re.sub(r'^\d+\.\s*', '', str(row['title'])).strip(),
            'year': int(row['release_year']) if pd.notna(row['release_year']) and row['release_year'] != '' else None,
            'rating': float(row['imdb_rating']) / 2 if pd.notna(row['imdb_rating']) and row['imdb_rating'] != '' else None,
            'genre': ', '.join([g.strip() for g in str(row['genre']).split(',')[:6]]) if pd.notna(row['genre']) and str(row['genre']).strip() else 'Unknown',
            'duration': int(row['duration_minutes']) if pd.notna(row['duration_minutes']) and row['duration_minutes'] != '' else 0,
            'overview': row['plot_summary'],
            'poster_url': row['poster_url'] if pd.notna(row['poster_url']) and row['poster_url'] != 'N/A' else f"https://placehold.co/150x225/A0A0A0/FFFFFF?text={re.sub(r'^\d+\.\s*', '', str(row['title'])).strip()[:15]}",
            'director': row['director'] if pd.notna(row['director']) and row['director'] != '' else 'Unknown'
        }
        for _, row in df.head(actual_limit).iterrows()
    ])

    return {"movies": movies_list}

@app.get("/search", response_model=List[MovieBase], summary="Search Movies by Title")
async def search_movies_api(
    q: str = Query(..., min_length=1, description="Movie title search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of search results")
):
    """
    Searches for movies by title, returning matches that start with or contain the query.
    """
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Movie data not loaded. Please try again later."
        )

    cleaned_query_for_validation = re.sub(r'[^a-zA-Z0-9]', '', q).lower()
    if not cleaned_query_for_validation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a valid movie name (letters or numbers)."
        )

    search_term_lower = q.lower()

    search_results_df = await asyncio.get_event_loop().run_in_executor(executor, lambda: _perform_search(df, search_term_lower, limit))

    if search_results_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No movies found matching '{q}'. Please try a different name."
        )

    movies_list = await asyncio.get_event_loop().run_in_executor(executor, lambda: [
        {
            'id': int(row['movie_id']),
            'title': re.sub(r'^\d+\.\s*', '', str(row['title'])).strip(),
            'year': int(row['release_year']) if pd.notna(row['release_year']) and row['release_year'] != '' else None,
            'rating': float(row['imdb_rating']) / 2 if pd.notna(row['imdb_rating']) and row['imdb_rating'] != '' else None,
            'genre': ', '.join([g.strip() for g in str(row['genre']).split(',')[:6]]) if pd.notna(row['genre']) and str(row['genre']).strip() else 'Unknown',
            'duration': int(row['duration_minutes']) if pd.notna(row['duration_minutes']) and row['duration_minutes'] != '' else 0,
            'overview': row['plot_summary'],
            'poster_url': row['poster_url'] if pd.notna(row['poster_url']) and row['poster_url'] != 'N/A' else f"https://placehold.co/150x225/A0A0A0/FFFFFF?text={re.sub(r'^\d+\.\s*', '', str(row['title'])).strip()[:15]}",
            'director': row['director'] if pd.notna(row['director']) and row['director'] != '' else 'Unknown'
        }
        for _, row in search_results_df.iterrows()
    ])
    return movies_list

def _perform_search(df_data, search_term_lower, limit):
    temp_df = df_data.copy()
    temp_df['temp_title_lower'] = temp_df['title'].astype(str).str.lower()
    temp_df['clean_search_title'] = temp_df['temp_title_lower'].str.replace(r'^\d+\.\s*', '', regex=True)

    startswith_results = temp_df[temp_df['clean_search_title'].str.startswith(search_term_lower)]
    contains_results = temp_df[temp_df['clean_search_title'].str.contains(search_term_lower)]

    return pd.concat([startswith_results, contains_results]).drop_duplicates().head(limit)


@app.get("/recommendations", response_model=RecommendationsResponse, summary="Get Movie Recommendations by Title (GET)")
async def recommend_movies_api_get(
    title: str = Query(..., min_length=1, description="Title of the movie to get recommendations for"),
    num_recommendations: int = Query(10, ge=1, le=20, description="Number of recommendations to retrieve")
):
    """
    Provides movie recommendations based on a given movie title via a GET request.
    """
    if df is None or cosine_sim is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model not initialized. Please try again later."
        )

    cleaned_title_for_validation = re.sub(r'[^a-zA-Z0-9]', '', title).lower()
    if not cleaned_title_for_validation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a valid movie name (letters or numbers) for recommendations."
        )

    recommendations_list = await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: get_recommendations_for_app(title, cosine_sim, df, num_recommendations)
    )

    if not recommendations_list:
        return {
            "recommendations": [],
            "message": f"No recommendations found for '{title}'. The movie might not be in the dataset, or no similar movies were found."
        }

    return {"recommendations": recommendations_list, "message": "Recommendations found."}

@app.post("/recommendations/by-body", response_model=RecommendationsResponse, summary="Get Movie Recommendations by Title (POST)")
async def recommend_movies_by_body_api(
    request: MovieRecommendationRequest = Body(..., description="Details for movie recommendation")
):
    """
    Provides movie recommendations based on a given movie title received in the request body.
    This demonstrates handling a POST request with a Pydantic request model.
    """
    if df is None or cosine_sim is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model not initialized. Please try again later."
        )

    cleaned_title_for_validation = re.sub(r'[^a-zA-Z0-9]', '', request.title).lower()
    if not cleaned_title_for_validation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a valid movie name (letters or numbers) for recommendations."
        )

    recommendations_list = await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: get_recommendations_for_app(request.title, cosine_sim, df, request.num_recommendations)
    )

    if not recommendations_list:
        return {
            "recommendations": [],
            "message": f"No recommendations found for '{request.title}'."
        }

    return {"recommendations": recommendations_list, "message": "Recommendations found."}

@app.get("/api-status", response_model=MessageResponse, summary="Check API Status")
async def get_api_status():
    """
    A simple endpoint to check the status of the API and model loading.
    """
    status_message = "API is running. "
    if df is not None and cosine_sim is not None:
        status_message += "Movie recommendation model loaded successfully."
    else:
        status_message += "Movie recommendation model not yet loaded or failed to load."
    return {"message": status_message}

@app.post("/ratings", response_model=UserRatingResponse, status_code=status.HTTP_201_CREATED, summary="Submit a Movie Rating")
async def submit_movie_rating(rating_request: UserRatingRequest = Body(...)):
    """
    Allows a user to submit a rating for a specific movie.
    If a rating for the same user and movie already exists, it will be updated.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not connect to the database.")

        cur = conn.cursor()

        cur.execute("SELECT 1 FROM users WHERE user_id = %s", (rating_request.user_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with ID {rating_request.user_id} not found.")

        cur.execute("SELECT 1 FROM combined_movies WHERE movie_id = %s", (rating_request.movie_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Movie with ID {rating_request.movie_id} not found.")

        query = """
        INSERT INTO user_ratings (user_id, movie_id, rating)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, movie_id) DO UPDATE SET
            rating = EXCLUDED.rating,
            timestamp = CURRENT_TIMESTAMP
        RETURNING rating_id, user_id, movie_id, rating, timestamp;
        """
        cur.execute(query, (rating_request.user_id, rating_request.movie_id, rating_request.rating))

        conn.commit()

        result = cur.fetchone()
        if result:
            return UserRatingResponse(
                rating_id=result[0],
                user_id=result[1],
                movie_id=result[2],
                rating=float(result[3]),
                timestamp=result[4].isoformat()
            )
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve rating after insert/update.")

    except psycopg2.Error as e:
        conn.rollback()
        print(f"Database error during rating submission: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/users/{user_id}/ratings", response_model=UserRatingsListResponse, summary="Get Ratings by User")
async def get_user_ratings(user_id: int = Path(..., description="ID of the user to retrieve ratings for")):
    """
    Retrieves all movie ratings submitted by a specific user.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not connect to the database.")

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT rating_id, user_id, movie_id, rating, timestamp FROM user_ratings WHERE user_id = %s", (user_id,))

        ratings_data = cur.fetchall()

        if not ratings_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No ratings found for user ID {user_id}.")

        ratings_list = [
            UserRatingResponse(
                rating_id=row['rating_id'],
                user_id=row['user_id'],
                movie_id=row['movie_id'],
                rating=float(row['rating']),
                timestamp=row['timestamp'].isoformat()
            )
            for row in ratings_data
        ]
        return UserRatingsListResponse(ratings=ratings_list)

    except Exception as e:
        print(f"Error fetching user ratings: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve user ratings: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/movies/{movie_id}/average-rating", response_model=dict, summary="Get Average Rating for a Movie")
async def get_movie_average_rating(movie_id: int = Path(..., description="ID of the movie to get the average rating for")):
    """
    Calculates and returns the average user rating for a specific movie.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not connect to the database.")

        cur = conn.cursor()
        cur.execute("SELECT AVG(rating) FROM user_ratings WHERE movie_id = %s", (movie_id,))

        avg_rating = cur.fetchone()[0]

        if avg_rating is None:
            return {"movie_id": movie_id, "average_rating": None, "message": "No ratings found for this movie."}

        return {"movie_id": movie_id, "average_rating": round(float(avg_rating), 2)}

    except Exception as e:
        print(f"Error fetching movie average rating: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve average rating: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/genres", response_model=List[str], summary="Get All Available Movie Genres")
async def get_all_genres_api():
    """
    Retrieves a list of all unique movie genres available in the dataset
    """
    global unique_genres_cached
    if not unique_genres_cached:
        if df is not None:
            genre_counts = Counter()
            for genres_str in df['genre'].dropna():
                for genre in genres_str.split(','):
                    cleaned_genre = genre.strip().title()
                    if cleaned_genre:
                        genre_counts[cleaned_genre] += 1
            unique_genres_cached.clear()
            for genre, count in genre_counts.items():
                if count >= GENRE_MIN_MOVIE_THRESHOLD:
                    unique_genres_cached.add(genre)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Genres not loaded. Please try again later or ensure movie data is loaded."
            )
    return sorted(list(unique_genres_cached))

@app.post("/flexible-recommendations", response_model=RecommendationsResponse, summary="Get Movie Recommendations by Title or Genre")
async def flexible_recommendations_api(
    request: FlexibleRecommendationRequest = Body(..., description="Details for movie recommendation by title or genre")
):
    """
    Provides movie recommendations based on a given movie title or genre.
    """
    if df is None or (request.search_by == "movie_name" and cosine_sim is None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model not initialized. Please try again later."
        )

    recommendations_list = []
    message = ""

    cleaned_query_for_validation = re.sub(r'[^a-zA-Z0-9]', '', request.query).lower()
    if not cleaned_query_for_validation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Please enter a valid {request.search_by.replace('_', ' ')} (letters or numbers)."
        )

    if request.search_by == "movie_name":
        message = f"Recommendations for movies similar to '{request.query}':"
        recommendations_list = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: get_recommendations_for_app(request.query, cosine_sim, df, request.num_recommendations)
        )
    elif request.search_by == "genre":
        message = f"Top movies in the '{request.query}' genre:"
        recommendations_list = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: get_recommendations_by_genre(request.query, df, request.num_recommendations)
        )
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid 'search_by' type. Must be 'movie_name' or 'genre'.")

    if not recommendations_list:
        return {
            "recommendations": [],
            "message": f"No recommendations found for '{request.query}' by {request.search_by.replace('_', ' ')}. Please try a different query."
        }

    return {"recommendations": recommendations_list, "message": message}
