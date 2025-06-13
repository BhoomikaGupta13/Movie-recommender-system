# 🎬 Movie Recommender System

## Overview

This project delivers a comprehensive, full-stack movie recommendation system designed to provide personalized movie suggestions. Leveraging data scraped from IMDb, the system utilizes a PostgreSQL database for robust storage and a FastAPI backend for efficient API communication. Frontend interactions are handled by a dynamic HTML, CSS (Tailwind CSS), and JavaScript interface, allowing users to browse movies, search, get recommendations based on movie titles or genres, and even rate movies. The current recommendation engine is built upon TF-IDF vectorization and Cosine Similarity to find highly relevant content based on movie attributes. User ratings are collected and stored but do not yet directly influence the recommendations.

## 🎥 Demo

A video demonstration of the project is uploaded, showcasing its features and responsiveness.

## 🌟 Features

- **Interactive Web Interface:** A user-friendly and responsive frontend built with HTML, Tailwind CSS, and vanilla JavaScript.
- **Comprehensive Movie Catalog:** Browse a wide range of movies with detailed information.
- **Smart Search Functionality:** Search for movies by title with real-time suggestions.
- **Flexible Recommendation Engine:**
  - *Movie-based Recommendations:* Discover movies similar to a given title using TF-IDF and Cosine Similarity (based on genre, plot summary, and director).
  - *Genre-based Recommendations:* Explore top-rated movies within specific genres.
- **User Rating System:** Users can submit ratings for movies, and the system displays individual user ratings and average community ratings for each film.
- **Robust Backend API:** A high-performance API developed using the FastAPI framework in Python, handling all data retrieval and recommendation logic.
- **Persistent Data Storage:** Movie data, including details, genres, directors, description and user ratings, is securely stored and managed in a PostgreSQL database.
- **Automated Data Scraping:** Includes a script to scrape and populate movie data from IMDb.[1]

## 🧰 Technologies Used

### 💻 Frontend

- **HTML5:** Structure and content of the web pages.
- **Tailwind CSS:** A utility-first CSS framework for rapid and responsive UI development.
- **JavaScript (Vanilla JS):** Dynamic client-side logic, API interactions (using fetch), and DOM manipulation.
- **Font Awesome:** For various icons used throughout the interface.

### ⚙️ Backend

- **Python 3.x:** The primary programming language.
- **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
- **Pandas:** Data manipulation and analysis, especially for preparing movie data.
- **Scikit-learn:** Used for TfidfVectorizer and cosine_similarity to build the recommendation model.
- **Psycopg2-binary:** PostgreSQL database adapter for Python.
- **NLTK (Natural Language Toolkit):** Used for text preprocessing (e.g., tokenization) which is essential before applying TF-IDF vectorization.
- **Uvicorn:** ASGI server for running the FastAPI application.
- **Pydantic:** For data validation and settings management with Python type hints.

### 🗄️ Database

- **PostgreSQL:** A powerful, open-source relational database system for storing movie and user rating data.
- **PgAdmin:** A popular administration and development platform for PostgreSQL.

### 🌐 Data Acquisition & Processing

- **Requests:** For making HTTP requests to fetch web content.
- **Beautiful Soup:** For parsing HTML and XML documents to extract data from IMDb.
- **Selenium:** For automating web browser interaction to scrape dynamic content from IMDb.
- **Webdriver Manager (Chrome):** Simplifies the management and installation of browser drivers for Selenium.[1]

## 🧱 Project Structure

- **main.py:** The FastAPI application, defining API endpoints for movies, search, various recommendation methods, and user ratings. It also handles the loading and initialization of the recommendation model.
- **app.js:** Frontend JavaScript responsible for fetching data from the FastAPI backend, rendering movie cards, handling search/recommendation logic (including type toggling for movie name vs. genre), and managing modal interactions and user ratings.
- **index.html:** The main HTML file for the web interface, integrating Tailwind CSS and linking app.js.
- **styles.css:** Custom CSS for additional styling and overrides.
- **data_scrapping.py:** Script to scrape movie details from IMDb using requests, BeautifulSoup, and Selenium, and then insert/update them into the PostgreSQL database.
- **database.py:** Contains the MovieDatabase class for connecting to PostgreSQL, creating tables, and performing CRUD operations for movie and related data (genres, directors, actors), including user ratings.[1]

## 🔄 Data Pipeline

- **Scraping:** IMDb Top 250 and detailed movie pages are scraped for metadata using Selenium and BeautifulSoup, with multi-threading for efficiency.
- **Cleaning & Validation:** Data is validated (e.g., year, rating, runtime) and cleaned (missing values, type conversions).
- **Database Ingestion:** Cleaned data is inserted into a normalized PostgreSQL schema with tables for movies, genres, directors, actors, and ratings.
- **Model Training:** On API startup, movies are loaded, and a TF-IDF matrix is built on combined features (genre, plot, director). Cosine similarity matrix is computed for recommendations.
- **Serving:** FastAPI exposes endpoints for querying movies, searching, getting recommendations, and submitting/viewing ratings.
- **Frontend Integration:** The HTML/CSS/JS frontend communicates with the API, displaying results and collecting user input.[1]

## 🧠 Recommendation Model

The core of the recommendation system currently uses a content-based filtering approach:

- **TF-IDF Vectorization:** Movie features (plot summaries, genres, directors) are transformed into numerical TF-IDF vectors. This technique reflects the importance of a word in a document relative to a collection of documents.
- **Cosine Similarity:** The similarity between movies is calculated using the cosine of the angle between their TF-IDF vectors. A higher cosine similarity score indicates greater similarity.[1]


## 📡 API Endpoints

The FastAPI backend exposes the following key endpoints:

- **GET `/movies`**  
  Retrieve a list of all movies.

- **GET `/search`**  
  Search for movies by title.

- **GET `/recommendations`**  
  Get movie recommendations based on a title.

- **POST `/recommendations/by-body`**  
  Get movie recommendations based on a title provided in the request body.

- **GET `/genres`**  
  Retrieve all available movie genres.

- **POST `/ratings`**  
  Submit a rating for a specific movie.


## 🎯 Conclusion
This project stands as a fully functional prototype of a movie recommendation platform, bringing together advanced data handling, smart recommendation logic, and a sleek web interface. While it's not currently deployed for public use, it demonstrates the complete workflow—from scraping and storing data to serving intelligent recommendations. 🍿  
Perfect as a foundation for future deployment, collaboration, or academic showcase. 🚀

