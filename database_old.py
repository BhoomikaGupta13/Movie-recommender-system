import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import List, Dict, Optional
import time

class MovieDatabase:
    # A property to hold messages from add_movie for better status reporting in main.py
    last_message: str = ""

    def __init__(self, dbname: str, user: str, password: str, host: str = 'localhost', port: int = 5432):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None # Initialize conn to None
        self._connect_and_create_db() # Connects and creates db if needed, then sets self.conn
        self.create_tables() # This call will now be more robust

    def _connect_and_create_db(self):
        # Connect to default database (postgres) to create the target db if it doesn't exist
        try:
            conn_no_db = psycopg2.connect(
                dbname='postgres',
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            conn_no_db.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor_no_db = conn_no_db.cursor()

            # Check if imdb_db exists
            cursor_no_db.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.dbname}'")
            exists = cursor_no_db.fetchone()
            if not exists:
                print(f"Database '{self.dbname}' does not exist. Creating it...")
                # Use sql.Identifier for safe database name creation
                cursor_no_db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.dbname)))
                print(f"Database '{self.dbname}' created successfully.")
            else:
                print(f"Database '{self.dbname}' already exists.")

            cursor_no_db.close()
            conn_no_db.close()

            # Now connect to the actual imdb_db
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.conn.autocommit = False # Set autocommit to False for explicit transactions

        except psycopg2.OperationalError as e:
            print(f"Error connecting to PostgreSQL: {e}")
            print("Please ensure PostgreSQL is running and connection details are correct.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred during database connection/creation: {e}")
            exit()

    def execute_query(self, query: str, params: tuple = None, commit: bool = False) -> List[Dict]:
        """
        Executes a SQL query and returns results if available.
        Optionally commits the transaction.
        """
        if not self.conn:
            print("Error: Database connection not established.")
            return []

        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            try:
                # Uncomment the line below to see all SQL queries in the console for intense debugging
                # print(f"DEBUG: Executing query: {cursor.mogrify(query, params).decode('utf-8')}")
                cursor.execute(query, params or ())
                if commit:
                    self.conn.commit() # Explicit commit
                if cursor.description:
                    return cursor.fetchall()
                # For DDL/DML statements like INSERT/UPDATE/ALTER, fetchall() might be empty
                # We can check cursor.rowcount for affected rows if needed, but for simplicity, returning empty list
                return []
            except psycopg2.Error as e:
                print(f"Error executing query: {e}")
                self.conn.rollback() # Rollback on error
                return []
            except Exception as e:
                print(f"An unexpected error occurred during query execution: {e}")
                self.conn.rollback()
                return []

    def create_tables(self):
        """
        Creates all necessary tables and indexes if they don't exist,
        ensuring 'vote_count' and 'poster_url' columns have proper defaults
        and converting any existing NULLs to their default non-NULL values.
        """
        # Step 1: Ensure the 'movies' table and its critical columns exist and are set up.
        # This block is executed first and committed immediately to guarantee schema.
        setup_movies_table_sql = """
        CREATE TABLE IF NOT EXISTS combined_movies (
            movie_id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            release_year INTEGER CHECK (release_year BETWEEN 1888 AND 2025),
            duration_minutes INTEGER CHECK (duration_minutes > 0),
            imdb_rating NUMERIC(3,1) CHECK (imdb_rating BETWEEN 0 AND 10),
            vote_count BIGINT DEFAULT 0,
            plot_summary TEXT DEFAULT 'No description available',
            poster_url VARCHAR(255) DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (title, release_year)
        );

        -- Add columns if they don't exist and set defaults, for pre-existing tables without these columns
        ALTER TABLE combined_movies ADD COLUMN IF NOT EXISTS vote_count BIGINT;
        ALTER TABLE combined_movies ALTER COLUMN vote_count SET DEFAULT 0;
        ALTER TABLE combined_movies ALTER COLUMN vote_count DROP NOT NULL; -- Ensure NULLs can be updated

        ALTER TABLE combined_movies ADD COLUMN IF NOT EXISTS poster_url VARCHAR(255);
        ALTER TABLE combined_movies ALTER COLUMN poster_url SET DEFAULT '';
        ALTER TABLE combined_movies ALTER COLUMN poster_url DROP NOT NULL; -- Ensure NULLs can be updated
        """
        try:
            self.execute_query(setup_movies_table_sql, commit=True)
            print("DEBUG (create_tables): 'combined_movies' table and essential columns/defaults ensured.")
        except Exception as e:
            print(f"CRITICAL ERROR (create_tables): Failed to set up 'combined_movies' table or its columns: {e}")
            raise # Re-raise to stop execution if essential table setup fails

        # Step 2: Explicitly convert any existing NULLs in vote_count and poster_url to their defaults.
        # This is vital for older data where these columns might be NULL.
        update_nulls_sql = """
        UPDATE combined_movies SET vote_count = 0 WHERE vote_count IS NULL;
        UPDATE combined_movies SET poster_url = '' WHERE poster_url IS NULL;
        """
        try:
            self.execute_query(update_nulls_sql, commit=True)
            print("DEBUG (create_tables): Attempted to convert existing NULL vote_count and poster_url values to defaults (0 and '').")
        except Exception as e:
            print(f"WARNING (create_tables): Failed to update existing NULLs for vote_count/poster_url: {e}")

        # Step 3: Create other related tables and indexes.
        remaining_schema_sql = """
        CREATE TABLE IF NOT EXISTS genres (
            genre_id SERIAL PRIMARY KEY,
            name VARCHAR(50) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_genres (
            movie_id INTEGER REFERENCES movies(movie_id) ON DELETE CASCADE,
            genre_id INTEGER REFERENCES genres(genre_id) ON DELETE CASCADE,
            PRIMARY KEY (movie_id, genre_id)
        );

        CREATE TABLE IF NOT EXISTS directors (
            director_id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_directors (
            movie_id INTEGER REFERENCES combined_movies(movie_id) ON DELETE CASCADE,
            director_id INTEGER REFERENCES directors(director_id) ON DELETE CASCADE,
            PRIMARY KEY (movie_id, director_id)
        );

        CREATE TABLE IF NOT EXISTS actors (
            actor_id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_actors (
            movie_id INTEGER REFERENCES combined_movies(movie_id) ON DELETE CASCADE,
            actor_id INTEGER REFERENCES actors(actor_id) ON DELETE CASCADE,
            PRIMARY KEY (movie_id, actor_id)
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ratings (
            rating_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            movie_id INTEGER REFERENCES combined_movies(movie_id) ON DELETE CASCADE,
            rating NUMERIC(2,1) CHECK (rating BETWEEN 0.5 AND 5.0),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, movie_id)
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_movies_title ON combined_movies(title);
        CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);
        CREATE INDEX IF NOT EXISTS idx_genres_name ON genres(name);
        CREATE INDEX IF NOT EXISTS idx_directors_name ON directors(name);
        CREATE INDEX IF NOT EXISTS idx_actors_name ON actors(name);
        """
        try:
            self.execute_query(remaining_schema_sql, commit=True)
            print("DEBUG (create_tables): Auxiliary tables and indexes ensured.")
        except Exception as e:
            print(f"ERROR (create_tables): Failed to set up auxiliary tables or indexes: {e}")
            # Do not re-raise, as movies table might still be usable

        print("Ensured normalized tables and unique constraints exist.")


    def _get_or_create_id(self, table_name: str, name_column: str, item_name: str) -> Optional[int]:
        """
        Helper to get the ID of an item (genre, director, actor) or create it if it doesn't exist.
        """
        id_column_name = f"{table_name[:-1]}_id" if table_name.endswith('s') else f"{table_name}_id"
        if table_name == 'actors':
            id_column_name = 'actor_id'

        query = sql.SQL("""
            INSERT INTO {table} ({name_col})
            VALUES (%s)
            ON CONFLICT ({name_col}) DO UPDATE SET {name_col} = EXCLUDED.{name_col}
            RETURNING {id_col}
        """).format(
            table=sql.Identifier(table_name),
            name_col=sql.Identifier(name_column),
            id_col=sql.Identifier(id_column_name)
        )

        result = self.execute_query(query, (item_name,))
        return result[0][id_column_name] if result else None

    def add_movie(self, title: str, release_year: int, duration: int,
                    imdb_rating: float, vote_count: Optional[int],
                    plot: str, poster_url: str, # poster_url is now a required parameter
                    genre: str = None, director: str = None, stars: str = None) -> str: # Return status string

        movie_id = None
        self.last_message = "" # Reset message

        # Normalize vote_count and poster_url before passing to database
        # Convert None to 0 for vote_count, or ensure it's an int
        vote_count_val = int(vote_count) if vote_count is not None else 0
        # Convert None to empty string for poster_url, or ensure it's a string
        poster_url_val = poster_url if poster_url is not None else ''

        # NEW DEBUG: Print values received by add_movie
        print(f"DB Function: add_movie received for '{title}' (Year: {release_year})")
        print(f"  -> Duration: {duration}, Rating: {imdb_rating}, Vote Count: {vote_count_val} (Type: {type(vote_count_val)})")
        print(f"  -> Plot: {plot[:50]}..., Poster: '{poster_url_val}', Genre: {genre}, Director: {director}, Stars: {stars}")

        try:
            # First, try to get the existing movie_id and current data to handle status correctly
            select_existing_movie_query = """
            SELECT movie_id, vote_count, poster_url FROM combined_movies WHERE title = %s AND release_year = %s
            """
            existing_movie_result = self.execute_query(select_existing_movie_query, (title, release_year))

            existing_movie_id = None
            existing_vote_count = None
            existing_poster_url = None
            if existing_movie_result:
                existing_movie_id = existing_movie_result[0]['movie_id']
                existing_vote_count = existing_movie_result[0]['vote_count']
                existing_poster_url = existing_movie_result[0]['poster_url']
                print(f"DEBUG (DB): Movie '{title}' (Year: {release_year}) found with existing ID: {existing_movie_id}. Existing Vote Count: {existing_vote_count}, Existing Poster URL: '{existing_poster_url}'")

            # Prepare values for insertion/update using the normalized values
            params = (title, release_year, duration, imdb_rating, vote_count_val, plot, poster_url_val)

            insert_or_update_movie_query = """
            INSERT INTO combined_movies (
                title, release_year, duration_minutes, imdb_rating,
                vote_count, plot_summary, poster_url
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (title, release_year) DO UPDATE SET
                duration_minutes = EXCLUDED.duration_minutes,
                imdb_rating = EXCLUDED.imdb_rating,
                vote_count = EXCLUDED.vote_count,
                plot_summary = EXCLUDED.plot_summary,
                poster_url = EXCLUDED.poster_url
            RETURNING movie_id
            """
            result = self.execute_query(insert_or_update_movie_query, params)

            if result:
                movie_id = result[0]['movie_id']
                # Check if the vote_count or poster_url was actually updated for an existing record
                # Compare the new normalized values with the existing database values
                if existing_movie_id: # If it existed
                    # For existing records, check if the *newly provided* values are different from *what was in the DB*
                    if existing_vote_count != vote_count_val or existing_poster_url != poster_url_val:
                        self.last_message = "updated"
                        print(f"DEBUG (DB): Movie '{title}' (ID: {movie_id}) updated. Old vote_count: {existing_vote_count}, New vote_count: {vote_count_val}. Old poster_url: '{existing_poster_url}', New poster_url: '{poster_url_val}'")
                    else:
                        self.last_message = "exists_no_change"
                        print(f"DEBUG (DB): Movie '{title}' (ID: {movie_id}) already exists, vote_count and poster_url unchanged.")
                else: # If it was a new insert
                    self.last_message = "inserted"
                    print(f"DEBUG (DB): New movie '{title}' inserted successfully with ID: {movie_id}. Vote Count: {vote_count_val}, Poster URL: '{poster_url_val}'")
            else:
                self.last_message = "error" # Should ideally not be hit with RETURNING
                print(f"DEBUG (DB): Failed to insert/update movie '{title}' and no movie_id returned.")
                self.conn.rollback() # Rollback if this unexpected case occurs
                return self.last_message # Return early if main movie record fails

            if movie_id:
                # Handle Genres
                if genre and genre != 'N/A':
                    genre_names = [g.strip() for g in genre.split(',') if g.strip()]
                    for g_name in genre_names:
                        genre_id = self._get_or_create_id('genres', 'name', g_name)
                        if genre_id:
                            self.execute_query(
                                "INSERT INTO movie_genres (movie_id, genre_id) VALUES (%s, %s) ON CONFLICT (movie_id, genre_id) DO NOTHING",
                                (movie_id, genre_id)
                            )

                # Handle Directors
                if director and director != 'N/A':
                    director_names = [d.strip() for d in director.split(',') if d.strip()]
                    for d_name in director_names:
                        director_id = self._get_or_create_id('directors', 'name', d_name)
                        if director_id:
                            self.execute_query(
                                "INSERT INTO movie_directors (movie_id, director_id) VALUES (%s, %s) ON CONFLICT (movie_id, director_id) DO NOTHING",
                                (movie_id, director_id)
                            )

                # Handle Actors (formerly 'Stars')
                if stars and stars != 'N/A':
                    actor_names = [s.strip() for s in stars.split(',') if s.strip()]
                    for a_name in actor_names:
                        actor_id = self._get_or_create_id('actors', 'name', a_name)
                        if actor_id:
                            self.execute_query(
                                "INSERT INTO movie_actors (movie_id, actor_id) VALUES (%s, %s) ON CONFLICT (movie_id, actor_id) DO NOTHING",
                                (movie_id, actor_id)
                            )
                self.conn.commit() # Commit all changes for this movie record
                return self.last_message # Return the determined status
            else:
                self.last_message = "error"
                print(f"DEBUG (DB): No movie_id available after main insert/update for '{title}'. Rolling back.")
                self.conn.rollback()
                return self.last_message
        except psycopg2.errors.NumericValueOutOfRange as e:
            self.last_message = "error_value_out_of_range"
            print(f"Database error: Numeric value out of range for vote_count for '{title}': {e}")
            self.conn.rollback()
            return self.last_message
        except Exception as e:
            self.last_message = "error"
            print(f"Error in add_movie for {title}: {e}")
            self.conn.rollback() # Rollback all operations for this movie if any step fails
            return self.last_message

    def get_movies(self, limit: int = 100) -> List[Dict]:
        query = """
        SELECT
            m.movie_id, m.title, m.release_year, m.duration_minutes, m.imdb_rating,
            m.vote_count,
            m.plot_summary, m.poster_url, m.created_at,
            COALESCE(STRING_AGG(DISTINCT g.name, ', ') FILTER (WHERE g.name IS NOT NULL), '') AS genre,
            COALESCE(STRING_AGG(DISTINCT d.name, ', ') FILTER (WHERE d.name IS NOT NULL), '') AS director,
            COALESCE(STRING_AGG(DISTINCT a.name, ', ') FILTER (WHERE a.name IS NOT NULL), '') AS stars
        FROM combined_movies m
        LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.genre_id
        LEFT JOIN movie_directors md ON m.movie_id = md.movie_id
        LEFT JOIN directors d ON md.director_id = d.director_id
        LEFT JOIN movie_actors ma ON m.movie_id = ma.movie_id
        LEFT JOIN actors a ON ma.actor_id = a.actor_id
        GROUP BY m.movie_id, m.title, m.release_year, m.duration_minutes, m.imdb_rating, m.vote_count, m.plot_summary, m.poster_url, m.created_at
        ORDER BY m.imdb_rating DESC
        LIMIT %s;
        """
        return self.execute_query(query, (limit,))

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")