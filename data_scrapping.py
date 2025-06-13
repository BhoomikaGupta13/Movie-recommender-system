import requests
from bs4 import BeautifulSoup
import time
from random import uniform
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for parallel processing

# Import Selenium components
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Import ChromeDriverManager
from webdriver_manager.chrome import ChromeDriverManager

from database1 import MovieDatabase # Assuming database.py is in the same directory

# --- Your existing scrape_movie_details function (no changes needed here) ---
# This function is called for each individual movie URL
def scrape_movie_details(movie_url, headers):
    try:
        movie_response = requests.get(movie_url, headers=headers, timeout=10)
        movie_response.raise_for_status()
        movie_soup = BeautifulSoup(movie_response.text, 'html.parser')

        # --- Extract description ---
        description = "N/A"
        desc_element = movie_soup.select_one('span[data-testid="plot-xl"]')
        if desc_element:
            description = desc_element.get_text(strip=True)
        else:
            desc_element = movie_soup.select_one('p[data-testid="plot"] span')
            if desc_element:
                description = desc_element.get_text(strip=True)
            else:
                meta_desc = movie_soup.find('meta', {'name': 'description'})
                if meta_desc:
                    description = meta_desc['content'].strip()

        # --- Extract genres ---
        metadata_genre = movie_soup.select('div.ipc-chip-list__scroller')
        if metadata_genre:
            genre_links = metadata_genre[0].select('a')
            genres = ', '.join(g.get_text(strip=True) for g in genre_links)
        else:
            genres = 'N/A'

        # --- Extract metadata like director and stars ---
        directors = []
        stars = []
        credit_blocks = movie_soup.select('li[data-testid="title-pc-principal-credit"]')

        for block in credit_blocks:
            label = block.select_one('span.ipc-metadata-list-item__label')
            if label:
                role = label.get_text(strip=True)
                names = [a.get_text(strip=True) for a in block.select('a')]
                if role == 'Director' or role == 'Directors':
                    directors = names
                elif role == 'Stars':
                    stars = names
        
        if not stars:
            star_section = movie_soup.select('a[href*="tt_ov_st_"]')
            stars = [star.get_text(strip=True) for star in star_section if star.get_text(strip=True)]
        
        if not stars:
            stars = ['N/A']

        # --- Extract runtime ---
        runtime = 'N/A'
        runtime_element = movie_soup.select_one('li[data-testid="title-techspec_runtime"] div.ipc-metadata-list-item__content-container')
        if runtime_element:
            runtime = runtime_element.get_text(strip=True)
        else:
            metadata_items = movie_soup.select('span.cli-title-metadata-item')
            if len(metadata_items) >= 2:
                runtime = metadata_items[1].get_text(strip=True)

        # --- Extract year ---
        year_from_detail = None
        metadata_list_around_title = movie_soup.select_one('ul.ipc-inline-list.ipc-inline-list--show-dividers')
        if metadata_list_around_title:
            year_candidate_tags = metadata_list_around_title.find_all('li', class_='ipc-inline-list__item')
            for tag in year_candidate_tags:
                text = tag.get_text(strip=True)
                if re.match(r'^\d{4}$', text):
                    year_from_detail = int(text)
                    break
        if year_from_detail is None:
            year_link_tag = movie_soup.find('a', href=re.compile(r'/title/tt\d+/releaseinfo'))
            if year_link_tag:
                match = re.search(r'\d{4}', year_link_tag.get_text(strip=True))
                if match:
                    year_from_detail = int(match.group(0))

        # --- Extract Movie Poster URL ---
        poster_url = "N/A"
        poster_element = movie_soup.select_one('div[data-testid="hero-media__poster"] img.ipc-image')
        if poster_element and 'src' in poster_element.attrs:
            poster_url = poster_element['src']
        else:
            poster_element = movie_soup.select_one('img.ipc-image[data-testid="hero-media__poster"]')
            if poster_element and 'src' in poster_element.attrs:
                poster_url = poster_element['src']
            elif movie_soup.find('meta', property='og:image'):
                poster_url = movie_soup.find('meta', property='og:image')['content']
                if '._' in poster_url:
                    poster_url = poster_url.split('._')[0] + '._V1_.jpg'

        if poster_url != "N/A" and not poster_url.startswith('http'):
            poster_url = "N/A"

        return {
            'genre': genres,
            'director': ', '.join(directors) if directors else 'N/A',
            'stars': ', '.join(stars) if stars else 'N/A',
            'plot': description,
            'runtime': runtime,
            'year': year_from_detail,
            'poster_url': poster_url if poster_url else 'N/A'
        }
    except Exception as e:
        print(f"Failed to fetch details for {movie_url}: {e}")
        return {
            'genre': 'N/A',
            'director': 'N/A',
            'stars': 'N/A',
            'plot': 'N/A',
            'runtime': 'N/A',
            'year': None,
            'poster_url': 'N/A'
        }

# --- UPDATED scrape_imdb function to use Selenium with webdriver_manager and concurrent futures ---
def scrape_imdb():
    url = "https://www.imdb.com/chart/top/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }

    print("Fetching IMDb Top 250 Movies using Selenium and concurrent processing...\n")
    movies_data = []
    movie_urls_to_scrape = [] # To store URLs for parallel processing

    try:
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument(f'user-agent={headers["User-Agent"]}')
        
        driver = webdriver.Chrome(service=service, options=options)

        driver.get(url)

        try:
            # Wait for at least 250 elements of the movie list to be present
            WebDriverWait(driver, 30).until( # Increased timeout slightly
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item'))
            )
            print("Selenium: Waited for movie list elements to load.")
        except TimeoutException:
            print("Selenium: Timed out waiting for all movie list elements to load. Proceeding with what's available.")

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit() # Close the browser as we have the full page source

        movie_rows = soup.select('li.ipc-metadata-list-summary-item')
        
        print(f"Selenium: Found {len(movie_rows)} movie rows after page load.")

        if not movie_rows:
            print("Error: No movies found after Selenium page load.")
            return None

        print("IMDb Top Movies:")
        print("=" * 50)

        num_to_scrape = min(250, len(movie_rows))
        print(f"Attempting to scrape {num_to_scrape} movies.")

        # --- Phase 1: Extract basic info and all movie URLs from the main list ---
        # We don't need sleep here as we're just parsing the local soup object
        for i, row in enumerate(movie_rows[:num_to_scrape], 1):
            title_column = row.select_one('div.ipc-title a')
            if not title_column:
                print(f"Warning: Could not find title for row {i}. Skipping.")
                continue

            title = title_column.get_text(strip=True)
            movie_url = "https://www.imdb.com" + title_column['href'].split('?')[0]

            rating_column = row.select_one('span.ipc-rating-star--rating')
            rating = rating_column.text.strip() if rating_column else 'N/A'

            views = "N/A"
            views_element = row.select_one('span.ipc-rating-star--voteCount')
            if views_element:
                raw_text = views_element.text.strip().replace('(', '').replace(')', '')
                try:
                    if 'M' in raw_text:
                        views = int(float(raw_text.replace('M', '')) * 1_000_000)
                    elif 'K' in raw_text:
                        views = int(float(raw_text.replace('K', '')) * 1_000)
                    elif ',' in raw_text:
                        views = int(raw_text.replace(',', ''))
                    else:
                        views = int(raw_text)
                except ValueError:
                    views = 'N/A'

            metadata = row.select('span.cli-title-metadata-item')
            year_chart = int(metadata[0].text.strip()) if len(metadata) >= 3 and metadata[0].text.strip().isdigit() else None

            # Store basic info and the URL for later detailed scraping
            movie_urls_to_scrape.append({
                'Rank': i,
                'Title': title,
                'Rating': rating,
                'Views': views,
                'Year_Chart': year_chart, # Keep chart year separate for now
                'Movie_URL': movie_url
            })

        print(f"Extracted {len(movie_urls_to_scrape)} movie URLs for detailed scraping.")

        # --- Phase 2: Scrape details for each movie in parallel ---
        # Using ThreadPoolExecutor for concurrent requests to individual movie pages
        # Max workers can be adjusted based on your internet speed and IMDb's tolerance
        MAX_WORKERS = 10 # You can increase this (e.g., 20, 30) if your connection and IMDb allow
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_movie = {
                executor.submit(scrape_movie_details, movie_info['Movie_URL'], headers): movie_info
                for movie_info in movie_urls_to_scrape
            }
            
            for i, future in enumerate(as_completed(future_to_movie), 1):
                movie_info = future_to_movie[future]
                try:
                    details = future.result()
                    
                    final_year = details['year'] if details['year'] is not None else movie_info['Year_Chart']
                    poster_url = details['poster_url'] if details['poster_url'] else 'N/A'

                    movies_data.append({
                        'Rank': movie_info['Rank'],
                        'Title': movie_info['Title'],
                        'Rating': movie_info['Rating'],
                        'Views': movie_info['Views'],
                        'Year': final_year,
                        'Runtime': details['runtime'],
                        'Genre': details['genre'],
                        'Director': details['director'],
                        'Stars': details['stars'],
                        'Plot': details['plot'],
                        'Poster_URL': poster_url
                    })
                    
                    # Print progress
                    if i % 25 == 0 or i == num_to_scrape: # Print every 25 movies or at the end
                        print(f"Scraped details for {i}/{num_to_scrape} movies.")

                except Exception as exc:
                    print(f"Movie {movie_info['Title']} ({movie_info['Movie_URL']}) generated an exception: {exc}")

        df = pd.DataFrame(movies_data)
        return df

    except WebDriverException as e:
        print(f"Selenium WebDriver error: {e}")
        print("Ensure you have Chrome installed. webdriver_manager should handle the driver download.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred in scrape_imdb: {e}")
        return None

# --- Your existing convert_runtime function ---
def convert_runtime(runtime_str):
    if not runtime_str or runtime_str == 'N/A':
        return None

    runtime_str = runtime_str.lower()
    
    hours = 0
    minutes = 0

    hour_match = re.search(r'(\d+)\s*(?:h|hour|hours)', runtime_str)
    if hour_match:
        hours = int(hour_match.group(1))

    minute_match = re.search(r'(\d+)\s*(?:m|min|mins|minute|minutes)', runtime_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    if not hour_match and not minute_match:
        single_number_match = re.search(r'^\s*(\d+)\s*$', runtime_str)
        if single_number_match:
            minutes = int(single_number_match.group(1))

    total_minutes = hours * 60 + minutes
    
    if total_minutes > 0:
        return total_minutes
    
    return None

# --- Your existing validate_imdb_data function ---
def validate_imdb_data(df):
    print("\n=== Data Validation ===")
    print("\nMissing values per column (before cleaning):")
    print(df.isnull().sum())

    invalid_years = df[(df['Year'].notna()) & ((df['Year'] < 1888) | (df['Year'] > 2025))]
    print("\nInvalid years (if any):")
    if not invalid_years.empty:
        print(invalid_years[['Rank', 'Title', 'Year']])
    else:
        print("No invalid years found.")

    def is_valid_rating(rating_str):
        try:
            rating = float(rating_str)
            return 0 <= rating <= 10
        except (ValueError, TypeError):
            return False

    invalid_ratings = df[~df['Rating'].apply(is_valid_rating)]
    print("\nInvalid ratings (if any):")
    if not invalid_ratings.empty:
        print(invalid_ratings[['Rank', 'Title', 'Rating']])
    else:
        print("No invalid ratings found.")

    print("\nUnique Runtime formats (before cleaning):")
    print(df['Runtime'].unique())

    print("\nPoster_URL stats:")
    print(f"Null values: {df['Poster_URL'].isnull().sum()}")
    print(f"Empty strings: {(df['Poster_URL'] == '').sum()}")
    print(f"'N/A' values: {(df['Poster_URL'] == 'N/A').sum()}")
    print(f"Valid URLs: {df['Poster_URL'].str.startswith('http').sum()}")

# --- Your existing clean_imdb_data function ---
def clean_imdb_data(df):
    print("\n=== Cleaning Data ===")
    df = df.copy()

    initial_rows = len(df)
    df.dropna(subset=['Title', 'Rating', 'Year', 'Runtime', 'Poster_URL'], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with missing key values.")

    df['Runtime_Minutes'] = df['Runtime'].apply(convert_runtime)

    def convert_views_to_int(views_str):
        if isinstance(views_str, str):
            cleaned_str = views_str.strip().upper()
            if not cleaned_str or cleaned_str == 'N/A':
                return None
            try:
                if 'M' in cleaned_str:
                    num_part = float(cleaned_str.replace('M', ''))
                    return int(num_part * 1_000_000)
                elif 'K' in cleaned_str:
                    num_part = float(cleaned_str.replace('K', ''))
                    return int(num_part * 1_000)
                elif ',' in cleaned_str:
                    return int(cleaned_str.replace(',', ''))
                else:
                    return int(cleaned_str)
            except ValueError as e:
                print(f"Warning: ValueError converting '{views_str}' to int: {e}. Setting to None.")
                return None
        return views_str

    df['Views'] = df['Views'].apply(convert_views_to_int)

    df['Poster_URL'] = df['Poster_URL'].apply(lambda x: 'N/A' if pd.isna(x) or x == '' else x)

    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce', downcast='integer')
    df['Runtime_Minutes'] = pd.to_numeric(df['Runtime_Minutes'], errors='coerce')
    df['Views'] = pd.to_numeric(df['Views'], errors='coerce').astype('Int64')

    df['Title'] = df['Title'].str.strip()

    df = df.drop(columns=['Runtime'])

    print("\nCleaned DataFrame preview:")
    print(df[['Title', 'Views', 'Rating', 'Year', 'Poster_URL', 'Stars']].head())
    print(f"\nCleaned DataFrame shape: {df.shape}")
    print(f"\nCleaned DataFrame dtypes: \n{df.dtypes}")
    print(f"\nCleaned DataFrame 'Poster_URL' stats: \n{df['Poster_URL'].value_counts(dropna=False)}")

    return df

if __name__ == "__main__":
    movies_df = scrape_imdb()

    if movies_df is not None and not movies_df.empty:
        print("\nSuccessfully scraped data. DataFrame contains:")
        print(movies_df.info())
        print("\nFirst 5 rows of raw data:")
        print(movies_df.head())

        validate_imdb_data(movies_df)
        cleaned_df = clean_imdb_data(movies_df)

        # Ensure you use your correct database name and credentials here
        db = MovieDatabase(dbname='imdb_db1', user='postgres', password='bhoomi@123', host='localhost', port='5432')

        print("\n--- Inserting data into PostgreSQL ---")
        inserted_count = 0
        updated_count = 0
        already_exists_count = 0
        error_count = 0

        for index, row in cleaned_df.iterrows():
            movie_title = row['Title']
            movie_year = row.get('Year')
            movie_vote_count = row.get('Views')
            movie_poster_url = row.get('Poster_URL', 'N/A')

            print(f"Attempting to add/update movie '{movie_title}' (Year: {movie_year})")
            print(f"   --> Vote Count: {movie_vote_count} (Type: {type(movie_vote_count)})")
            print(f"   --> Poster URL: {movie_poster_url}")
            print(f"   --> Stars: {row.get('Stars', 'N/A')}")

            try:
                add_movie_status = db.add_movie(
                    title=movie_title,
                    release_year=movie_year,
                    duration=row.get('Runtime_Minutes'),
                    imdb_rating=row.get('Rating'),
                    vote_count=movie_vote_count,
                    plot=row.get('Plot', 'N/A'),
                    poster_url=movie_poster_url,
                    genre=row.get('Genre', 'N/A'),
                    director=row.get('Director', 'N/A'),
                    stars=row.get('Stars', 'N/A'),
                )

                if add_movie_status == "inserted":
                    inserted_count += 1
                elif add_movie_status == "updated":
                    updated_count += 1
                elif add_movie_status == "exists_no_change":
                    already_exists_count += 1
                else:
                    print(f"Error status returned for movie '{movie_title}'.")
                    error_count += 1

            except Exception as e:
                print(f"Unhandled error processing movie '{movie_title}': {e}")
                error_count += 1

        db.close()
        print("\n--- Database Insertion Summary ---")
        print(f"Total movies processed: {len(cleaned_df)}")
        print(f"Movies newly inserted: {inserted_count}")
        print(f"Movies updated: {updated_count}")
        print(f"Movies already existing (no update needed): {already_exists_count}")
        print(f"Movies with insertion/update errors: {error_count}")
        print("All movies processed. Database connection closed.")
    else:
        print("No data was scraped or an error occurred during scraping.")

