document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Elements ---
    const moviesContainer = document.getElementById('movies-container');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const movieModal = document.getElementById('movie-modal');
    const closeModal = document.querySelector('.close');
    const movieDetailsContainer = document.getElementById('movie-details');
    const searchInput = document.getElementById('search-movies');
    const recommendationMovieTitleInput = document.getElementById('recommendation-movie-title');
    const getRecommendationsButton = document.getElementById('get-recommendations-button');
    const suggestionsDropdown = document.getElementById('suggestions-dropdown');
    const globalLoader = document.getElementById('global-loader');
    const seeMoreMoviesButton = document.getElementById('see-more-movies-button');

    // DOM Elements for search type
    const searchTypeMovieRadio = document.querySelector('input[name="search-type"][value="movie_name"]');
    const searchTypeGenreRadio = document.querySelector('input[name="search-type"][value="genre"]');
    const movieTitleInputWrapper = document.getElementById('movie-title-input-wrapper');
    const genreDropdownWrapper = document.getElementById('genre-dropdown-wrapper');
    const genreDropdown = document.getElementById('genre-dropdown');


    // --- FastAPI Backend URL ---
    const BASE_URL = 'http://127.0.0.1:8000';

    let allMovies = []; 
    let allGenres = []; 
    let currentOpenedMovieId = null; // To keep track of the movie in the modal for rating

    const USER_ID = 1; // Assuming a fixed user ID for now

    // --- Constants for movie loading ---
    const INITIAL_MOVIE_LIMIT = 100;
    const FULL_MOVIE_LIMIT = 5000; // A sufficiently large number to fetch all movies

    // --- Loader Functions ---
    function showLoader() {
        globalLoader.classList.remove('hidden');
    }

    function hideLoader() {
        globalLoader.classList.add('hidden');
    }

    // --- Message Display Helper ---
    function displayMessageInContainer(container, message, isError = false) {
        container.innerHTML = `
            <div class="col-span-full text-center py-8 ${isError ? 'text-red-500' : 'text-gray-500'}">
                <i class="fas ${isError ? 'fa-exclamation-triangle' : 'fa-info-circle'} text-3xl mb-2"></i>
                <p>${message}</p>
            </div>
        `;
    }

    // --- Movie Card Renderer ---
    function renderMovieCards(movies, container, showSimilarity = false) {
        container.innerHTML = ''; // Clear previous content
        if (movies.length === 0) {
            displayMessageInContainer(container, "No movies found.");
            return;
        }

        movies.forEach(movie => {
            const movieCard = document.createElement('div');
            movieCard.className = 'movie-card bg-gray-50 rounded-lg overflow-hidden cursor-pointer transition hover:shadow-lg';

            let similarityDisplay = '';
            // Only show similarity if it's explicitly for recommendations and the score is present
            if (showSimilarity && movie.similarity_score !== undefined && movie.similarity_score !== 1.0) {
                similarityDisplay = `<p class="text-gray-600 text-sm mt-1">Similarity: ${(movie.similarity_score * 100).toFixed(1)}%</p>`;
            }

            movieCard.innerHTML = `
                <img src="${movie.poster_url || 'https://placehold.co/150x225/A0A0A0/FFFFFF?text=No+Poster'}"
                    alt="${movie.title}"
                    onerror="this.onerror=null;this.src='https://placehold.co/150x225/A0A0A0/FFFFFF?text=No+Poster';"
                    class="w-full h-48 object-cover">
                <div class="p-3">
                    <h3 class="font-medium text-gray-800 truncate">${movie.title}</h3>
                    <div class="flex items-center mt-1">
                        <span class="text-yellow-400 mr-1">
                            ${'★'.repeat(Math.floor(movie.rating || 0))}${'☆'.repeat(5 - Math.floor(movie.rating || 0))}
                        </span>
                        <span class="text-gray-600 text-sm">${(movie.rating || 0).toFixed(1)}</span>
                    </div>
                    ${similarityDisplay}
                </div>
            `;
            movieCard.addEventListener('click', () => openMovieModal(movie));
            container.appendChild(movieCard);
        });
    }

    // --- Search Suggestions Renderer ---
    function renderSuggestions(movies) {
        suggestionsDropdown.innerHTML = '';
        if (movies.length === 0) {
            suggestionsDropdown.classList.add('hidden');
            return;
        }

        movies.slice(0, 5).forEach(movie => {
            const suggestionItem = document.createElement('div');
            suggestionItem.className = 'p-2 hover:bg-gray-100 cursor-pointer border-b border-gray-100 last:border-0';
            suggestionItem.textContent = movie.title;
            suggestionItem.addEventListener('click', () => {
                recommendationMovieTitleInput.value = movie.title;
                suggestionsDropdown.classList.add('hidden');
                getRecommendationsButton.click(); // Trigger recommendation search
            });
            suggestionsDropdown.appendChild(suggestionItem);
        });
        suggestionsDropdown.classList.remove('hidden');
    }

    // --- Fetch All Movies (Initial Load with limit) ---
    async function loadInitialMovies() {
        try {
            showLoader();
            const response = await fetch(`${BASE_URL}/movies?limit=${INITIAL_MOVIE_LIMIT}`); // Request initial 100
            const data = await response.json();

            if (response.ok && data.movies) {
                allMovies = data.movies; // Store the initial 100 movies
                renderMovieCards(allMovies, moviesContainer);
                // Show "See More" button if there are potentially more movies
                if (allMovies.length === INITIAL_MOVIE_LIMIT) {
                    seeMoreMoviesButton.classList.remove('hidden');
                } else {
                    seeMoreMoviesButton.classList.add('hidden'); // Hide if all are loaded (less than initial limit)
                }
            } else {
                displayMessageInContainer(moviesContainer, data.detail || 'Failed to load movies.', true);
                throw new Error(data.detail || 'Failed to load movies.');
            }
        } catch (error) {
            console.error('Error loading movies:', error);
            displayMessageInContainer(moviesContainer, `Failed to load movies: ${error.message}`, true);
        } finally {
            hideLoader();
        }
    }

    // --- Load All Movies (when "See More" is clicked) ---
    async function loadAllMoviesFromAPI() {
        try {
            showLoader();
            // Request a large number to get all movies. You can also hardcode 250 if that's the max.
            const response = await fetch(`${BASE_URL}/movies?limit=${FULL_MOVIE_LIMIT}`);
            const data = await response.json();

            if (response.ok && data.movies) {
                allMovies = data.movies; // Update allMovies with the full list
                renderMovieCards(allMovies, moviesContainer);
                seeMoreMoviesButton.classList.add('hidden'); // Hide the button once all are loaded
            } else {
                displayMessageInContainer(moviesContainer, data.detail || 'Failed to load all movies.', true);
                throw new Error(data.detail || 'Failed to load all movies.');
            }
        } catch (error) {
            console.error('Error loading all movies:', error);
            displayMessageInContainer(moviesContainer, `Failed to load all movies: ${error.message}`, true);
        } finally {
            hideLoader();
        }
    }

    // Load Genres for Dropdown
    async function loadGenres() {
        try {
            showLoader();
            const response = await fetch(`${BASE_URL}/genres`);
            const genres = await response.json();

            if (response.ok && genres) {
                allGenres = genres;
                genreDropdown.innerHTML = '<option value="">Select a genre</option>'; // Default option
                allGenres.forEach(genre => {
                    const option = document.createElement('option');
                    option.value = genre;
                    option.textContent = genre;
                    genreDropdown.appendChild(option);
                });
            } else {
                console.error("Failed to load genres:", genres.detail || response.status);
            }
        } catch (error) {
            console.error("Error fetching genres:", error);
        } finally {
            hideLoader();
        }
    }

    //  Toggle visibility of movie title input or genre dropdown
    function toggleSearchType() {
        if (searchTypeMovieRadio.checked) {
            movieTitleInputWrapper.classList.remove('hidden');
            genreDropdownWrapper.classList.add('hidden');
            // Clear genre selection when switching to movie name
            genreDropdown.value = "";
        } else {
            movieTitleInputWrapper.classList.add('hidden');
            genreDropdownWrapper.classList.remove('hidden');
            // Clear movie title input when switching to genre
            recommendationMovieTitleInput.value = "";
            suggestionsDropdown.classList.add('hidden'); // Hide suggestions
        }
    }


    // --- Movie Details Modal ---
    // Inside your app.js file

    async function openMovieModal(movie) {
        showLoader();
        try {
            currentOpenedMovieId = movie.id;

            // Clear previous content and populate with new movie data
            movieDetailsContainer.innerHTML = `
                <h2 class="text-3xl font-bold text-gray-900 mb-4">${movie.title}</h2>
                <div class="flex flex-col md:flex-row gap-6">
                    <div class="flex-shrink-0">
                        <img src="${movie.poster_url || 'https://placehold.co/200x300/A0A0A0/FFFFFF?text=No+Poster'}"
                                alt="${movie.title}"
                                onerror="this.onerror=null;this.src='https://placehold.co/200x300/A0A0A0/FFFFFF?text=No+Poster';"
                                class="w-48 h-auto rounded-lg shadow-md">
                    </div>
                    <div>
                        <p class="text-gray-700 mb-2"><strong>Year:</strong> ${movie.year || 'N/A'}</p>
                        <p class="text-gray-700 mb-2"><strong>Genre:</strong> ${movie.genre || 'Unknown'}</p>
                        <p class="text-gray-700 mb-2"><strong>Runtime:</strong> ${movie.duration || 'N/A'} minutes</p>
                        <p class="text-gray-700 mb-2"><strong>Director:</strong> ${movie.director || 'N/A'}</p>
                        ${movie.similarity_score !== undefined && movie.similarity_score !== 1.0 ? `<p class="text-gray-700 mb-2"><strong>Similarity:</strong> ${(movie.similarity_score * 100).toFixed(1)}%</p>` : ''}
                        <p class="text-gray-700 mb-4"><strong>Overview:</strong> ${movie.overview || 'No overview available.'}</p>

                        <hr class="my-4">

                        <div class="mb-4">
                            <h4 class="text-lg font-semibold text-gray-800 mb-2">Your Rating:</h4>
                            <div id="rating-stars" class="flex space-x-1 text-yellow-500 text-2xl cursor-pointer">
                                <i class="far fa-star" data-rating="1"></i>
                                <i class="far fa-star" data-rating="2"></i>
                                <i class="far fa-star" data-rating="3"></i>
                                <i class="far fa-star" data-rating="4"></i>
                                <i class="far fa-star" data-rating="5"></i>
                            </div>
                            <p class="text-sm text-gray-600 mt-1" id="current-user-rating-text">Click a star to rate this movie!</p>
                        </div>

                        <div class="mb-4">
                            <h4 class="text-lg font-semibold text-gray-800 mb-2">Average User Rating:</h4>
                            <div id="average-rating-display" class="flex items-center">
                                <span class="text-yellow-500 text-xl font-bold mr-2" id="avg-rating-value">N/A</span>
                                <div id="avg-rating-stars" class="flex space-x-0.5 text-yellow-500">
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            `;

            const avgRatingValueElement = movieDetailsContainer.querySelector("#avg-rating-value");
            const avgRatingStarsContainer = movieDetailsContainer.querySelector("#avg-rating-stars");
            if (avgRatingValueElement && avgRatingStarsContainer) {
                if (typeof movie.rating === "number" && !isNaN(movie.rating)) {
                    avgRatingValueElement.textContent = movie.rating.toFixed(1);
                    displayStars(avgRatingStarsContainer, movie.rating);
                } else {
                    avgRatingValueElement.textContent = "N/A";
                    avgRatingStarsContainer.innerHTML = '';
                }
            }

            // --- ADDED: Event listener for "Your Rating" stars ---
            const ratingStarsContainer = movieDetailsContainer.querySelector("#rating-stars");
            if (ratingStarsContainer) {
                ratingStarsContainer.addEventListener('click', handleStarRatingClick);
            }
            // --- END ADDED ---

            await loadUserRating(movie.id);

            movieModal.classList.remove('hidden');

        } catch (error) {
            console.error("Error opening movie modal:", error);
            // Replaced alert with display in console for better UX
            console.log("Could not load movie details: " + error.message);
        } finally {
            hideLoader();
        }
    }

    closeModal.addEventListener('click', () => {
        movieModal.classList.add('hidden');
        currentOpenedMovieId = null;
        movieDetailsContainer.innerHTML = "";
    });
    movieModal.addEventListener('click', e => { if (e.target === movieModal) movieModal.classList.add('hidden'); });

    // --- Rating Logic ---

    async function loadUserRating(movieId) {
        const userRatingTextElement = movieDetailsContainer.querySelector("#current-user-rating-text");
        const starsContainer = movieDetailsContainer.querySelector("#rating-stars");

        if (!userRatingTextElement || !starsContainer) return;

        try {
            const response = await fetch(`${BASE_URL}/users/${USER_ID}/ratings`);
            if (response.ok) {
                const data = await response.json();
                const userRating = data.ratings.find(rating => rating.movie_id === movieId);
                if (userRating) {
                    displayStars(starsContainer, userRating.rating);
                    userRatingTextElement.textContent = `You rated this ${userRating.rating.toFixed(1)} out of 5.`;
                } else {
                    displayStars(starsContainer, 0);
                    userRatingTextElement.textContent = "Click a star to rate this movie!";
                }
            } else if (response.status === 404) {
                displayStars(starsContainer, 0);
                userRatingTextElement.textContent = "Click a star to rate this movie!";
            } else {
                const errorData = await response.json();
                console.error(`Error loading user ratings: ${errorData.detail || response.status}`);
                displayStars(starsContainer, 0);
                userRatingTextElement.textContent = "Could not load your rating.";
            }
        } catch (error) {
            console.error("Fetch error for user ratings:", error);
            displayStars(starsContainer, 0);
            userRatingTextElement.textContent = "Error loading your rating.";
        }
    }

    async function loadAverageRating(movieId) {
        const avgRatingValueElement = movieDetailsContainer.querySelector("#avg-rating-value");
        const avgRatingStarsContainer = movieDetailsContainer.querySelector("#avg-rating-stars");

        if (!avgRatingValueElement || !avgRatingStarsContainer) return;

        try {
            // Updated endpoint to the correct one
            const response = await fetch(`${BASE_URL}/movies/${movieId}/average-rating`);
            const data = await response.json();

            if (response.ok && data.average_rating !== undefined) {
                if (data.average_rating !== null) {
                    const avgRating = data.average_rating;
                    avgRatingValueElement.textContent = avgRating.toFixed(1);
                    displayStars(avgRatingStarsContainer, avgRating); // Use existing displayStars helper
                } else {
                    avgRatingValueElement.textContent = "N/A";
                    avgRatingStarsContainer.innerHTML = ''; // Clear stars if no ratings
                }
            } else {
                console.error("Failed to load average rating:", data.detail || response.status);
                avgRatingValueElement.textContent = "Error";
                avgRatingStarsContainer.innerHTML = '';
            }
        } catch (error) {
            console.error("Error fetching average rating:", error);
            avgRatingValueElement.textContent = "Error";
            avgRatingStarsContainer.innerHTML = '';
        }
    }


    function displayStars(container, rating) {
    container.innerHTML = '';
    for (let i = 1; i <= 5; i++) {
        let className = 'far fa-star star';
        if (i <= Math.floor(rating)) {
            className = 'fas fa-star star';
        } else if (i - rating <= 0.5 && rating % 1 !== 0) {
            className = 'fas fa-star-half-alt star';
        }
        const star = document.createElement('i');
        star.className = className;
        star.dataset.rating = i; // Always set for click
        container.appendChild(star);
    }
}



    // Inside your handleStarRatingClick function in app.js
    async function handleStarRatingClick(event) {
        // Correctly get the target and its data-rating
        const target = event.target;
        const clickedRating = target.dataset.rating; // Get the data-rating directly from the clicked <i> element

        if (!clickedRating) return; // If no data-rating, do nothing

        if (currentOpenedMovieId === null) {
            console.error("No movie selected for rating.");
            console.log("Please select a movie to rate.");
            return;
        }

        const rating = parseFloat(clickedRating); // Convert the string rating to a float

        // Immediately update the visual stars for "Your Rating"
        const starsContainer = movieDetailsContainer.querySelector("#rating-stars");
        const userRatingTextElement = movieDetailsContainer.querySelector("#current-user-rating-text");
        if (starsContainer && userRatingTextElement) {
            displayStars(starsContainer, rating); // Visually fill the stars
            userRatingTextElement.textContent = `You rated this ${rating.toFixed(1)} out of 5.`;
        }

        showLoader();
        try {
            const response = await fetch(`${BASE_URL}/ratings`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    user_id: USER_ID,
                    movie_id: currentOpenedMovieId,
                    rating: rating,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Failed to submit rating: ${response.status}`);
            }

            const result = await response.json();
            console.log("Rating submitted/updated:", result);

            // Removed: await loadAverageRating(currentOpenedMovieId);
            // This prevents the average rating from updating immediately after a user's submission.
            // The average rating will reflect what was loaded when the modal opened,
            // or update upon subsequent modal re-opening.

        } catch (error) {
            console.error("Error submitting rating:", error);
            console.log(`Error submitting rating: ${error.message}`);
            // Optionally, revert the stars if the submission failed
            // You might want to reload the user's previous rating here if submission fails
            await loadUserRating(currentOpenedMovieId);
        } finally {
            hideLoader();
        }
    }


    // --- Search Input Event Listener (for "All Movies" section) ---
    searchInput.addEventListener('input', async (e) => {
        const searchTerm = e.target.value.trim();

        if (searchTerm === '') {
            loadInitialMovies(); // Revert to initial 100 movies if search is cleared
            return;
        }

        try {
            showLoader();
            seeMoreMoviesButton.classList.add('hidden'); // Hide "See More" button during search results
            const response = await fetch(`${BASE_URL}/search?q=${encodeURIComponent(searchTerm)}`);

            const searchResult = await response.json();

            if (!response.ok) {
                displayMessageInContainer(moviesContainer, searchResult.detail || `Server error: ${response.status}`, true);
            } else if (searchResult.message) {
                displayMessageInContainer(moviesContainer, searchResult.message); // Should not happen with search endpoint, but good to have
            } else {
                renderMovieCards(searchResult, moviesContainer);
            }
        } catch (error) {
            console.error('Search error:', error);
            displayMessageInContainer(moviesContainer, `Error searching movies: ${error.message}`, true);
        } finally {
            hideLoader();
        }
    });

    // --- Get Recommendations Button Event Listener (for "Similar Movies" section) ---
    getRecommendationsButton.addEventListener('click', async () => {
        let query = "";
        let searchBy = "";

        if (searchTypeMovieRadio.checked) {
            query = recommendationMovieTitleInput.value.trim();
            searchBy = "movie_name";
        } else if (searchTypeGenreRadio.checked) {
            query = genreDropdown.value; // Value of the selected option
            searchBy = "genre";
        }

        const cleanedQueryForValidation = query.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
        if (!cleanedQueryForValidation) {
            displayMessageInContainer(recommendationsContainer, `Please select a genre or enter a valid movie name for recommendations.`);
            // Clear input if invalid and it was a movie name search
            if (searchBy === "movie_name") {
                recommendationMovieTitleInput.value = '';
            }
            return;
        }

        getRecommendationsButton.disabled = true;
        getRecommendationsButton.classList.add("opacity-50", "cursor-not-allowed");

        try {
            showLoader();
            const response = await fetch(`${BASE_URL}/flexible-recommendations`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    query: query,
                    search_by: searchBy,
                    num_recommendations: 10 
                }),
            });
            const data = await response.json();

            if (response.ok && data.recommendations) {
                const recommendations = data.recommendations;
                if (recommendations.length === 0) {
                    displayMessageInContainer(recommendationsContainer, data.message || 'No recommendations found for this query.');
                } else {
                    renderMovieCards(recommendations, recommendationsContainer, searchBy === "movie_name");
                }
            } else {
                displayMessageInContainer(recommendationsContainer, data.detail || `Server error: ${response.status}`, true);
            }
        } catch (error) {
            console.error('Recommendation error:', error);
            displayMessageInContainer(recommendationsContainer, `Error getting recommendations: ${error.message}`, true);
        } finally {
            getRecommendationsButton.disabled = false;
            getRecommendationsButton.classList.remove("opacity-50", "cursor-not-allowed");
            hideLoader();
        }
    });

    // --- Recommendation Input (Enter Key for Movie Title Search) ---
    recommendationMovieTitleInput.addEventListener('keyup', function (e) {
        // Only trigger on Enter if "Movie Name" search is selected
        if (e.key === 'Enter' && searchTypeMovieRadio.checked) {
            getRecommendationsButton.click();
        }
    });

    // NEW: Trigger recommendation search when genre is selected
    genreDropdown.addEventListener('change', function () {
        if (searchTypeGenreRadio.checked && genreDropdown.value !== "") {
            getRecommendationsButton.click();
        } else if (genreDropdown.value === "") {
             displayMessageInContainer(recommendationsContainer, 'Please select a genre to get recommendations.');
        }
    });


    // --- Search Suggestions (Debounced Input for Movie Title Search) ---
    let debounceTimeout;
    recommendationMovieTitleInput.addEventListener('input', (e) => {
        // Only provide suggestions if "Movie Name" search is selected
        if (!searchTypeMovieRadio.checked) {
            suggestionsDropdown.classList.add('hidden');
            return;
        }

        const searchTerm = e.target.value.trim();

        clearTimeout(debounceTimeout);

        const cleanedSearchTermForValidation = searchTerm.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
        if (!cleanedSearchTermForValidation) {
            suggestionsDropdown.classList.add('hidden');
            return;
        }

        debounceTimeout = setTimeout(async () => {
            try {
                showLoader();
                const response = await fetch(`${BASE_URL}/search?q=${encodeURIComponent(searchTerm)}&limit=5`);
                const suggestions = await response.json();

                if (response.ok && suggestions && suggestions.length > 0) {
                    renderSuggestions(suggestions);
                } else {
                    suggestionsDropdown.classList.add('hidden');
                }
            } catch (error) {
                console.error('Error fetching suggestions:', error);
                suggestionsDropdown.classList.add('hidden');
            } finally {
                hideLoader();
            }
        }, 500); // Debounce for 500ms
    });

    // --- Hide Suggestions on Outside Click ---
    document.addEventListener('click', (e) => {
        if (!recommendationMovieTitleInput.contains(e.target) && !suggestionsDropdown.contains(e.target)) {
            suggestionsDropdown.classList.add('hidden');
        }
    });

    // --- Event Listener for "See More" button ---
    if (seeMoreMoviesButton) {
        seeMoreMoviesButton.addEventListener('click', loadAllMoviesFromAPI);
    }

    // NEW: Event Listeners for radio buttons to toggle search type
    searchTypeMovieRadio.addEventListener('change', toggleSearchType);
    searchTypeGenreRadio.addEventListener('change', toggleSearchType);


    // --- Initial Loads on DOM Content Loaded ---
    loadInitialMovies(); // Load initial set of movies for the "All Movies" section
    loadGenres(); // Load genres for the dropdown
    toggleSearchType(); // Set initial visibility of search inputs (movie name by default)
});