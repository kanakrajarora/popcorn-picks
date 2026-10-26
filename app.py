import random

import pandas as pd
import streamlit as st

from recommender import MovieRecommender

# Rename the project by changing these two lines.
APP_NAME = "PopcornPicks"
APP_TAGLINE = "Movie picks based on how people with similar taste rated."

PAGES = ["Home", "Picks for you", "Find similar", "Rating check", "Model report"]

st.set_page_config(
    page_title=f"{APP_NAME} | Movie recommendations",
    page_icon=":clapper:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light styling only. Colors come from rgba so it works in light and dark themes.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1150px; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 8px;
        padding: 10px 14px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
    .movie-title { font-weight: 600; font-size: 1.02rem; line-height: 1.3; margin: 0; }
    .muted { opacity: .65; font-size: .85rem; margin: .15rem 0 .6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Model and cached helpers
# ----------------------------------------------------------------------

@st.cache_resource
def load_recommender():
    return MovieRecommender("models")


try:
    recommender = load_recommender()
except Exception as error:
    st.error("The recommendation model could not be loaded.")
    st.write("Make sure the artifact files are inside the `models` folder.")
    st.exception(error)
    st.stop()

movies = recommender.movies
ratings = recommender.ratings
metadata = recommender.metadata
valid_users = sorted(int(u) for u in recommender.user_ids)
valid_user_set = set(valid_users)


@st.cache_data
def movie_titles():
    return movies["title"].astype(str).sort_values().tolist()


@st.cache_data
def genre_options():
    found = {
        g
        for value in movies["genres"].fillna("")
        for g in str(value).split("|")
        if g and g != "(no genres listed)"
    }
    return sorted(found)


@st.cache_data(show_spinner=False)
def cached_recommendations(_rec, user_id, n, genre):
    return _rec.recommend_movies(user_id=user_id, n=n, genre=genre)


@st.cache_data(show_spinner=False)
def cached_history(_rec, user_id):
    return _rec.user_history(user_id)


@st.cache_data(show_spinner=False)
def movie_rating_stats(title):
    row = movies[movies["title"].astype(str) == title].iloc[0]
    subset = ratings[ratings["movieId"] == row["movieId"]]["rating"]
    if subset.empty:
        return None
    return float(subset.mean()), int(subset.count())


def movie_row(title):
    return movies[movies["title"].astype(str) == title].iloc[0]


# ----------------------------------------------------------------------
# State and navigation
# ----------------------------------------------------------------------

st.session_state.setdefault("page", "Home")
st.session_state.setdefault("user_id", valid_users[0] if valid_users else 1)


def go(page, **state):
    """Callback: jump to a page and optionally preset widget values."""
    for key, value in state.items():
        st.session_state[key] = value
    st.session_state["page"] = page


def pick_random_user():
    if valid_users:
        st.session_state["user_id"] = random.choice(valid_users)


with st.sidebar:
    st.markdown(f"## {APP_NAME}")
    st.caption(APP_TAGLINE)

    st.radio("Go to", PAGES, key="page", label_visibility="collapsed")

    st.divider()
    st.markdown("**Viewing as**")
    st.number_input(
        "User ID",
        min_value=1,
        step=1,
        key="user_id",
        help="Every page uses this user. Change it here at any time.",
    )
    st.button(
        "Pick a random user",
        on_click=pick_random_user,
        use_container_width=True,
    )

    user_id = int(st.session_state["user_id"])
    if user_id in valid_user_set:
        n_rated = len(cached_history(recommender, user_id))
        st.caption(f"{n_rated:,} ratings on record for this user.")
    else:
        st.warning("This ID isn't in the model sample. Try a random user.")

page = st.session_state["page"]
user_known = user_id in valid_user_set


def need_valid_user():
    if not user_known:
        st.info(
            "The selected user ID isn't in the model sample. "
            "Choose **Pick a random user** in the sidebar to continue."
        )
        st.stop()


# ----------------------------------------------------------------------
# Home
# ----------------------------------------------------------------------

if page == "Home":
    st.title(APP_NAME)
    st.markdown(f"##### {APP_TAGLINE}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", f"{len(movies):,}")
    c2.metric("Users in sample", f"{ratings['userId'].nunique():,}")
    c3.metric("Ratings in sample", f"{len(ratings):,}")

    st.write("")
    st.markdown("### Start with something")

    a, b, c = st.columns(3)
    with a.container(border=True):
        st.markdown("**See your picks**")
        st.caption("Unseen movies ranked by predicted rating, filterable by genre.")
        st.button("Open picks", key="h1", on_click=go, args=("Picks for you",))
    with b.container(border=True):
        st.markdown("**Find a movie like another**")
        st.caption("Pick a title you like and see others with matching genres.")
        st.button("Find similar", key="h2", on_click=go, args=("Find similar",))
    with c.container(border=True):
        st.markdown("**Check a rating**")
        st.caption("See how much this user would likely enjoy a specific movie.")
        st.button("Check a rating", key="h3", on_click=go, args=("Rating check",))

    st.write("")
    with st.expander("How the recommendations work"):
        st.write(
            "The model is a matrix factorization (SVD) trained on past ratings. "
            "It learns hidden taste factors for each user and each movie, then "
            "predicts how a user would rate movies they haven't rated yet. "
            "The highest predictions become the recommendations."
        )
    with st.expander("Dataset and model details"):
        st.json(metadata)


# ----------------------------------------------------------------------
# Picks for you
# ----------------------------------------------------------------------

elif page == "Picks for you":
    need_valid_user()
    st.title("Picks for you")
    st.caption(f"Recommendations for user {user_id}. Results update as you change the filters.")

    f1, f2 = st.columns([2, 1])
    genre_choice = f1.selectbox("Genre", ["Any genre"] + genre_options())
    count = f2.slider("How many", 6, 30, 12, step=3)

    with st.spinner("Ranking movies..."):
        try:
            recs = cached_recommendations(
                recommender,
                user_id,
                count,
                None if genre_choice == "Any genre" else genre_choice,
            )
        except Exception as error:
            st.error(f"Couldn't generate recommendations: {error}")
            st.stop()

    if recs.empty:
        st.info(
            "Nothing left to recommend for this genre. "
            "Try another genre or choose Any genre."
        )
    else:
        for start in range(0, len(recs), 3):
            cols = st.columns(3)
            batch = recs.iloc[start:start + 3]
            for col, (idx, movie) in zip(cols, batch.iterrows()):
                score = float(movie["predicted_rating"])
                with col.container(border=True):
                    st.markdown(
                        f"<p class='movie-title'>{movie['title']}</p>"
                        f"<p class='muted'>{str(movie.get('genres', '')).replace('|', ', ')}</p>",
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(score / 5, 0.0), 1.0))
                    st.caption(f"Predicted rating {score:.2f} out of 5")
                    st.button(
                        "More like this",
                        key=f"similar_{start}_{idx}",
                        on_click=go,
                        args=("Find similar",),
                        kwargs={"similar_title": str(movie["title"])},
                        use_container_width=True,
                    )

    with st.expander("What this user rated highest"):
        history = cached_history(recommender, user_id)
        if history.empty:
            st.write("No ratings on record for this user.")
        else:
            cols = [c for c in ("title", "rating") if c in history.columns]
            st.dataframe(
                history[cols].head(20),
                use_container_width=True,
                hide_index=True,
            )


# ----------------------------------------------------------------------
# Find similar
# ----------------------------------------------------------------------

elif page == "Find similar":
    st.title("Find similar movies")
    st.caption("Start typing in the box to search the catalogue.")

    titles = movie_titles()
    if st.session_state.get("similar_title") not in titles:
        st.session_state["similar_title"] = titles[0]

    chosen = st.selectbox("Movie", titles, key="similar_title")
    row = movie_row(chosen)
    st.markdown(f"**Genres:** {str(row['genres']).replace('|', ', ')}")

    how_many = st.slider("How many results", 5, 20, 10)

    similar = recommender.similar_movies_by_genre(int(row["movieId"]), how_many)

    if similar.empty:
        st.info("This movie has no genre information, so there is nothing to compare.")
    else:
        table = similar[["title", "genres", "genre_similarity"]].copy()
        table["genre_similarity"] = (table["genre_similarity"] * 100).round(0)
        table["genres"] = table["genres"].astype(str).str.replace("|", ", ", regex=False)
        table = table.rename(
            columns={"title": "Movie", "genres": "Genres", "genre_similarity": "Genre match"}
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Genre match": st.column_config.ProgressColumn(
                    "Genre match",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )
        st.caption(
            "Matches are based on shared genres only (Jaccard overlap). "
            "They don't use other users' ratings."
        )

    if user_known:
        st.divider()
        st.button(
            f"How would user {user_id} rate this?",
            on_click=go,
            args=("Rating check",),
            kwargs={"rating_title": chosen},
        )


# ----------------------------------------------------------------------
# Rating check
# ----------------------------------------------------------------------

elif page == "Rating check":
    need_valid_user()
    st.title("Rating check")
    st.caption(f"Estimate how user {user_id} would rate a movie.")

    titles = movie_titles()
    if st.session_state.get("rating_title") not in titles:
        st.session_state["rating_title"] = titles[0]

    chosen = st.selectbox("Movie", titles, key="rating_title")
    row = movie_row(chosen)

    try:
        result = recommender.predict_rating(user_id, int(row["movieId"]))
        estimate = float(result["estimated_rating"])
    except Exception as error:
        st.error(f"Couldn't estimate a rating: {error}")
        st.stop()

    left, right = st.columns(2)
    left.metric("Estimated rating for this user", f"{estimate:.2f} / 5")

    stats = movie_rating_stats(chosen)
    if stats:
        avg, n = stats
        right.metric("Average of all users", f"{avg:.2f} / 5", f"{n:,} ratings", delta_color="off")
        diff = estimate - avg
        if abs(diff) < 0.15:
            st.write("This user is expected to rate it about the same as everyone else.")
        elif diff > 0:
            st.write("This user is expected to like it more than the average viewer.")
        else:
            st.write("This user is expected to like it less than the average viewer.")
    else:
        right.metric("Average of all users", "No ratings")

    st.caption("This is the model's estimate, not a rating the user gave.")


# ----------------------------------------------------------------------
# Model report
# ----------------------------------------------------------------------

elif page == "Model report":
    st.title("Model report")
    st.caption("How the models compared during evaluation.")

    comparison = recommender.get_model_comparison()

    st.markdown("### Rating prediction accuracy")
    st.write("Lower RMSE means predicted ratings sit closer to the real ones.")

    rmse_col = next((c for c in comparison.columns if c.lower() == "rmse"), None)
    model_col = next(
        (c for c in comparison.columns if c.lower() in ("model", "algorithm", "model_name")),
        None,
    )

    if rmse_col and model_col:
        plot = comparison[[model_col, rmse_col]].copy()
        plot[rmse_col] = pd.to_numeric(plot[rmse_col], errors="coerce")
        plot = plot.dropna()
        if not plot.empty:
            best = plot.loc[plot[rmse_col].idxmin()]
            st.metric("Best model", str(best[model_col]), f"RMSE {best[rmse_col]:.4f}", delta_color="off")
            st.bar_chart(plot.set_index(model_col)[rmse_col])

    with st.expander("Full comparison table"):
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("### Ranking quality")
    st.write("These measure how well the top of each recommendation list matches what users liked.")
    st.dataframe(recommender.get_ranking_metrics(), use_container_width=True, hide_index=True)
    st.caption(
        "Ranking scores depend on the candidate set and the relevance "
        "threshold used during evaluation."
    )


st.divider()
st.caption(f"{APP_NAME} · Academic project · Predictions are estimates, not guarantees.")