"""
Bollywood Movie Recommendation System
=====================================
Main Streamlit dashboard application.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from recommender import (
    load_data,
    build_recommendation_engine,
    get_content_recommendations,
    get_genre_recommendations,
    get_actor_director_recommendations,
    get_top_rated_movies,
    get_trending_movies,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bollywood Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – Cinematic dark-gold theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

    /* Root palette */
    :root {
        --gold:   #FFB81C;
        --gold2:  #E5960A;
        --deep:   #0D0D0D;
        --card:   #1A1A1A;
        --border: #2E2E2E;
        --muted:  #888888;
        --white:  #F5F0E8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--deep);
        color: var(--white);
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0D0D0D 0%, #1a0a00 50%, #0D0D0D 100%);
        border-bottom: 1px solid var(--border);
        padding: 2rem 0 1.5rem 0;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: var(--gold);
        margin: 0;
        letter-spacing: 2px;
    }
    .hero p {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Section headings */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: var(--gold);
        border-left: 4px solid var(--gold);
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Movie card */
    .movie-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .movie-card:hover { border-color: var(--gold); }
    .movie-card h4 {
        font-family: 'Playfair Display', serif;
        color: var(--gold);
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
    }
    .movie-card p {
        color: var(--muted);
        font-size: 0.82rem;
        margin: 0.15rem 0;
    }
    .badge {
        display: inline-block;
        background: #2a1f00;
        color: var(--gold);
        border: 1px solid var(--gold2);
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-top: 4px;
    }
    .star { color: var(--gold); font-size: 0.85rem; }
    .rank-num {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #2E2E2E;
        font-weight: 900;
        line-height: 1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

    /* Streamlit overrides */
    .stSelectbox label, .stMultiSelect label, .stSlider label,
    .stTextInput label, .stRadio label {
        color: var(--muted) !important;
        font-size: 0.85rem !important;
    }
    .stButton > button {
        background: var(--gold);
        color: #000;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover { background: var(--gold2); }

    div[data-testid="metric-container"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem;
    }
    div[data-testid="metric-container"] label { color: var(--muted) !important; }
    div[data-testid="metric-container"] div { color: var(--gold) !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { color: var(--muted); font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

    /* Watchlist */
    .watchlist-item {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# LOAD DATA & ENGINE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_engine(df):
    return build_recommendation_engine(df)

with st.spinner("Loading Bollywood universe…"):
    df = get_data()
    cosine_sim, indices = get_engine(df)

# ─────────────────────────────────────────────
# SESSION STATE – Watchlist
# ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🎬 Bollywood Recommender</h1>
        <p>Discover your next favourite Hindi film · Content-based AI recommendations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Playfair Display,serif;color:#FFB81C;font-size:1.2rem;margin-bottom:1rem;">🎛️ Filters & Preferences</p>',
        unsafe_allow_html=True,
    )

    # Genre multi-select
    all_genres = sorted(
        set(g.strip() for genres in df["genre"].dropna() for g in genres.split(","))
    )
    selected_genres = st.multiselect("🎭 Genre(s)", all_genres)

    # Year range
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("📅 Release Year", year_min, year_max, (2000, year_max))

    # Minimum rating
    min_rating = st.slider("⭐ Minimum Rating", 1.0, 10.0, 6.0, 0.5)

    # Actor / Director
    all_cast = sorted(
        set(a.strip() for casts in df["cast"].dropna() for a in casts.split(","))
    )
    selected_actor = st.selectbox("🎤 Actor / Actress", ["Any"] + all_cast)

    all_directors = sorted(df["director"].dropna().unique())
    selected_director = st.selectbox("🎥 Director", ["Any"] + list(all_directors))

    st.markdown("---")
    # Watchlist display
    st.markdown(
        '<p style="font-family:Playfair Display,serif;color:#FFB81C;font-size:1rem;">📋 My Watchlist</p>',
        unsafe_allow_html=True,
    )
    if st.session_state.watchlist:
        for wm in st.session_state.watchlist:
            st.markdown(
                f'<div class="watchlist-item"><span>🎞</span><span style="font-size:0.85rem">{wm}</span></div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
    else:
        st.markdown(
            '<p style="color:#555;font-size:0.82rem;">Add movies from recommendations below.</p>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# APPLY FILTERS to dataframe
# ─────────────────────────────────────────────
def apply_filters(data):
    filtered = data.copy()
    filtered = filtered[
        (filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])
    ]
    filtered = filtered[filtered["rating"] >= min_rating]
    if selected_genres:
        mask = filtered["genre"].apply(
            lambda g: any(sg.lower() in g.lower() for sg in selected_genres)
            if isinstance(g, str)
            else False
        )
        filtered = filtered[mask]
    if selected_actor != "Any":
        filtered = filtered[
            filtered["cast"].str.contains(selected_actor, case=False, na=False)
        ]
    if selected_director != "Any":
        filtered = filtered[
            filtered["director"].str.contains(selected_director, case=False, na=False)
        ]
    return filtered

filtered_df = apply_filters(df)

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎬 Total Movies", len(df))
col2.metric("🔍 Filtered Results", len(filtered_df))
col3.metric("⭐ Avg Rating (filtered)", f"{filtered_df['rating'].mean():.1f}" if len(filtered_df) else "—")
col4.metric("📅 Year Span", f"{year_range[0]}–{year_range[1]}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: render a movie card
# ─────────────────────────────────────────────
def render_movie_card(row, rank=None, show_watchlist_btn=True):
    genres_html = "".join(
        f'<span class="badge">{g.strip()}</span>'
        for g in str(row.get("genre", "")).split(",")
    )
    stars = "★" * int(round(row.get("rating", 0) / 2)) + "☆" * (5 - int(round(row.get("rating", 0) / 2)))
    rank_html = f'<span class="rank-num">#{rank}</span>' if rank else ""
    st.markdown(
        f"""
        <div class="movie-card">
            <div style="display:flex;gap:1rem;align-items:flex-start;">
                {rank_html}
                <div style="flex:1">
                    <h4>{row.get('title','')}</h4>
                    <p><b style="color:#ccc">{row.get('year','')} · {row.get('director','')}</b></p>
                    <p class="star">{stars} &nbsp;<span style="color:#FFB81C;font-weight:600">{row.get('rating','')}/10</span></p>
                    {genres_html}
                    <p style="margin-top:0.5rem;color:#aaa;font-size:0.8rem;line-height:1.4">{str(row.get('overview',''))[:160]}…</p>
                    <p style="color:#666;font-size:0.78rem">Cast: {str(row.get('cast',''))[:80]}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_watchlist_btn:
        title = row.get("title", "")
        if title not in st.session_state.watchlist:
            if st.button(f"+ Watchlist", key=f"wl_{title}_{rank}"):
                st.session_state.watchlist.append(title)
                st.rerun()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Search & Recommend", "🏆 Trending / Top-Rated", "📊 Visualizations", "🎭 Browse Filtered", "🔗 Similar Movies"]
)

# ══════════════════════════════════════════════
# TAB 1 – SEARCH & RECOMMEND
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Find Movies Like…</div>', unsafe_allow_html=True)

    search_mode = st.radio(
        "Recommendation mode",
        ["By Movie Title", "By Genre", "By Actor / Director"],
        horizontal=True,
    )

    recommendations = pd.DataFrame()

    if search_mode == "By Movie Title":
        movie_titles = sorted(df["title"].tolist())
        selected_movie = st.selectbox("🎬 Select a Bollywood Movie", movie_titles)
        n_recs = st.slider("Number of recommendations", 3, 15, 8)

        if st.button("🎯 Get Recommendations"):
            recommendations = get_content_recommendations(
                selected_movie, df, cosine_sim, indices, n=n_recs
            )

    elif search_mode == "By Genre":
        genre_choice = st.multiselect("🎭 Select Genre(s)", all_genres, default=["Action"])
        n_recs = st.slider("Number of recommendations", 3, 15, 8)
        if st.button("🎯 Get Recommendations"):
            if genre_choice:
                recommendations = get_genre_recommendations(df, genre_choice, n=n_recs)
            else:
                st.warning("Please select at least one genre.")

    else:  # By Actor / Director
        pref_actor = st.selectbox("🎤 Actor / Actress", ["Any"] + all_cast, key="rec_actor")
        pref_director = st.selectbox("🎥 Director", ["Any"] + list(all_directors), key="rec_dir")
        n_recs = st.slider("Number of recommendations", 3, 15, 8)
        if st.button("🎯 Get Recommendations"):
            actor_val = None if pref_actor == "Any" else pref_actor
            dir_val = None if pref_director == "Any" else pref_director
            recommendations = get_actor_director_recommendations(
                df, actor=actor_val, director=dir_val, n=n_recs
            )

    if not recommendations.empty:
        st.markdown(
            f'<div class="section-title">🎬 {len(recommendations)} Recommendations</div>',
            unsafe_allow_html=True,
        )
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            render_movie_card(row, rank=idx)
    elif "Get Recommendations" in str(st.session_state):
        st.info("No recommendations found. Try adjusting your filters or selection.")

# ══════════════════════════════════════════════
# TAB 2 – TRENDING / TOP-RATED
# ══════════════════════════════════════════════
with tab2:
    col_tr, col_tp = st.columns(2)

    with col_tr:
        st.markdown('<div class="section-title">🔥 Trending (Recent & Popular)</div>', unsafe_allow_html=True)
        trending = get_trending_movies(df, n=8)
        for idx, (_, row) in enumerate(trending.iterrows(), 1):
            render_movie_card(row, rank=idx)

    with col_tp:
        st.markdown('<div class="section-title">🏆 All-Time Top-Rated</div>', unsafe_allow_html=True)
        top_rated = get_top_rated_movies(df, n=8)
        for idx, (_, row) in enumerate(top_rated.iterrows(), 1):
            render_movie_card(row, rank=idx)

# ══════════════════════════════════════════════
# TAB 3 – VISUALIZATIONS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📊 Dataset Insights</div>', unsafe_allow_html=True)

    # Dark matplotlib style
    plt.rcParams.update({
        "figure.facecolor": "#0D0D0D",
        "axes.facecolor": "#1A1A1A",
        "axes.edgecolor": "#2E2E2E",
        "axes.labelcolor": "#F5F0E8",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "text.color": "#F5F0E8",
        "grid.color": "#2E2E2E",
    })

    vc1, vc2 = st.columns(2)

    # ── Genre distribution (pie chart)
    with vc1:
        st.markdown("**🎭 Genre Distribution**")
        genre_counts = {}
        for genres in df["genre"].dropna():
            for g in genres.split(","):
                g = g.strip()
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_series = pd.Series(genre_counts).sort_values(ascending=False).head(10)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = plt.cm.get_cmap("YlOrBr")(np.linspace(0.3, 1.0, len(genre_series)))
        wedges, texts, autotexts = ax1.pie(
            genre_series.values,
            labels=genre_series.index,
            autopct="%1.0f%%",
            colors=colors,
            startangle=140,
            textprops={"fontsize": 8},
        )
        for at in autotexts:
            at.set_color("#0D0D0D")
            at.set_fontsize(7)
        ax1.set_title("Top 10 Genres", color="#FFB81C", fontsize=11)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # ── Ratings histogram
    with vc2:
        st.markdown("**⭐ Ratings Distribution**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        n, bins, patches = ax2.hist(df["rating"], bins=20, color="#FFB81C", edgecolor="#0D0D0D", alpha=0.85)
        ax2.axvline(df["rating"].mean(), color="#E5960A", linestyle="--", linewidth=1.5, label=f"Mean: {df['rating'].mean():.1f}")
        ax2.set_xlabel("Rating (out of 10)")
        ax2.set_ylabel("Number of Movies")
        ax2.set_title("Rating Distribution", color="#FFB81C", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    vc3, vc4 = st.columns(2)

    # ── Movies per year
    with vc3:
        st.markdown("**📅 Movies Released per Year**")
        year_counts = df["year"].value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.fill_between(year_counts.index, year_counts.values, color="#FFB81C", alpha=0.3)
        ax3.plot(year_counts.index, year_counts.values, color="#FFB81C", linewidth=2)
        ax3.set_xlabel("Year")
        ax3.set_ylabel("# Movies")
        ax3.set_title("Release Trend", color="#FFB81C", fontsize=11)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ── Top directors by avg rating
    with vc4:
        st.markdown("**🎥 Top Directors (Avg Rating)**")
        dir_rating = (
            df.groupby("director")["rating"]
            .agg(["mean", "count"])
            .query("count >= 2")
            .sort_values("mean", ascending=False)
            .head(10)
        )
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        bars = ax4.barh(dir_rating.index[::-1], dir_rating["mean"][::-1], color="#FFB81C", edgecolor="#0D0D0D")
        ax4.set_xlabel("Avg Rating")
        ax4.set_title("Top Directors", color="#FFB81C", fontsize=11)
        ax4.set_xlim(0, 10)
        ax4.grid(True, alpha=0.3, axis="x")
        for bar in bars:
            w = bar.get_width()
            ax4.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=7, color="#FFB81C")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ── Correlation heatmap
    st.markdown("**🔗 Feature Correlation Heatmap**")
    num_cols = df[["rating", "year"]].copy()
    num_cols["genre_count"] = df["genre"].apply(lambda g: len(g.split(",")) if isinstance(g, str) else 0)
    num_cols["cast_count"] = df["cast"].apply(lambda c: len(c.split(",")) if isinstance(c, str) else 0)
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    sns.heatmap(
        num_cols.corr(),
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        ax=ax5,
        linewidths=0.5,
        linecolor="#0D0D0D",
        annot_kws={"size": 9},
    )
    ax5.set_title("Correlation Matrix", color="#FFB81C", fontsize=11)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

# ══════════════════════════════════════════════
# TAB 4 – BROWSE FILTERED
# ══════════════════════════════════════════════
with tab4:
    st.markdown(
        f'<div class="section-title">🎭 Browse Movies ({len(filtered_df)} results)</div>',
        unsafe_allow_html=True,
    )
    if filtered_df.empty:
        st.warning("No movies match your current filters. Adjust the sidebar settings.")
    else:
        sort_col = st.selectbox("Sort by", ["rating", "year", "title"], index=0)
        sort_asc = st.checkbox("Ascending", value=False)
        sorted_df = filtered_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
        page_size = 10
        total_pages = max(1, (len(sorted_df) - 1) // page_size + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page - 1) * page_size
        page_df = sorted_df.iloc[start : start + page_size]
        st.caption(f"Showing {start + 1}–{min(start + page_size, len(sorted_df))} of {len(sorted_df)} movies · Page {page}/{total_pages}")
        for idx, (_, row) in enumerate(page_df.iterrows(), start + 1):
            render_movie_card(row, rank=idx)

# ══════════════════════════════════════════════
# TAB 5 – SIMILAR MOVIE COMPARISON
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🔗 Similar Movie Comparison</div>', unsafe_allow_html=True)
    st.caption("Pick two movies and compare their similarity score and attributes side by side.")

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        movie_a = st.selectbox("🎬 Movie A", sorted(df["title"].tolist()), key="cmp_a")
    with cmp2:
        movie_b = st.selectbox("🎬 Movie B", sorted(df["title"].tolist()), index=1, key="cmp_b")

    if st.button("🔍 Compare Movies"):
        if movie_a == movie_b:
            st.warning("Please select two different movies.")
        else:
            row_a = df[df["title"] == movie_a].iloc[0]
            row_b = df[df["title"] == movie_b].iloc[0]

            # Compute similarity score between the two
            idx_a = indices.get(movie_a)
            idx_b = indices.get(movie_b)
            if idx_a is not None and idx_b is not None:
                sim_score = float(cosine_sim[idx_a][idx_b]) * 100
            else:
                sim_score = 0.0

            st.markdown(
                f"""
                <div style="text-align:center;padding:1rem 0;">
                    <span style="font-family:Playfair Display,serif;font-size:1.8rem;color:#FFB81C;">
                        Similarity Score: {sim_score:.1f}%
                    </span>
                    <br><span style="color:#888;font-size:0.85rem;">Based on TF-IDF cosine similarity of combined features</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Side-by-side cards
            ca, cb = st.columns(2)
            with ca:
                render_movie_card(row_a, show_watchlist_btn=False)
            with cb:
                render_movie_card(row_b, show_watchlist_btn=False)

            # Attribute comparison table
            compare_data = {
                "Attribute": ["Year", "Rating", "Director", "Genre"],
                movie_a: [row_a["year"], row_a["rating"], row_a["director"], row_a["genre"]],
                movie_b: [row_b["year"], row_b["rating"], row_b["director"], row_b["genre"]],
            }
            st.dataframe(
                pd.DataFrame(compare_data).set_index("Attribute"),
                use_container_width=True,
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;color:#333;font-size:0.78rem;padding:2rem 0 0.5rem 0;border-top:1px solid #1E1E1E;margin-top:2rem;">
        Bollywood Recommender · Built with Streamlit, Scikit-learn & ❤️ for Hindi Cinema
    </div>
    """,
    unsafe_allow_html=True,
)
