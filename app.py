# ============================================================
# BICS 2303 – Intelligent Systems | Group Project
# Sephora Skincare Review Sentiment Analysis
# Streamlit Web App — Organisation Dashboard
#
# HOW TO RUN:
#   1. pip install streamlit pandas scikit-learn plotly
#      vaderSentiment nltk
#   2. Place reviews_sample.csv in the same folder
#   3. streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

import plotly.express as px
import plotly.graph_objects as go

# ── Download NLTK data ────────────────────────────────────
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt_tab', quiet=True)


# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Sephora Sentiment Dashboard",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #fafafa; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border-top: 4px solid #ff0050;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .metric-label {
        font-size: 13px;
        color: #888;
        margin: 0;
    }

    /* Page title */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .page-subtitle {
        font-size: 14px;
        color: #888;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a2e;
        padding-bottom: 6px;
        border-bottom: 2px solid #ff0050;
        margin-bottom: 1rem;
    }

    /* Sentiment badges */
    .badge-pos { background:#e8f5e9; color:#2e7d32; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-neg { background:#fce4ec; color:#c62828; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-neu { background:#e3f2fd; color:#1565c0; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_text(text):
    """
    Clean and preprocess a review text.
    Steps: lowercase → remove URLs → remove punctuation
           → tokenise → remove stopwords → lemmatise
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text   = str(text).lower()
    text   = re.sub(r"http\S+|www\S+", "", text)
    text   = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


def get_vader_label(text):
    """
    Use VADER to label a review as Positive, Negative, or Neutral
    based on the compound score of the text.
    """
    analyzer = SentimentIntensityAnalyzer()
    score    = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_compound_score(text):
    """Return VADER compound score for a review."""
    analyzer = SentimentIntensityAnalyzer()
    return round(analyzer.polarity_scores(str(text))["compound"], 4)


def sentiment_emoji(label):
    """Return emoji for a given sentiment label."""
    return {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(label, "❓")


def sentiment_color(label):
    """Return hex color for a given sentiment label."""
    return {"Positive": "#4CAF50", "Negative": "#f44336", "Neutral": "#2196F3"}.get(label, "#888")


# ============================================================
# DATA LOADING AND MODEL TRAINING (cached — runs only once)
# ============================================================

@st.cache_resource(show_spinner="Loading and training models... please wait ⏳")
def load_data_and_train():
    """
    Load the dataset, preprocess text, train all models,
    and return everything needed for the dashboard.
    This is cached so it only runs once per session.
    """

    # ── Step 1: Load CSV ──────────────────────────────────
    df = pd.read_csv("reviews_sample.csv")


    # Keep only columns we need
    keep = [c for c in ["review_text", "brand_name", "product_name"] if c in df.columns]
    df   = df[keep].copy()
    df.dropna(subset=["review_text"], inplace=True)
    df   = df[df["review_text"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    # ── Step 2: NLP Preprocessing ─────────────────────────
    df["clean_review"]   = df["review_text"].apply(preprocess_text)
    df["sentiment"]      = df["review_text"].apply(get_vader_label)
    df["compound_score"] = df["review_text"].apply(get_compound_score)

    df = df[df["clean_review"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    # ── Step 3: TF-IDF Vectorisation ─────────────────────
    tfidf   = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(df["clean_review"])

    # ── Step 4: Train Naive Bayes ─────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df["sentiment"])
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_tfidf, y)

    # Add NB predictions and confidence to dataframe
    probas           = nb.predict_proba(X_tfidf)
    df["nb_label"]   = le.inverse_transform(nb.predict(X_tfidf))
    df["confidence"] = [round(float(p.max()) * 100, 1) for p in probas]

    # ── Step 5: Dimensionality Reduction for Clustering ───
    svd       = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    compound  = df["compound_score"].values.reshape(-1, 1)
    X_cluster = np.hstack([X_reduced, compound])
    scaler    = MinMaxScaler()
    X_cluster = scaler.fit_transform(X_cluster)

    # ── Step 6: K-Means Clustering ────────────────────────
    best_k, best_km_score = 3, -1
    for k in range(2, 7):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        score  = silhouette_score(X_cluster, labels, sample_size=2000)
        if score > best_km_score:
            best_km_score = score
            best_k        = k

    kmeans           = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["km_cluster"] = kmeans.fit_predict(X_cluster)
    km_sil           = silhouette_score(X_cluster, df["km_cluster"], sample_size=2000)
    km_inertia       = kmeans.inertia_

    # ── Step 7: DBSCAN Clustering ─────────────────────────
    best_eps, best_db_sil, best_db_labels = 0.5, -1, None
    for eps in [0.3, 0.5, 0.7, 1.0]:
        db     = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_cluster)
        n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
        if n_cl >= 2:
            score = silhouette_score(X_cluster, labels, sample_size=2000)
            if score > best_db_sil:
                best_db_sil   = score
                best_eps      = eps
                best_db_labels = labels

    df["db_cluster"] = best_db_labels if best_db_labels is not None else np.zeros(len(df), dtype=int)
    db_n_clusters    = len(set(df["db_cluster"])) - (1 if -1 in df["db_cluster"].values else 0)
    db_noise         = int((df["db_cluster"] == -1).sum())

    # ── Step 8: Save model for Real-Time Prediction ───────
    with open("model.pkl", "wb") as f:
        pickle.dump({"nb": nb, "tfidf": tfidf, "le": le}, f)

    return {
        "df"          : df,
        "nb"          : nb,
        "tfidf"       : tfidf,
        "le"          : le,
        "best_k"      : best_k,
        "km_sil"      : round(km_sil, 4),
        "km_inertia"  : round(km_inertia, 2),
        "best_eps"    : best_eps,
        "best_db_sil" : round(best_db_sil, 4),
        "db_n_clusters": db_n_clusters,
        "db_noise"    : db_noise,
        "X_cluster"   : X_cluster,
    }


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 💄 Sephora Sentiment")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "📊  Sentiment Dashboard",
            "🏷️  Brand Analysis",
            "🔵  Clustering Analysis",
            "🔮  Real-Time Prediction",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**BICS 2303**")
    st.markdown("Intelligent Systems")
    st.markdown("Group Project")
    st.markdown("---")
    st.markdown("Dataset: Sephora Reviews")
    st.markdown("20,000 reviews sampled")


# ============================================================
# LOAD DATA
# ============================================================

# Check if CSV exists
if not os.path.exists("reviews_sample.csv"):
    st.error("❌ reviews_sample.csv not found! Please place it in the same folder as app.py")
    st.stop()

# Load everything
data = load_data_and_train()
df         = data["df"]
nb         = data["nb"]
tfidf      = data["tfidf"]
le         = data["le"]
best_k     = data["best_k"]
km_sil     = data["km_sil"]
km_inertia = data["km_inertia"]
best_eps   = data["best_eps"]
best_db_sil    = data["best_db_sil"]
db_n_clusters  = data["db_n_clusters"]
db_noise       = data["db_noise"]


# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "🏠  Home":

    st.markdown('<p class="page-title">💄 Sephora Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Organisation Analytics — Based purely on review text. Rating column not used.</p>', unsafe_allow_html=True)

    # ── Project overview ──────────────────────────────────
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Project Overview")
        st.markdown("""
        This dashboard analyses **Sephora skincare product reviews** to automatically determine
        whether each review is **Positive**, **Negative**, or **Neutral** — using the review text only.

        **Two phases:**
        - **Phase 1 — Sentiment Analysis:** VADER labels reviews automatically. Naive Bayes predicts sentiment.
        - **Phase 2 — Clustering:** K-Means and DBSCAN group similar reviews to find hidden patterns.

        **Business goal:** Identify which brands and products get the most positive and negative feedback.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Dataset summary metrics ───────────────────────────
    st.markdown('<p class="section-title">📦 Dataset Summary</p>', unsafe_allow_html=True)

    total   = len(df)
    n_pos   = len(df[df["sentiment"] == "Positive"])
    n_neg   = len(df[df["sentiment"] == "Negative"])
    n_neu   = len(df[df["sentiment"] == "Neutral"])
    avg_conf = round(df["confidence"].mean(), 1)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""<div class="metric-box">
            <p class="metric-value">{total:,}</p>
            <p class="metric-label">Total Reviews</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#4CAF50">
            <p class="metric-value" style="color:#4CAF50">{n_pos:,}</p>
            <p class="metric-label">😊 Positive</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#f44336">
            <p class="metric-value" style="color:#f44336">{n_neg:,}</p>
            <p class="metric-label">😞 Negative</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#2196F3">
            <p class="metric-value" style="color:#2196F3">{n_neu:,}</p>
            <p class="metric-label">😐 Neutral</p>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#FF9800">
            <p class="metric-value" style="color:#FF9800">{avg_conf}%</p>
            <p class="metric-label">Avg Confidence</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sentiment distribution bar chart ──────────────────
    st.markdown('<p class="section-title">📊 Sentiment Distribution Overview</p>', unsafe_allow_html=True)

    dist_df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count"    : [n_pos, n_neg, n_neu],
        "Percent"  : [
            round(n_pos/total*100, 1),
            round(n_neg/total*100, 1),
            round(n_neu/total*100, 1)
        ]
    })

    fig = px.bar(
        dist_df, x="Sentiment", y="Count",
        color="Sentiment",
        color_discrete_map={"Positive":"#4CAF50","Negative":"#f44336","Neutral":"#2196F3"},
        text="Percent",
        title="Number of reviews per sentiment class"
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ── Dataset info table ────────────────────────────────
    st.markdown('<p class="section-title">📁 Dataset Information</p>', unsafe_allow_html=True)

    info = pd.DataFrame({
        "Item"  : ["Source", "File", "Rows sampled", "Labelling method", "Rating used?", "Columns used"],
        "Detail": [
            "Kaggle — nadyinky/sephora-products-and-skincare-reviews",
            "reviews_sample.csv",
            "20,000 (random_state=42)",
            "VADER (text-based, automatic)",
            "❌ No — text only",
            "review_text, brand_name, product_name"
        ]
    })
    st.dataframe(info, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2 — SENTIMENT DASHBOARD
# ============================================================

elif page == "📊  Sentiment Dashboard":

    st.markdown('<p class="page-title">📊 Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">How positive, negative, and neutral are the Sephora reviews?</p>', unsafe_allow_html=True)

    total = len(df)
    n_pos = len(df[df["sentiment"] == "Positive"])
    n_neg = len(df[df["sentiment"] == "Negative"])
    n_neu = len(df[df["sentiment"] == "Neutral"])

    # ── Pie chart + Bar chart ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Sentiment Split (Pie Chart)</p>', unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[n_pos, n_neg, n_neu],
            names=["Positive", "Negative", "Neutral"],
            color_discrete_sequence=["#4CAF50", "#f44336", "#2196F3"],
            hole=0.4
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Sentiment Count (Bar Chart)</p>', unsafe_allow_html=True)
        fig_bar = px.bar(
            x=["Positive", "Negative", "Neutral"],
            y=[n_pos, n_neg, n_neu],
            color=["Positive", "Negative", "Neutral"],
            color_discrete_map={"Positive":"#4CAF50","Negative":"#f44336","Neutral":"#2196F3"},
            labels={"x": "Sentiment", "y": "Count"},
            text=[n_pos, n_neg, n_neu]
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, plot_bgcolor="white", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Confidence distribution ───────────────────────────
    st.markdown('<p class="section-title">Model Confidence Distribution</p>', unsafe_allow_html=True)
    fig_conf = px.histogram(
        df, x="confidence", color="sentiment",
        color_discrete_map={"Positive":"#4CAF50","Negative":"#f44336","Neutral":"#2196F3"},
        nbins=20,
        labels={"confidence":"Confidence %", "count":"Number of Reviews"},
        title="How confident is the Naive Bayes model for each prediction?"
    )
    fig_conf.update_layout(plot_bgcolor="white", height=320)
    st.plotly_chart(fig_conf, use_container_width=True)

    # ── Sample reviews by sentiment ───────────────────────
    st.markdown('<p class="section-title">Sample Reviews by Sentiment</p>', unsafe_allow_html=True)

    tab_pos, tab_neg, tab_neu = st.tabs(["😊 Positive Reviews", "😞 Negative Reviews", "😐 Neutral Reviews"])

    def show_sample_reviews(sentiment, n=5):
        """Show n sample reviews for a given sentiment."""
        samples = df[df["sentiment"] == sentiment].head(n)
        for _, row in samples.iterrows():
            st.markdown(f"""
            <div class="card" style="border-left: 4px solid {sentiment_color(sentiment)};">
                <p style="font-size:14px; color:#333; margin:0;">
                    "{str(row['review_text'])[:200]}..."
                </p>
                <p style="font-size:12px; color:#888; margin:4px 0 0;">
                    Confidence: <b>{row['confidence']}%</b> &nbsp;|&nbsp;
                    VADER score: <b>{row['compound_score']}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab_pos:
        show_sample_reviews("Positive")
    with tab_neg:
        show_sample_reviews("Negative")
    with tab_neu:
        show_sample_reviews("Neutral")


# ============================================================
# PAGE 3 — BRAND ANALYSIS
# ============================================================

elif page == "🏷️  Brand Analysis":

    st.markdown('<p class="page-title">🏷️ Brand & Product Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Which brands and products receive the most positive and negative reviews?</p>', unsafe_allow_html=True)

    if "brand_name" not in df.columns:
        st.warning("brand_name column not found in dataset.")
    else:

        # ── Top 5 brands — positive ───────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="section-title">Top 5 Brands — Most Positive Reviews</p>', unsafe_allow_html=True)
            top_pos_brands = (
                df[df["sentiment"] == "Positive"]
                .groupby("brand_name").size()
                .sort_values(ascending=False)
                .head(5).reset_index()
            )
            top_pos_brands.columns = ["Brand", "Positive Reviews"]
            fig = px.bar(
                top_pos_brands, x="Positive Reviews", y="Brand",
                orientation="h", color_discrete_sequence=["#4CAF50"],
                text="Positive Reviews"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                plot_bgcolor="white", height=300,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="section-title">Top 5 Brands — Most Negative Reviews</p>', unsafe_allow_html=True)
            top_neg_brands = (
                df[df["sentiment"] == "Negative"]
                .groupby("brand_name").size()
                .sort_values(ascending=False)
                .head(5).reset_index()
            )
            top_neg_brands.columns = ["Brand", "Negative Reviews"]
            fig2 = px.bar(
                top_neg_brands, x="Negative Reviews", y="Brand",
                orientation="h", color_discrete_sequence=["#f44336"],
                text="Negative Reviews"
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(
                plot_bgcolor="white", height=300,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Top 5 products — positive ─────────────────────
        if "product_name" in df.columns:
            col3, col4 = st.columns(2)

            with col3:
                st.markdown('<p class="section-title">Top 5 Products — Most Positive Reviews</p>', unsafe_allow_html=True)
                top_pos_prods = (
                    df[df["sentiment"] == "Positive"]
                    .groupby("product_name").size()
                    .sort_values(ascending=False)
                    .head(5).reset_index()
                )
                top_pos_prods.columns = ["Product", "Positive Reviews"]
                top_pos_prods["Product"] = top_pos_prods["Product"].str[:30]
                fig3 = px.bar(
                    top_pos_prods, x="Positive Reviews", y="Product",
                    orientation="h", color_discrete_sequence=["#66BB6A"],
                    text="Positive Reviews"
                )
                fig3.update_traces(textposition="outside")
                fig3.update_layout(
                    plot_bgcolor="white", height=300,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                st.markdown('<p class="section-title">Top 5 Products — Most Negative Reviews</p>', unsafe_allow_html=True)
                top_neg_prods = (
                    df[df["sentiment"] == "Negative"]
                    .groupby("product_name").size()
                    .sort_values(ascending=False)
                    .head(5).reset_index()
                )
                top_neg_prods.columns = ["Product", "Negative Reviews"]
                top_neg_prods["Product"] = top_neg_prods["Product"].str[:30]
                fig4 = px.bar(
                    top_neg_prods, x="Negative Reviews", y="Product",
                    orientation="h", color_discrete_sequence=["#EF5350"],
                    text="Negative Reviews"
                )
                fig4.update_traces(textposition="outside")
                fig4.update_layout(
                    plot_bgcolor="white", height=300,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig4, use_container_width=True)

        # ── Average sentiment score by brand ──────────────
        st.markdown('<p class="section-title">Average Sentiment Score by Brand (Top 10)</p>', unsafe_allow_html=True)

        top10_brands = df["brand_name"].value_counts().head(10).index
        avg_scores   = (
            df[df["brand_name"].isin(top10_brands)]
            .groupby("brand_name")["compound_score"]
            .mean().round(3)
            .sort_values(ascending=False)
            .reset_index()
        )
        avg_scores.columns = ["Brand", "Avg Compound Score"]

        fig5 = px.bar(
            avg_scores, x="Brand", y="Avg Compound Score",
            color="Avg Compound Score",
            color_continuous_scale=["#f44336", "#FF9800", "#4CAF50"],
            text="Avg Compound Score",
            title="Higher score = more positive overall sentiment"
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig5, use_container_width=True)

        # ── Full brand breakdown table ─────────────────────
        st.markdown('<p class="section-title">Full Brand Sentiment Breakdown (Top 15 Brands)</p>', unsafe_allow_html=True)

        top15       = df["brand_name"].value_counts().head(15).index
        brand_table = (
            df[df["brand_name"].isin(top15)]
            .groupby(["brand_name", "sentiment"])
            .size().unstack(fill_value=0)
        )
        for col in ["Positive", "Negative", "Neutral"]:
            if col not in brand_table.columns:
                brand_table[col] = 0

        brand_table["Total"] = brand_table.sum(axis=1)
        brand_table["Pos %"] = (brand_table["Positive"] / brand_table["Total"] * 100).round(1)
        brand_table["Neg %"] = (brand_table["Negative"] / brand_table["Total"] * 100).round(1)
        brand_table = brand_table.sort_values("Pos %", ascending=False)
        brand_table.index.name = "Brand"

        st.dataframe(brand_table, use_container_width=True)


# ============================================================
# PAGE 4 — CLUSTERING ANALYSIS
# ============================================================

elif page == "🔵  Clustering Analysis":

    st.markdown('<p class="page-title">🔵 Clustering Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">K-Means vs DBSCAN — which algorithm clusters reviews better?</p>', unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics = [
        ("Best K",          str(best_k),          "#ff0050"),
        ("K-Means Sil.",    str(km_sil),           "#4CAF50"),
        ("Inertia",         f"{km_inertia:,.0f}",  "#FF9800"),
        ("DBSCAN Clusters", str(db_n_clusters),    "#2196F3"),
        ("Noise Points",    f"{db_noise:,}",       "#9C27B0"),
        ("DBSCAN Sil.",     str(best_db_sil),      "#00BCD4"),
    ]
    for col, (label, value, color) in zip([col1,col2,col3,col4,col5,col6], metrics):
        with col:
            st.markdown(f"""<div class="metric-box" style="border-top-color:{color}">
                <p class="metric-value" style="color:{color}; font-size:20px">{value}</p>
                <p class="metric-label">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── K-Means visualization ─────────────────────────────
    st.markdown('<p class="section-title">K-Means Clustering Results</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Reviews per cluster bar chart
        km_dist = df["km_cluster"].value_counts().sort_index().reset_index()
        km_dist.columns = ["Cluster", "Reviews"]
        km_dist["Cluster"] = km_dist["Cluster"].apply(lambda x: f"Cluster {x}")

        fig_km = px.bar(
            km_dist, x="Cluster", y="Reviews",
            color="Cluster", text="Reviews",
            title=f"Reviews per K-Means cluster (K={best_k})"
        )
        fig_km.update_traces(textposition="outside")
        fig_km.update_layout(showlegend=False, plot_bgcolor="white", height=320)
        st.plotly_chart(fig_km, use_container_width=True)

    with col2:
        # Dominant sentiment per cluster
        st.markdown("**Dominant sentiment per cluster:**")
        st.markdown("<br>", unsafe_allow_html=True)

        for c in sorted(df["km_cluster"].unique()):
            cluster_df  = df[df["km_cluster"] == c]
            top_label   = cluster_df["sentiment"].value_counts().index[0]
            top_count   = cluster_df["sentiment"].value_counts().iloc[0]
            total_c     = len(cluster_df)
            avg_score   = round(cluster_df["compound_score"].mean(), 3)
            emoji       = sentiment_emoji(top_label)
            color       = sentiment_color(top_label)

            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color}; margin-bottom:8px;">
                <b>Cluster {c}</b> — {emoji} mostly <b>{top_label}</b><br>
                <span style="font-size:12px; color:#666;">
                    {total_c:,} reviews &nbsp;|&nbsp;
                    Dominant: {top_count:,} ({round(top_count/total_c*100,1)}%) &nbsp;|&nbsp;
                    Avg score: {avg_score}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── DBSCAN visualization ──────────────────────────────
    st.markdown('<p class="section-title">DBSCAN Clustering Results</p>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        db_dist = df["db_cluster"].value_counts().sort_index().reset_index()
        db_dist.columns = ["Cluster", "Reviews"]
        db_dist["Label"] = db_dist["Cluster"].apply(
            lambda x: "Noise (-1)" if x == -1 else f"Cluster {x}"
        )
        fig_db = px.bar(
            db_dist, x="Label", y="Reviews",
            color="Label", text="Reviews",
            title=f"Reviews per DBSCAN cluster (eps={best_eps})"
        )
        fig_db.update_traces(textposition="outside")
        fig_db.update_layout(showlegend=False, plot_bgcolor="white", height=320)
        st.plotly_chart(fig_db, use_container_width=True)

    with col4:
        st.markdown("**DBSCAN cluster info:**")
        st.markdown("<br>", unsafe_allow_html=True)

        for c in sorted(df["db_cluster"].unique()):
            label_name = "Noise (outliers)" if c == -1 else f"Cluster {c}"
            cluster_df = df[df["db_cluster"] == c]
            count      = len(cluster_df)
            pct        = round(count / len(df) * 100, 1)
            color      = "#9C27B0" if c == -1 else "#2196F3"

            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color}; margin-bottom:8px;">
                <b>{label_name}</b><br>
                <span style="font-size:12px; color:#666;">
                    {count:,} reviews ({pct}% of total)
                    {"— reviews that don't fit any cluster" if c == -1 else ""}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparison table ──────────────────────────────────
    st.markdown('<p class="section-title">K-Means vs DBSCAN Comparison</p>', unsafe_allow_html=True)

    winner = "K-Means" if km_sil >= best_db_sil else "DBSCAN"

    comp_df = pd.DataFrame({
        "Metric"              : ["Silhouette Score", "Clusters Found", "Noise Points", "Needs K upfront?", "Detects outliers?", "Cluster shape assumed"],
        "K-Means"             : [km_sil,   best_k,        "0",   "Yes", "No",  "Spherical"],
        "DBSCAN"              : [best_db_sil, db_n_clusters, f"{db_noise:,}", "No",  "Yes", "Any shape"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if winner == "K-Means":
        st.success(f"✅ K-Means performed better — Silhouette Score: {km_sil} vs {best_db_sil}")
    else:
        st.success(f"✅ DBSCAN performed better — Silhouette Score: {best_db_sil} vs {km_sil}")

    # ── What is Silhouette Score ───────────────────────────
    with st.expander("ℹ️ What is Silhouette Score?"):
        st.markdown("""
        **Silhouette Score** measures how well each review fits its assigned cluster.

        - **Score close to +1.0** → reviews are well-separated and clearly grouped
        - **Score near 0** → reviews are on the boundary between clusters
        - **Negative score** → reviews may be in the wrong cluster

        Formula: `(b - a) / max(a, b)`
        where `a` = average distance to same-cluster reviews,
        `b` = average distance to nearest other cluster reviews.
        """)


# ============================================================
# PAGE 5 — REAL-TIME PREDICTION
# ============================================================

elif page == "🔮  Real-Time Prediction":

    st.markdown('<p class="page-title">🔮 Real-Time Sentiment Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Type any skincare review — the model predicts its sentiment instantly.</p>', unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────
    st.markdown('<p class="section-title">Enter a Review</p>', unsafe_allow_html=True)

    user_review = st.text_area(
        "Type or paste a skincare product review here:",
        placeholder="e.g. This moisturiser is absolutely amazing! My skin feels so soft and hydrated...",
        height=140,
        label_visibility="collapsed"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        predict_btn = st.button("🔮 Predict Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=False)

    # ── Prediction ────────────────────────────────────────
    if predict_btn:
        if user_review.strip() == "":
            st.warning("⚠️ Please enter a review first!")
        else:
            # Preprocess and predict
            cleaned    = preprocess_text(user_review)
            vectorised = tfidf.transform([cleaned])
            pred       = nb.predict(vectorised)[0]
            proba      = nb.predict_proba(vectorised)[0]
            label      = le.inverse_transform([pred])[0]
            conf       = round(float(proba[pred]) * 100, 1)
            vader_sc   = get_compound_score(user_review)
            vader_lbl  = get_vader_label(user_review)
            emoji      = sentiment_emoji(label)
            color      = sentiment_color(label)

            # Result display
            st.markdown("---")
            st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class="metric-box" style="border-top-color:{color}">
                    <p class="metric-value">{emoji} {label}</p>
                    <p class="metric-label">Naive Bayes Prediction</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-box" style="border-top-color:#FF9800">
                    <p class="metric-value">{conf}%</p>
                    <p class="metric-label">Confidence Score</p>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="metric-box" style="border-top-color:#9C27B0">
                    <p class="metric-value">{vader_sc}</p>
                    <p class="metric-label">VADER Compound Score</p>
                </div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class="metric-box" style="border-top-color:{sentiment_color(vader_lbl)}">
                    <p class="metric-value">{sentiment_emoji(vader_lbl)} {vader_lbl}</p>
                    <p class="metric-label">VADER Label</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Colour-coded result banner
            if label == "Positive":
                st.success(f"😊 This review expresses a **positive** experience! The model is {conf}% confident.")
            elif label == "Negative":
                st.error(f"😞 This review expresses a **negative** experience. The model is {conf}% confident.")
            else:
                st.info(f"😐 This review is **neutral** — no strong sentiment detected. Confidence: {conf}%.")

            # Probability breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Probability breakdown for all three classes:**")
            prob_df = pd.DataFrame({
                "Sentiment": le.classes_,
                "Probability %": [round(float(p)*100, 1) for p in proba]
            }).sort_values("Probability %", ascending=False)

            fig_prob = px.bar(
                prob_df, x="Sentiment", y="Probability %",
                color="Sentiment",
                color_discrete_map={"Positive":"#4CAF50","Negative":"#f44336","Neutral":"#2196F3"},
                text="Probability %",
                title="How confident is the model for each class?"
            )
            fig_prob.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_prob.update_layout(showlegend=False, plot_bgcolor="white", height=300)
            st.plotly_chart(fig_prob, use_container_width=True)

    # ── Try example reviews ───────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Try These Example Reviews</p>', unsafe_allow_html=True)

    examples = [
        ("Positive example", "This moisturiser is absolutely amazing! My skin has never felt so soft and hydrated. I noticed a difference after just one week. Highly recommend to everyone!"),
        ("Negative example", "Worst product I have ever bought. Broke me out badly and caused severe redness. Complete waste of money. Do not buy!"),
        ("Neutral example",  "It is okay I guess. Does what it says but nothing special. The texture is fine. Not sure if I would repurchase."),
    ]

    for title, review in examples:
        with st.expander(f"📝 {title}"):
            st.markdown(f"*{review}*")
            st.code(f'predict_sentiment("{review[:50]}...")')


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style='text-align:center; color:#bbb; font-size:12px; padding:2rem 0 1rem;'>
    BICS 2303 — Intelligent Systems | Group Project &nbsp;|&nbsp;
    Sephora Skincare Review Sentiment Analysis &nbsp;|&nbsp;
    Dataset: Kaggle (nadyinky) — 20,000 reviews
</div>
""", unsafe_allow_html=True)
