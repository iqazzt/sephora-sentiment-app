# ============================================================
# pipeline.py
# BICS 2303 – Intelligent Systems | Group Project
# Sephora Skincare Review Sentiment Analysis
#
# This file contains ALL the original processing logic.
# No algorithm has been changed. No logic has been modified.
# Only change: CSV path updated from Google Drive to local.
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns


# ── Download NLTK data ────────────────────────────────────
def download_nltk():
    """Download all required NLTK resources."""
    nltk.download('stopwords',    quiet=True)
    nltk.download('punkt',        quiet=True)
    nltk.download('wordnet',      quiet=True)
    nltk.download('punkt_tab',    quiet=True)
    nltk.download('vader_lexicon',quiet=True)


# ============================================================
# PHASE 2 — NLP PREPROCESSING
# Original clean_text() function — unchanged.
# ============================================================

# Initialise stopwords and lemmatizer at module level
download_nltk()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess a single review text.
    Steps: lowercase → remove URLs → remove special characters
           → tokenise → remove stopwords → lemmatise
    Unchanged from original code.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenise
    tokens = word_tokenize(text)
    # Remove stopwords + Lemmatise
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)


# ============================================================
# PHASE 3 — VADER SENTIMENT LABELLING
# Original functions — unchanged.
# ============================================================

vader = SentimentIntensityAnalyzer()


def get_compound_score(text):
    """
    Return VADER compound score for a review.
    Applied to original review_text (not cleaned).
    Unchanged from original code.
    """
    if not isinstance(text, str):
        return 0
    return vader.polarity_scores(text)['compound']


def assign_sentiment_label(score):
    """
    Assign sentiment label based on VADER compound score.
    Thresholds: >= 0.05 → Positive, <= -0.05 → Negative, else Neutral.
    Unchanged from original code.
    """
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# ============================================================
# PHASE 4 — NAIVE BAYES PREDICTION
# Original predict_new_review() function — unchanged.
# Depends on tfidf_vectorizer and nb_model from run_pipeline().
# ============================================================

def make_predict_function(tfidf_vectorizer, nb_model):
    """
    Returns the original predict_new_review() function
    with the trained vectorizer and model injected.
    Logic is 100% identical to original.
    """
    def predict_new_review(custom_review):
        """
        Takes raw review text, cleans it, transforms with TF-IDF,
        predicts sentiment using Naive Bayes.
        Unchanged from original code.
        """
        # Step 1: Clean the input text
        cleaned_input = clean_text(custom_review)
        # Step 2: Transform using TF-IDF
        tfidf_input = tfidf_vectorizer.transform([cleaned_input])
        # Step 3: Predict using Naive Bayes
        prediction = nb_model.predict(tfidf_input)
        return prediction[0]

    return predict_new_review


# ============================================================
# MAIN PIPELINE FUNCTION
# Wraps all original phases into one callable function.
# CSV path is the only modification: local path instead of Drive.
# ============================================================

def run_pipeline(csv_path="reviews_sample.csv"):
    """
    Runs the complete sentiment analysis pipeline.
    Returns all objects needed by the Streamlit app.

    Modification from original:
        - CSV loaded from local path (was Google Drive path)
        - Everything else is 100% identical to original code
    """

    # ── PHASE 1: Data Loading ─────────────────────────────
    # ONLY CHANGE: path updated from Google Drive to local
    df = pd.read_csv(csv_path, encoding='latin1')

    # Select relevant columns — unchanged
    df = df[['review_text', 'product_name', 'brand_name']]

    # ── PHASE 2: NLP Preprocessing ────────────────────────
    df["clean_review"] = df["review_text"].apply(clean_text)

    # Remove empty rows after preprocessing — unchanged
    df = df[df["clean_review"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    # ── PHASE 3: VADER Sentiment Labelling ────────────────
    # Apply to original review_text — unchanged
    df['compound_score']   = df['review_text'].apply(get_compound_score)
    df['sentiment_label']  = df['compound_score'].apply(assign_sentiment_label)

    # ── PHASE 4: TF-IDF + Naive Bayes ────────────────────
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95
    )
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_review'])
    y       = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        test_size=0.2,
        random_state=42
    )

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    y_pred   = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    # Build predict function — same logic as original
    predict_new_review = make_predict_function(tfidf_vectorizer, nb_model)

    # ── PHASE 5A: Dimension Reduction ─────────────────────
    # For clustering — unchanged
    svd_cluster = TruncatedSVD(n_components=50, random_state=42)
    X_cluster   = svd_cluster.fit_transform(X_tfidf)

    scaler    = MinMaxScaler()
    X_cluster = scaler.fit_transform(X_cluster)

    # For 2D visualisation — unchanged
    svd_visual = TruncatedSVD(n_components=2, random_state=42)
    X_visual   = svd_visual.fit_transform(X_tfidf)

    # ── PHASE 5B: K-Means Clustering ──────────────────────
    best_k     = 2
    best_score = -1

    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        score  = silhouette_score(X_cluster, labels)
        if score > best_score:
            best_score = score
            best_k     = k

    kmeans_final         = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans_final.fit_predict(X_cluster)
    kmeans_silhouette    = silhouette_score(X_cluster, df['kmeans_cluster'])

    # ── PHASE 5C: Mini-Batch K-Means ──────────────────────
    mb_best_k     = 2
    mb_best_score = -1

    for k in range(2, 7):
        mb_kmeans  = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=256)
        mb_labels  = mb_kmeans.fit_predict(X_cluster)
        mb_score   = silhouette_score(X_cluster, mb_labels)
        if mb_score > mb_best_score:
            mb_best_score = mb_score
            mb_best_k     = k

    mb_kmeans_final              = MiniBatchKMeans(n_clusters=mb_best_k, random_state=42, n_init=10, batch_size=256)
    df['minibatch_kmeans_cluster'] = mb_kmeans_final.fit_predict(X_cluster)
    mb_kmeans_silhouette         = silhouette_score(X_cluster, df['minibatch_kmeans_cluster'])

    # ── PHASE 5D: DBSCAN Clustering ───────────────────────
    dbscan               = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(X_cluster)

    n_clusters   = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'].values else 0)
    noise_points = list(df['dbscan_cluster']).count(-1)

    if n_clusters > 1:
        dbscan_silhouette = silhouette_score(X_cluster, df['dbscan_cluster'])
    else:
        dbscan_silhouette = -1

    # Comparison table — unchanged
    comparison_table = pd.DataFrame({
        'Metric': [
            'Silhouette Score',
            'Number of Clusters',
            'Detects Outliers',
            'Needs K Value'
        ],
        'K-Means': [
            round(kmeans_silhouette, 4),
            best_k,
            'No',
            'Yes'
        ],
        'DBSCAN': [
            round(dbscan_silhouette, 4),
            n_clusters,
            'Yes',
            'No'
        ]
    })

    # ── PHASE 6: Brand & Product Analysis ─────────────────
    # Brand analysis — unchanged
    brand_counts          = df['brand_name'].value_counts()
    popular_brands        = brand_counts[brand_counts >= 10].index
    df_filtered_brands    = df[df['brand_name'].isin(popular_brands)]
    brand_sentiment       = df_filtered_brands.groupby(['brand_name', 'sentiment_label']).size().unstack(fill_value=0)
    brand_sentiment['Total']               = brand_sentiment.sum(axis=1)
    brand_sentiment['Positive_Percentage'] = (brand_sentiment.get('Positive', 0) / brand_sentiment['Total']) * 100
    brand_sentiment['Negative_Percentage'] = (brand_sentiment.get('Negative', 0) / brand_sentiment['Total']) * 100
    most_positive_brands  = brand_sentiment.sort_values(by='Positive_Percentage', ascending=False)
    most_negative_brands  = brand_sentiment.sort_values(by='Negative_Percentage', ascending=False)

    # Product analysis — unchanged
    product_counts        = df['product_name'].value_counts()
    popular_products      = product_counts[product_counts >= 10].index
    df_filtered_products  = df[df['product_name'].isin(popular_products)]
    product_sentiment     = df_filtered_products.groupby(['product_name', 'sentiment_label']).size().unstack(fill_value=0)
    product_sentiment['Total']               = product_sentiment.sum(axis=1)
    product_sentiment['Positive_Percentage'] = (product_sentiment.get('Positive', 0) / product_sentiment['Total']) * 100
    product_sentiment['Negative_Percentage'] = (product_sentiment.get('Negative', 0) / product_sentiment['Total']) * 100
    most_positive_products = product_sentiment.sort_values(by='Positive_Percentage', ascending=False)
    most_negative_products = product_sentiment.sort_values(by='Negative_Percentage', ascending=False)

    # ── Return all objects ────────────────────────────────
    return {
        # Data
        "df"                    : df,

        # Phase 4 — NB results
        "accuracy"              : accuracy,
        "clf_report_text"       : clf_report_text,
        "clf_report"            : clf_report,
        "cm"                    : cm,
        "y_test"                : y_test,
        "y_pred"                : y_pred,
        "predict_new_review"    : predict_new_review,
        "tfidf_vectorizer"      : tfidf_vectorizer,
        "nb_model"              : nb_model,

        # Phase 5 — Clustering
        "X_visual"              : X_visual,
        "X_cluster"             : X_cluster,
        "best_k"                : best_k,
        "kmeans_silhouette"     : round(kmeans_silhouette, 4),
        "mb_best_k"             : mb_best_k,
        "mb_kmeans_silhouette"  : round(mb_kmeans_silhouette, 4),
        "n_clusters"            : n_clusters,
        "noise_points"          : noise_points,
        "dbscan_silhouette"     : round(dbscan_silhouette, 4),
        "comparison_table"      : comparison_table,

        # Phase 6 — Brand / Product
        "most_positive_brands"  : most_positive_brands,
        "most_negative_brands"  : most_negative_brands,
        "most_positive_products": most_positive_products,
        "most_negative_products": most_negative_products,
        "brand_sentiment"       : brand_sentiment,
        "product_sentiment"     : product_sentiment,
    }
