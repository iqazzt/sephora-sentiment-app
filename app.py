# ============================================================
# app.py
# BICS 2303 – Intelligent Systems | Group Project
# Sephora Skincare Review Sentiment Analysis
# Streamlit Dashboard
#
# HOW TO RUN:
#   1. pip install streamlit pandas numpy scikit-learn
#      matplotlib seaborn nltk vaderSentiment
#   2. Place reviews_sample.csv in the same folder
#   3. streamlit run app.py
#
# This file ONLY handles display.
# All logic lives in pipeline.py — unchanged.
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sephora Sentiment Dashboard",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Basic styling ─────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border-top: 4px solid #ff0050;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .metric-label {
        font-size: 13px;
        color: #888;
        margin: 4px 0 0;
    }
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a2e;
        padding-bottom: 6px;
        border-bottom: 2px solid #ff0050;
        margin-bottom: 1rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD PIPELINE — cached so it runs only once per session
# ============================================================

@st.cache_resource(show_spinner="Running pipeline... this may take a few minutes ⏳")
def load_pipeline():
    """
    Calls run_pipeline() from pipeline.py.
    Cached — runs only once, results reused across all pages.
    """
    from pipeline import run_pipeline
    return run_pipeline(csv_path="reviews_sample.csv")


# Check CSV exists before running
if not os.path.exists("reviews_sample.csv"):
    st.error("❌ reviews_sample.csv not found. Please place it in the same folder as app.py")
    st.stop()

# Load everything from pipeline
results = load_pipeline()

# Unpack all objects from pipeline
df                    = results["df"]
accuracy              = results["accuracy"]
clf_report_text       = results["clf_report_text"]
clf_report            = results["clf_report"]
cm                    = results["cm"]
y_test                = results["y_test"]
y_pred                = results["y_pred"]
predict_new_review    = results["predict_new_review"]
X_visual              = results["X_visual"]
best_k                = results["best_k"]
kmeans_silhouette     = results["kmeans_silhouette"]
mb_best_k             = results["mb_best_k"]
mb_kmeans_silhouette  = results["mb_kmeans_silhouette"]
n_clusters            = results["n_clusters"]
noise_points          = results["noise_points"]
dbscan_silhouette     = results["dbscan_silhouette"]
comparison_table      = results["comparison_table"]
most_positive_brands  = results["most_positive_brands"]
most_negative_brands  = results["most_negative_brands"]
most_positive_products= results["most_positive_products"]
most_negative_products= results["most_negative_products"]
brand_sentiment       = results["brand_sentiment"]
product_sentiment     = results["product_sentiment"]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 💄 Sephora Sentiment")
    st.markdown("---")
    page = st.radio(
        "Go to",
        [
            "🏠  Home",
            "📊  Sentiment Dashboard",
            "🏷️  Brand & Product Analysis",
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
    st.markdown(f"**Total reviews:** {len(df):,}")
    st.markdown(f"**Best K (KMeans):** {best_k}")
    st.markdown(f"**Best K (MiniBatch):** {mb_best_k}")


# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "🏠  Home":

    st.markdown("# 💄 Sephora Skincare Review Sentiment Dashboard")
    st.markdown("**Organisation Analytics View** — Sentiment determined from review text only. Rating column not used.")
    st.divider()

    # ── Project overview ──────────────────────────────────
    st.markdown("### 📋 Project Overview")
    st.markdown("""
    This dashboard analyses Sephora skincare product reviews to automatically classify
    each review as **Positive**, **Negative**, or **Neutral** using the review text only.

    **Phase 1 — Sentiment Analysis**
    - VADER automatically labels each review from the text
    - Multinomial Naive Bayes is trained to predict sentiment

    **Phase 2 — Clustering**
    - K-Means, Mini-Batch K-Means, and DBSCAN group similar reviews
    - Silhouette Score selects the best configuration

    **Phase 3 — Business Insight**
    - Identifies which brands and products attract the most positive and negative feedback
    """)
    st.divider()

    # ── Dataset summary ───────────────────────────────────
    st.markdown('<p class="section-title">Dataset Summary</p>', unsafe_allow_html=True)

    n_pos = len(df[df['sentiment_label'] == 'Positive'])
    n_neg = len(df[df['sentiment_label'] == 'Negative'])
    n_neu = len(df[df['sentiment_label'] == 'Neutral'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <p class="metric-value">{len(df):,}</p>
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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sentiment distribution bar chart ──────────────────
    st.markdown('<p class="section-title">Sentiment Distribution</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = ['#4CAF50', '#f44336', '#2196F3']
    labels  = ['Positive', 'Negative', 'Neutral']
    counts  = [n_pos, n_neg, n_neu]
    bars    = ax.bar(labels, counts, color=colors, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment Label Distribution")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── Dataset info ──────────────────────────────────────
    st.markdown('<p class="section-title">Dataset Info</p>', unsafe_allow_html=True)
    info = pd.DataFrame({
        "Item"  : ["Source", "File", "Columns used", "Labelling method", "Rating used?"],
        "Detail": [
            "Kaggle — nadyinky/sephora-products-and-skincare-reviews",
            "reviews_sample.csv",
            "review_text, product_name, brand_name",
            "VADER (text-based, automatic)",
            "No — text only"
        ]
    })
    st.dataframe(info, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2 — SENTIMENT DASHBOARD
# ============================================================

elif page == "📊  Sentiment Dashboard":

    st.markdown("# 📊 Sentiment Analysis Dashboard")
    st.markdown("Naive Bayes model trained on VADER-labelled reviews.")
    st.divider()

    # ── Accuracy metric ───────────────────────────────────
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <p class="metric-value">{round(accuracy * 100, 2)}%</p>
            <p class="metric-label">Accuracy Score</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        macro_f1 = round(clf_report.get('macro avg', {}).get('f1-score', 0) * 100, 2)
        st.markdown(f"""<div class="metric-box" style="border-top-color:#FF9800">
            <p class="metric-value" style="color:#FF9800">{macro_f1}%</p>
            <p class="metric-label">Macro F1-Score</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        n_test = len(y_test)
        st.markdown(f"""<div class="metric-box" style="border-top-color:#9C27B0">
            <p class="metric-value" style="color:#9C27B0">{n_test:,}</p>
            <p class="metric-label">Test Set Size (20%)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pie + Bar charts ──────────────────────────────────
    st.markdown('<p class="section-title">Sentiment Distribution Charts</p>', unsafe_allow_html=True)

    n_pos = len(df[df['sentiment_label'] == 'Positive'])
    n_neg = len(df[df['sentiment_label'] == 'Negative'])
    n_neu = len(df[df['sentiment_label'] == 'Neutral'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pie Chart**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [n_pos, n_neg, n_neu],
            labels=['Positive', 'Negative', 'Neutral'],
            colors=['#4CAF50', '#f44336', '#2196F3'],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Bar Chart**")
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ['Positive', 'Negative', 'Neutral'],
            [n_pos, n_neg, n_neu],
            color=['#4CAF50', '#f44336', '#2196F3'],
            width=0.5
        )
        for bar, count in zip(bars, [n_pos, n_neg, n_neu]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 30,
                    f'{count:,}', ha='center', fontsize=10)
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Counts")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── Classification report ─────────────────────────────
    st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
    st.code(clf_report_text)

    # ── Confusion matrix ──────────────────────────────────
    st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
    labels = sorted(df['sentiment_label'].unique())
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close()

    # ── Sample reviews ────────────────────────────────────
    st.markdown('<p class="section-title">Sample Reviews by Sentiment</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["😊 Positive", "😞 Negative", "😐 Neutral"])

    with tab1:
        samples = df[df['sentiment_label'] == 'Positive'][['review_text', 'compound_score']].head(5)
        st.dataframe(samples, use_container_width=True)
    with tab2:
        samples = df[df['sentiment_label'] == 'Negative'][['review_text', 'compound_score']].head(5)
        st.dataframe(samples, use_container_width=True)
    with tab3:
        samples = df[df['sentiment_label'] == 'Neutral'][['review_text', 'compound_score']].head(5)
        st.dataframe(samples, use_container_width=True)


# ============================================================
# PAGE 3 — BRAND & PRODUCT ANALYSIS
# ============================================================

elif page == "🏷️  Brand & Product Analysis":

    st.markdown("# 🏷️ Brand & Product Analysis")
    st.markdown("Based purely on review text sentiment. Rating column not used.")
    st.divider()

    # ── Most positive brand ───────────────────────────────
    st.markdown('<p class="section-title">Most Positive Brand</p>', unsafe_allow_html=True)
    st.dataframe(
        most_positive_brands[['Positive', 'Total', 'Positive_Percentage']].head(1),
        use_container_width=True
    )

    # ── Most negative brand ───────────────────────────────
    st.markdown('<p class="section-title">Most Negative Brand</p>', unsafe_allow_html=True)
    st.dataframe(
        most_negative_brands[['Negative', 'Total', 'Negative_Percentage']].head(1),
        use_container_width=True
    )

    # ── Top 5 positive brands chart ───────────────────────
    st.markdown('<p class="section-title">Top 5 Brands — Most Positive Reviews</p>', unsafe_allow_html=True)
    top5_pos = most_positive_brands.head(5).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        top5_pos['brand_name'],
        top5_pos['Positive_Percentage'],
        color='#4CAF50'
    )
    for bar, val in zip(bars, top5_pos['Positive_Percentage']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_xlabel("Positive Review %")
    ax.set_title("Top 5 Most Positive Brands")
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── Top 5 negative brands chart ───────────────────────
    st.markdown('<p class="section-title">Top 5 Brands — Most Negative Reviews</p>', unsafe_allow_html=True)
    top5_neg = most_negative_brands.head(5).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        top5_neg['brand_name'],
        top5_neg['Negative_Percentage'],
        color='#f44336'
    )
    for bar, val in zip(bars, top5_neg['Negative_Percentage']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_xlabel("Negative Review %")
    ax.set_title("Top 5 Most Negative Brands")
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── Most positive product ─────────────────────────────
    st.markdown('<p class="section-title">Most Positive Product</p>', unsafe_allow_html=True)
    st.dataframe(
        most_positive_products[['Positive', 'Total', 'Positive_Percentage']].head(1),
        use_container_width=True
    )

    # ── Most negative product ─────────────────────────────
    st.markdown('<p class="section-title">Most Negative Product</p>', unsafe_allow_html=True)
    st.dataframe(
        most_negative_products[['Negative', 'Total', 'Negative_Percentage']].head(1),
        use_container_width=True
    )

    # ── Full brand sentiment table ─────────────────────────
    st.markdown('<p class="section-title">Full Brand Sentiment Table</p>', unsafe_allow_html=True)
    st.dataframe(
        brand_sentiment.sort_values('Positive_Percentage', ascending=False),
        use_container_width=True
    )

    # ── Full product sentiment table ───────────────────────
    st.markdown('<p class="section-title">Full Product Sentiment Table</p>', unsafe_allow_html=True)
    st.dataframe(
        product_sentiment.sort_values('Positive_Percentage', ascending=False),
        use_container_width=True
    )


# ============================================================
# PAGE 4 — CLUSTERING ANALYSIS
# ============================================================

elif page == "🔵  Clustering Analysis":

    st.markdown("# 🔵 Clustering Analysis")
    st.markdown("K-Means, Mini-Batch K-Means, and DBSCAN — all on the same page.")
    st.divider()

    # ── Metrics row ───────────────────────────────────────
    st.markdown('<p class="section-title">Clustering Summary Metrics</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <p class="metric-value">{best_k}</p>
            <p class="metric-label">KMeans Best K</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#4CAF50">
            <p class="metric-value">{kmeans_silhouette}</p>
            <p class="metric-label">KMeans Silhouette</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#FF9800">
            <p class="metric-value">{mb_best_k}</p>
            <p class="metric-label">MiniBatch Best K</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#9C27B0">
            <p class="metric-value">{mb_kmeans_silhouette}</p>
            <p class="metric-label">MiniBatch Silhouette</p>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-box" style="border-top-color:#2196F3">
            <p class="metric-value">{dbscan_silhouette}</p>
            <p class="metric-label">DBSCAN Silhouette</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── K-Means visualisation — original seaborn code ─────
    st.markdown('<p class="section-title">K-Means Clustering Visualisation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot — original seaborn code preserved
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x=X_visual[:, 0],
            y=X_visual[:, 1],
            hue=df['kmeans_cluster'],
            palette='viridis',
            s=40,
            ax=ax
        )
        ax.set_title("K-Means Clustering")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        st.pyplot(fig)
        plt.close()

    with col2:
        # Cluster analysis — original logic preserved
        st.markdown("**Cluster statistics:**")
        cluster_counts = df['kmeans_cluster'].value_counts()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(df)) * 100
            cluster_data       = df[df['kmeans_cluster'] == cluster]
            dominant_sentiment = cluster_data['sentiment_label'].mode()[0]
            st.markdown(f"""
            **Cluster {cluster}**
            - Reviews: {count:,} ({percentage:.2f}%)
            - Dominant sentiment: {dominant_sentiment}
            """)

    # ── Mini-Batch K-Means visualisation ──────────────────
    st.markdown('<p class="section-title">Mini-Batch K-Means Clustering Visualisation</p>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x=X_visual[:, 0],
            y=X_visual[:, 1],
            hue=df['minibatch_kmeans_cluster'],
            palette='plasma',
            s=40,
            ax=ax
        )
        ax.set_title("Mini-Batch K-Means Clustering")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**Cluster statistics:**")
        mb_counts = df['minibatch_kmeans_cluster'].value_counts()
        for cluster, count in mb_counts.items():
            percentage = (count / len(df)) * 100
            cluster_data       = df[df['minibatch_kmeans_cluster'] == cluster]
            dominant_sentiment = cluster_data['sentiment_label'].mode()[0]
            st.markdown(f"""
            **Cluster {cluster}**
            - Reviews: {count:,} ({percentage:.2f}%)
            - Dominant sentiment: {dominant_sentiment}
            """)

    # ── DBSCAN visualisation — original seaborn code ──────
    st.markdown('<p class="section-title">DBSCAN Clustering Visualisation</p>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        # Scatter plot — original seaborn code preserved
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x=X_visual[:, 0],
            y=X_visual[:, 1],
            hue=df['dbscan_cluster'],
            palette='plasma',
            s=40,
            ax=ax
        )
        ax.set_title("DBSCAN Clustering")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        st.pyplot(fig)
        plt.close()

    with col6:
        # DBSCAN cluster analysis — original logic preserved
        st.markdown("**Cluster statistics:**")
        dbscan_counts = df['dbscan_cluster'].value_counts()
        for cluster, count in dbscan_counts.items():
            percentage = (count / len(df)) * 100
            if cluster == -1:
                st.markdown(f"""
                **Noise Points**
                - Reviews: {count:,} ({percentage:.2f}%)
                - These reviews do not belong to any cluster
                """)
            else:
                st.markdown(f"""
                **Cluster {cluster}**
                - Reviews: {count:,} ({percentage:.2f}%)
                """)

    # ── Comparison table — original logic preserved ────────
    st.markdown('<p class="section-title">K-Means vs DBSCAN Comparison Table</p>', unsafe_allow_html=True)
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    # Best model — original logic preserved
    if kmeans_silhouette > dbscan_silhouette:
        st.success("✅ Better clustering model: **K-Means** — Reason: Higher silhouette score")
    else:
        st.success("✅ Better clustering model: **DBSCAN** — Reason: Higher silhouette score")


# ============================================================
# PAGE 5 — REAL-TIME PREDICTION
# ============================================================

elif page == "🔮  Real-Time Prediction":

    st.markdown("# 🔮 Real-Time Sentiment Prediction")
    st.markdown("Enter any review — the trained Naive Bayes model predicts its sentiment.")
    st.divider()

    # ── Input ─────────────────────────────────────────────
    st.markdown('<p class="section-title">Enter a Review</p>', unsafe_allow_html=True)

    custom_review = st.text_area(
        "Type or paste a review here:",
        placeholder="e.g. This moisturiser is absolutely amazing! My skin feels so soft...",
        height=130,
        label_visibility="collapsed"
    )

    predict_btn = st.button("🔮 Predict Sentiment", type="primary")

    # ── Prediction — calls original predict_new_review() ──
    if predict_btn:
        if custom_review.strip() == "":
            st.warning("⚠️ Please enter a review first!")
        else:
            # Calls the exact original predict_new_review() from pipeline
            result = predict_new_review(custom_review)

            # Display result
            st.markdown("---")
            st.markdown("### Prediction Result")

            if result == "Positive":
                st.success(f"😊 **Predicted Sentiment: {result}**")
            elif result == "Negative":
                st.error(f"😞 **Predicted Sentiment: {result}**")
            else:
                st.info(f"😐 **Predicted Sentiment: {result}**")

            # Also show VADER score for reference
            from pipeline import get_compound_score, assign_sentiment_label
            compound = get_compound_score(custom_review)
            vader_label = assign_sentiment_label(compound)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Naive Bayes Prediction", result)
            with col2:
                st.metric("VADER Compound Score", compound)

    st.divider()

    # ── Demo tests — original demo cases ──────────────────
    st.markdown('<p class="section-title">Demo Testing — Original Cases</p>', unsafe_allow_html=True)
    st.markdown("These are the exact 3 demo cases from the original code.")

    demo_reviews = [
        ("Test 1 — Positive", "This moisturizer is amazing!"),
        ("Test 2 — Negative", "Do not buy"),
        ("Test 3 — Neutral",  "The product is okay."),
    ]

    for title, review in demo_reviews:
        result = predict_new_review(review)
        emoji  = "😊" if result == "Positive" else "😞" if result == "Negative" else "😐"
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{title}**")
            st.markdown(f"*{review}*")
        with col2:
            st.markdown(f"**Predicted:** {emoji} {result}")
        with col3:
            st.markdown("")
        st.markdown("---")


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#bbb; font-size:12px; padding:2rem 0 1rem;'>
    BICS 2303 — Intelligent Systems | Group Project |
    Sephora Skincare Review Sentiment Analysis |
    Dataset: Kaggle (nadyinky)
</div>
""", unsafe_allow_html=True)
