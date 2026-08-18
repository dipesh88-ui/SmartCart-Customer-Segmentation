import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
/* =========================================================
   SMARTCART VISIBILITY / DARK UI
   ========================================================= */

.stApp {
    background: #0f1117;
    color: #f3f4f6;
}

/* Main content */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #20222c;
}

[data-testid="stSidebar"] * {
    color: #f3f4f6;
}

/* Hero */
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
    border: 1px solid #d9e1ee;
    margin-bottom: 1.2rem;
}

.hero h1 {
    margin-bottom: 0.25rem;
    font-size: 2.2rem;
    color: #111827 !important;
    font-weight: 800;
}

.hero p {
    color: #374151 !important;
    margin: 0;
    font-size: 1rem;
}

/* Section headings */
.section-title {
    color: #f9fafb !important;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #f9fafb !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #d9dee8 !important;
    padding: 14px !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
}

/* Metric label */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
    color: #4b5563 !important;
    font-weight: 600 !important;
}

/* Metric number */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #111827 !important;
    font-weight: 800 !important;
}

/* Metric delta */
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #374151 !important;
}

/* Normal Streamlit text */
.stMarkdown,
.stText,
p,
li,
label {
    color: #e5e7eb;
}

/* Text inside white containers/cards */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* Selectbox / inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
}

div[data-baseweb="select"] span {
    color: #111827 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}

/* Info boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Make horizontal rules visible */
hr {
    border-color: #374151;
}

/* Sidebar radio text */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #f3f4f6 !important;
}

/* Captions */
.stCaption {
    color: #9ca3af !important;
}
</style>
""", unsafe_allow_html=True)

# Plotly charts use a light template so chart text/axes remain readable
px.defaults.template = "plotly_white"



# =========================================================
# DATA + MODEL PIPELINE
# =========================================================
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


@st.cache_data
def prepare_data(df):
    data = df.copy()

    # Same missing-value treatment as notebook
    if "Income" in data.columns:
        data["Income"] = data["Income"].fillna(data["Income"].median())

    # Feature engineering
    if "Year_Birth" in data.columns:
        data["Age"] = 2026 - data["Year_Birth"]

    if "Dt_Customer" in data.columns:
        data["Dt_Customer"] = pd.to_datetime(
            data["Dt_Customer"],
            dayfirst=True,
            errors="coerce"
        )

        reference_date = data["Dt_Customer"].max()
        data["Customer_Tenure_Days"] = (
            reference_date - data["Dt_Customer"]
        ).dt.days

    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    available_spending = [c for c in spending_cols if c in data.columns]
    data["Total_Spending"] = data[available_spending].sum(axis=1)

    if "Kidhome" in data.columns and "Teenhome" in data.columns:
        data["Total_Children"] = data["Kidhome"] + data["Teenhome"]

    # Education transformation
    if "Education" in data.columns:
        data["Education"] = data["Education"].replace({
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate"
        })

    # Marital status transformation
    if "Marital_Status" in data.columns:
        data["Living_With"] = data["Marital_Status"].replace({
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone"
        })

    # Same columns dropped in notebook
    cols_to_drop = [
        "ID",
        "Year_Birth",
        "Marital_Status",
        "Kidhome",
        "Teenhome",
        "Dt_Customer"
    ] + spending_cols

    cols_to_drop = [c for c in cols_to_drop if c in data.columns]
    cleaned = data.drop(columns=cols_to_drop, errors="ignore").copy()

    # Same outlier rules as notebook
    if "Age" in cleaned.columns:
        cleaned = cleaned[cleaned["Age"] < 90]

    if "Income" in cleaned.columns:
        cleaned = cleaned[cleaned["Income"] < 600_000]

    cleaned = cleaned.reset_index(drop=True)

    # One-hot encoding
    cat_cols = [c for c in ["Education", "Living_With"] if c in cleaned.columns]

    if cat_cols:
        try:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        encoded = ohe.fit_transform(cleaned[cat_cols])
        enc_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(cat_cols),
            index=cleaned.index
        )

        encoded_df = pd.concat(
            [cleaned.drop(columns=cat_cols), enc_df],
            axis=1
        )
    else:
        encoded_df = cleaned.copy()

    # Make sure all model columns are numeric
    encoded_df = encoded_df.apply(pd.to_numeric, errors="coerce")
    encoded_df = encoded_df.replace([np.inf, -np.inf], np.nan)
    encoded_df = encoded_df.fillna(encoded_df.median(numeric_only=True))
    encoded_df = encoded_df.fillna(0)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded_df)

    # PCA: same as notebook
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # K-Means: same final k=4 as notebook
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_pca)

    # Agglomerative: same as notebook
    agg = AgglomerativeClustering(
        n_clusters=4,
        linkage="ward"
    )
    labels_agg = agg.fit_predict(X_pca)

    model_df = encoded_df.copy()
    model_df["cluster"] = labels_agg

    # Add cluster labels to cleaned data
    result = cleaned.copy()
    result["cluster"] = labels_agg
    result["PCA1"] = X_pca[:, 0]
    result["PCA2"] = X_pca[:, 1]
    result["PCA3"] = X_pca[:, 2]

    return {
        "raw": data,
        "cleaned": cleaned,
        "encoded": encoded_df,
        "scaled": X_scaled,
        "pca": X_pca,
        "pca_model": pca,
        "kmeans": kmeans,
        "kmeans_labels": labels_kmeans,
        "agg_labels": labels_agg,
        "result": result,
        "model_df": model_df,
        "scaler": scaler
    }


# =========================================================
# LOAD DATA
# =========================================================
DEFAULT_FILE = "smartcart_customers.csv"

st.sidebar.title("🛒 SmartCart")
st.sidebar.caption("Customer Segmentation Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload SmartCart CSV",
    type=["csv"],
    help="Leave empty to use smartcart_customers.csv"
)

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = load_data(DEFAULT_FILE)
except FileNotFoundError:
    st.error(
        "smartcart_customers.csv was not found. "
        "Put the CSV in the same folder as app.py or upload it from the sidebar."
    )
    st.stop()

try:
    data = prepare_data(raw_df)
except Exception as e:
    st.error(f"Could not process the dataset: {e}")
    st.stop()


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📊 Data Overview",
        "🧹 Preprocessing",
        "🧩 PCA Analysis",
        "🎯 Clustering",
        "👥 Customer Segments"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Pipeline based on the provided notebook:\n\n"
    "Feature Engineering → Outlier Removal → One-Hot Encoding → "
    "StandardScaler → PCA (3D) → K-Means / Agglomerative Clustering"
)


# =========================================================
# COMMON VALUES
# =========================================================
result = data["result"]
pca_data = data["pca"]
pca_model = data["pca_model"]

n_customers = len(result)
n_clusters = result["cluster"].nunique()

avg_income = result["Income"].mean() if "Income" in result.columns else 0
avg_spending = (
    result["Total_Spending"].mean()
    if "Total_Spending" in result.columns
    else 0
)

silhouette = silhouette_score(
    pca_data,
    data["agg_labels"]
)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🛒 SmartCart Customer Segmentation</h1>
    <p>
        Interactive customer analytics dashboard based on the
        SmartCart clustering notebook.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown('<div class="section-title">Business Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{n_customers:,}")
    c2.metric("Clusters", n_clusters)
    c3.metric("Avg Income", f"{avg_income:,.0f}")
    c4.metric("Avg Spending", f"{avg_spending:,.0f}")

    st.write("")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Distribution")

        cluster_counts = (
            result["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Customers")
        )

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Customers",
            text="Customers",
            title="Customers per Cluster"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Income vs Spending")

        if {"Income", "Total_Spending"}.issubset(result.columns):
            fig = px.scatter(
                result,
                x="Total_Spending",
                y="Income",
                color=result["cluster"].astype(str),
                hover_data=[
                    c for c in ["ID", "Age", "Education", "Living_With"]
                    if c in result.columns
                ],
                labels={"color": "Cluster"},
                title="Customer Income & Spending"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")

    numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
    summary_cols = [
        c for c in [
            "Income",
            "Recency",
            "Response",
            "Age",
            "Total_Spending",
            "Total_Children",
            "Customer_Tenure_Days"
        ]
        if c in numeric_cols
    ]

    if summary_cols:
        summary = (
            result.groupby("cluster")[summary_cols]
            .mean()
            .round(2)
        )
        st.dataframe(summary, use_container_width=True)


# =========================================================
# DATA OVERVIEW
# =========================================================
elif page == "📊 Data Overview":

    st.subheader("Raw Dataset")

    a, b, c = st.columns(3)
    a.metric("Rows", raw_df.shape[0])
    b.metric("Columns", raw_df.shape[1])
    c.metric(
        "Missing Values",
        int(raw_df.isnull().sum().sum())
    )

    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": raw_df.columns,
        "Data Type": raw_df.dtypes.astype(str).values,
        "Missing": raw_df.isnull().sum().values,
        "Unique Values": [
            raw_df[col].nunique(dropna=True)
            for col in raw_df.columns
        ]
    })

    st.dataframe(info_df, use_container_width=True)

    st.subheader("Numeric Correlation")

    corr = data["cleaned"].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PREPROCESSING
# =========================================================
elif page == "🧹 Preprocessing":

    st.subheader("Feature Engineering & Cleaning")

    st.markdown("""
    The dashboard follows the same major preprocessing steps used
    in the notebook:

    1. Fill missing `Income` values using the median.
    2. Create `Age`.
    3. Create `Customer_Tenure_Days`.
    4. Create `Total_Spending`.
    5. Create `Total_Children`.
    6. Simplify `Education`.
    7. Create `Living_With` from marital status.
    8. Drop unnecessary columns.
    9. Remove age and income outliers.
    10. One-hot encode categorical variables.
    11. Standardize numerical features.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Raw Rows", len(raw_df))
    c2.metric("Rows After Cleaning", len(data["cleaned"]))
    c3.metric("Removed Rows", len(raw_df) - len(data["cleaned"]))

    st.subheader("Cleaned Data")
    st.dataframe(
        data["cleaned"].head(20),
        use_container_width=True
    )

    st.subheader("Encoded Data")
    st.dataframe(
        data["encoded"].head(10),
        use_container_width=True
    )

    st.subheader("Selected Feature Distributions")

    feature_options = [
        c for c in [
            "Income",
            "Recency",
            "Age",
            "Total_Spending",
            "Total_Children",
            "Customer_Tenure_Days"
        ]
        if c in data["cleaned"].columns
    ]

    selected_feature = st.selectbox(
        "Select feature",
        feature_options
    )

    fig = px.histogram(
        data["cleaned"],
        x=selected_feature,
        nbins=30,
        title=f"Distribution of {selected_feature}"
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PCA
# =========================================================
elif page == "🧩 PCA Analysis":

    st.subheader("Principal Component Analysis")

    explained = pca_model.explained_variance_ratio_

    c1, c2, c3 = st.columns(3)
    c1.metric("PCA1 Variance", f"{explained[0] * 100:.2f}%")
    c2.metric("PCA2 Variance", f"{explained[1] * 100:.2f}%")
    c3.metric("PCA3 Variance", f"{explained[2] * 100:.2f}%")

    st.metric(
        "Total Variance Explained",
        f"{explained.sum() * 100:.2f}%"
    )

    pca_df = pd.DataFrame({
        "PCA1": pca_data[:, 0],
        "PCA2": pca_data[:, 1],
        "PCA3": pca_data[:, 2],
        "Cluster": data["agg_labels"].astype(str)
    })

    st.subheader("3D PCA Projection")

    fig = px.scatter_3d(
        pca_df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Cluster",
        title="3D PCA Customer Projection",
        opacity=0.75
    )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=45)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Explained Variance")

    variance_df = pd.DataFrame({
        "Component": ["PCA1", "PCA2", "PCA3"],
        "Explained Variance": explained,
        "Percentage": explained * 100
    })

    fig = px.bar(
        variance_df,
        x="Component",
        y="Percentage",
        text="Percentage",
        title="Variance Explained by PCA Components"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# CLUSTERING
# =========================================================
elif page == "🎯 Clustering":

    st.subheader("Clustering Analysis")

    st.info(
        "The notebook evaluates K-Means for different K values and "
        "then uses K=4 for the final K-Means and Agglomerative models."
    )

    k_values = list(range(2, 11))
    wcss = []
    sil_scores = []

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = km.fit_predict(pca_data)

        wcss.append(km.inertia_)
        sil_scores.append(
            silhouette_score(pca_data, labels)
        )

    metric_df = pd.DataFrame({
        "K": k_values,
        "WCSS": wcss,
        "Silhouette": sil_scores
    })

    left, right = st.columns(2)

    with left:
        fig = px.line(
            metric_df,
            x="K",
            y="WCSS",
            markers=True,
            title="Elbow / WCSS"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            metric_df,
            x="K",
            y="Silhouette",
            markers=True,
            title="Silhouette Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final K=4 Comparison")

    c1, c2, c3 = st.columns(3)
    c1.metric("K-Means Clusters", 4)
    c2.metric("Agglomerative Clusters", 4)
    c3.metric(
        "Agglomerative Silhouette",
        f"{silhouette:.3f}"
    )

    comparison_df = pd.DataFrame({
        "PCA1": pca_data[:, 0],
        "PCA2": pca_data[:, 1],
        "PCA3": pca_data[:, 2],
        "KMeans": data["kmeans_labels"],
        "Agglomerative": data["agg_labels"]
    })

    algorithm = st.radio(
        "Visualize algorithm",
        ["K-Means", "Agglomerative"],
        horizontal=True
    )

    label_col = (
        "KMeans"
        if algorithm == "K-Means"
        else "Agglomerative"
    )

    fig = px.scatter_3d(
        comparison_df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color=comparison_df[label_col].astype(str),
        title=f"3D {algorithm} Clusters",
        labels={label_col: "Cluster"}
    )

    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================
elif page == "👥 Customer Segments":

    st.subheader("Explore Customer Segments")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(result["cluster"].unique())
    )

    cluster_data = result[
        result["cluster"] == selected_cluster
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("Customers", len(cluster_data))

    if "Income" in cluster_data.columns:
        c2.metric(
            "Average Income",
            f"{cluster_data['Income'].mean():,.0f}"
        )

    if "Total_Spending" in cluster_data.columns:
        c3.metric(
            "Average Spending",
            f"{cluster_data['Total_Spending'].mean():,.0f}"
        )

    st.subheader(f"Cluster {selected_cluster} Profile")

    profile_cols = [
        c for c in [
            "Income",
            "Recency",
            "Response",
            "Age",
            "Total_Spending",
            "Total_Children",
            "Customer_Tenure_Days"
        ]
        if c in cluster_data.columns
    ]

    if profile_cols:
        profile = (
            cluster_data[profile_cols]
            .mean()
            .round(2)
            .to_frame("Average")
        )
        st.dataframe(profile, use_container_width=True)

    if {"Income", "Total_Spending"}.issubset(cluster_data.columns):
        fig = px.scatter(
            cluster_data,
            x="Total_Spending",
            y="Income",
            hover_data=[
                c for c in ["Age", "Education", "Living_With"]
                if c in cluster_data.columns
            ],
            title=f"Cluster {selected_cluster}: Income vs Spending"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customers in Selected Cluster")
    st.dataframe(
        cluster_data.drop(
            columns=["PCA1", "PCA2", "PCA3"],
            errors="ignore"
        ),
        use_container_width=True
    )

    csv = cluster_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Selected Cluster",
        data=csv,
        file_name=f"smartcart_cluster_{selected_cluster}.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "SmartCart Customer Segmentation • "
    "Based on the provided Jupyter Notebook • "
    "PCA + K-Means + Agglomerative Clustering"
)