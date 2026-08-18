🛒 SmartCart Customer Segmentation

An interactive Customer Segmentation Dashboard built with Python, Streamlit, Pandas, Scikit-learn, and Plotly.

SmartCart analyzes customer characteristics and purchasing behavior to identify meaningful customer groups using PCA, K-Means, and Agglomerative Clustering. The project also provides an interactive Streamlit dashboard for exploring preprocessing, dimensionality reduction, clustering results, and individual customer segments.

🚀 Live Demo

🔗 Live Application: https://smartcartcustomers.streamlit.app/

You can open the deployed application directly in your browser and explore the customer segmentation dashboard.

📌 Project Overview

Customer segmentation is the process of dividing customers into groups based on similar characteristics and behavior.

The SmartCart project applies an unsupervised machine learning workflow to discover these groups without using predefined target labels.

The overall workflow is:

Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Outlier Removal
       ↓
One-Hot Encoding
       ↓
StandardScaler
       ↓
PCA
       ↓
K-Means / Agglomerative Clustering
       ↓
Customer Segments
       ↓
Interactive Streamlit Dashboard

✨ Features

🛒 Interactive SmartCart dashboard

📊 Customer data overview

🧹 Data preprocessing and cleaning

🔧 Feature engineering

🔤 Categorical feature encoding

📏 Feature scaling using StandardScaler

🧩 PCA-based dimensionality reduction

🎯 K-Means clustering

🌳 Agglomerative clustering

📈 Elbow / WCSS analysis

📐 Silhouette Score analysis

📊 Interactive Plotly visualizations

🌐 3D PCA customer visualization

👥 Individual customer segment analysis

💰 Income vs. spending analysis

⬇️ Download selected customer cluster as CSV

📁 CSV upload support

🧠 Machine Learning Workflow

1. Data Cleaning

The dataset is inspected and prepared before applying machine learning.

Missing values are handled and unnecessary or unsuitable records are removed according to the preprocessing workflow.

2. Feature Engineering

Additional customer-related features are created, including:

Age

Customer Tenure

Total Spending

Total Children

Living With

These features help represent customer behavior more effectively.

3. Outlier Removal

Extreme values are removed from selected features so that unusual observations do not dominate the clustering process.

4. One-Hot Encoding

Categorical variables such as education and living arrangement are converted into numerical features using One-Hot Encoding.

5. Feature Scaling

StandardScaler is used to standardize the features before dimensionality reduction and clustering.

6. PCA

Principal Component Analysis (PCA) is used to reduce the dimensionality of the processed dataset.

The dashboard uses three principal components for visualization and clustering.

7. Clustering

Two unsupervised clustering techniques are used:

K-Means Clustering

Agglomerative Hierarchical Clustering

The final clustering workflow uses 4 clusters.

📊 Dashboard Sections

🏠 Dashboard

The main dashboard provides a quick overview of:

Total customers

Number of clusters

Average income

Average spending

Customer distribution

Income vs. spending

Cluster summary

📊 Data Overview

Provides information about:

Raw dataset

Number of rows and columns

Missing values

Data types

Unique values

Correlation between numerical features

🧹 Preprocessing

Displays the processed dataset and important preprocessing operations such as:

Missing-value treatment

Feature engineering

Outlier removal

Categorical encoding

Numerical transformation

🧩 PCA Analysis

Provides:

PCA explained variance

PCA1, PCA2, and PCA3

Interactive 3D PCA visualization

Cluster distribution in reduced-dimensional space

🎯 Clustering

Provides:

K-Means evaluation for different values of K

WCSS / Elbow curve

Silhouette Score

K-Means visualization

Agglomerative clustering visualization

👥 Customer Segments

Users can select a cluster and explore:

Number of customers

Average income

Average spending

Segment profile

Income vs. spending

Customers belonging to the selected segment

Downloadable CSV of the selected cluster

🛠️ Technologies Used

Technology

Purpose

Python

Core programming language

Pandas

Data manipulation

NumPy

Numerical computation

Scikit-learn

Machine learning and preprocessing

Plotly

Interactive visualizations

Streamlit

Web application/dashboard

Jupyter Notebook

Data analysis and experimentation

📂 Project Structure

SmartCart-Customer-Segmentation/
│
├── app.py
├── smartcart_customers.csv
├── smartcart_clustering.ipynb
├── requirements.txt
└── README.md

⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/SmartCart-Customer-Segmentation.git

Move into the project directory:

cd SmartCart-Customer-Segmentation

Install the dependencies:

pip install -r requirements.txt

▶️ Run Locally

Make sure the dataset is available in the project directory.

Then run:

streamlit run app.py

The application will open in your browser.

📦 Requirements

The main dependencies are:

streamlit
pandas
numpy
scikit-learn
plotly

🎯 Project Objective

The primary objective of SmartCart Customer Segmentation is to discover meaningful customer groups based on their demographic characteristics and purchasing behavior.

These customer segments can support data-driven business decisions such as:

🎯 Targeted marketing

🎁 Personalized offers

🛍️ Customer recommendations

📢 Customer engagement

📈 Business strategy

💡 Customer behavior analysis

📈 Why Customer Segmentation?

Different customers have different purchasing patterns.

For example, one group may represent customers with:

High income

High spending

Frequent purchases

Another group may represent:

Lower spending

Lower engagement

Different purchasing behavior

Identifying these groups helps businesses understand who their customers are and how they behave.

🔬 Algorithms Used

K-Means

K-Means divides customers into K groups by assigning each customer to the nearest cluster centroid.

The project evaluates different K values using:

WCSS / Elbow Method

Silhouette Score

The final workflow uses K = 4.

Agglomerative Clustering

Agglomerative clustering starts with individual observations and progressively merges similar observations into larger groups.

The final workflow uses:

n_clusters = 4

Ward linkage

📊 Evaluation

The clustering process can be evaluated using:

WCSS

Within-Cluster Sum of Squares helps analyze how compact the clusters are and is commonly used with the Elbow Method.

Silhouette Score

The Silhouette Score measures how well-separated and internally cohesive the generated clusters are.

A higher score generally indicates better-defined clusters.

🌐 Live Demo

Try the deployed application:

https://smartcartcustomers.streamlit.app/

👨‍💻 Author

Dipesh Kumar

B.Tech CSE | Data Science & Machine Learning Enthusiast

⭐ Support

If you find this project useful, consider giving the repository a ⭐ star.

Feel free to explore the dashboard, experiment with the clustering analysis, and build upon the project.
