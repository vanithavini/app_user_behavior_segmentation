#  📊 App User Behavior Segmentation Using Unsupervised Machine Learning

## 📌 Project Overview

This project focuses on **segmenting app users based on their behavior** using unsupervised machine learning techniques. By analyzing user activity patterns, the project groups users into meaningful segments that can help businesses improve engagement, reduce churn, and design targeted strategies.


## 🧩 Problem Statement

Applications generate large volumes of user interaction data. Understanding user behavior manually is difficult. This project aims to:

* Analyze app user behavior
* Identify hidden user segments
* Convert technical clusters into business-meaningful user groups


## 🛠️ Tech Stack & Tools

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **IDE:** Jupyter Notebook / VS Code


## 📂 Project Structure
```
app_user_behavior_segmentation/
│
├── app_venv(Environment for this )
│
├── business_use_files/
|   ├── customer_level_profiles.csv
│   ├── high_engagement_users.csv
│   ├── moderate_engagement_users.csv
│   ├── low_engagement_users.csv
│   └── occasional_users.csv
│
├── data/
│   ├── raw/
│   │   └── raw_data.csv
│   └── cleaned/
│       ├── cleaned_data.csv
│       └── scaled_for_clustering.csv
├── source/
│       ├── data_understanding.py
│       └── data_cleaning_and_F_engineering
│
├── eda.ipynb
│
├── main.ipynb
├── mains.ipynb
│
├── README.md
├── requirements.txt
├── standard_scaler.pkl
```

## 🔍 Module-wise Description
### ✅ Module 1: Data Understanding* Loaded raw dataset
* Checked shape, columns, data types
* Identified missing values and duplicates
* Analyzed categorical feature distributions

### ✅ Module 2: Data Cleaning & Feature Engineering
* Handled missing values using median imputation
* Removed identifier columns (`user_id`)
* Encoded categorical variables using One-Hot Encoding
* Scaled features using StandardScaler
* Saved cleaned and scaled datasets

### ✅ Module 3: Exploratory Data Analysis (EDA)
* Analyzed engagement score distribution
* Studied session activity vs churn risk
* Checked outliers using boxplots
* Analyzed feature correlations using heatmap

### ✅ Module 4: Clustering & User Segmentation
* Applied **K-Means clustering**
* Used **Elbow Method** to determine optimal clusters
* Evaluated clusters using **Silhouette Score**
* Compared with Hierarchical and BIRCH clustering
* Visualized clusters using **PCA**
* Performed cluster profiling
* Mapped clusters to business-friendly user segments


## 👥 User Segments Identified

* **High Engagement Users** – Very active, low churn risk
* **Moderate Engagement Users** – Regular users with growth potential
* **Low Engagement Users** – Inactive users with high churn risk
* **Occasional Users** – Rarely active users

Each segment is exported as a separate CSV file for business use.

Finally created a customer level profile and exported as a separate CSV file for business use.


### ✅ Module 5: Improving Clustering & PCA Visualization
* Done Feature selection and Outlier removal
* Applied **MiniBatchKMeans clustering**
* Used **Silhouette score** to determine optimal clusters
* Evaluated clusters using best k value with thier **Silhouette Score**
* Compared with Hierarchical and BIRCH clustering
* Visualized improved clusters using **PCA**


## 📈 Business Value
* Enables targeted marketing campaigns
* Helps improve user retention
* Supports personalized user experiences
* Optimizes customer engagement strategies


## 🚀 How to Run the Project
1. Clone the repository
2. Install dependencies:
    pip install -r requirements.txt

3. Run modules in order:

   * Data Understanding
   * Data Cleaning & Feature Engineering
   * EDA
   * Clustering(main.ipynb & mains.ipynb)


## 📌 Limitations
* Based on historical data
* User behavior may change over time
* Static clustering (not real-time)


## 🔮 Future Scope
* Real-time user segmentation
* Time-based behavior analysis
* Use DBSCAN for noise handling
* Integrate deep learning-based clustering

