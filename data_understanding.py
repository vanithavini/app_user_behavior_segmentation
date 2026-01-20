import pandas as pd  
import numpy as np   

# Load the dataset
df = pd.read_csv(r"C:\Users\Vanitha\OneDrive\Desktop\app_user_behavior_dataset.csv")

# Quick Check
print("Dataset Shape:", df.shape) 
print("Columns:", df.columns.tolist())  
print("First 5 rows:\n", df.head())  

# Check data types
print("Data types: ", df.info())

# Summary statistics for numerical columns
print("Summary Statistics for Numeric columns:\n ", df.describe())

# Check missing values
print("Missing values per column:\n", df.isnull().sum())

# Check duplicates
print("Number of duplicate rows:", df.duplicated().sum())

# Categorical Value Distribution Check
categorical_cols = [
    'gender', 'country', 'device_type',
    'app_version', 'subscription_type',
    'marketing_source'
]

for col in categorical_cols:
    print(f"\n{col.upper()} DISTRIBUTION")
    print(df[col].value_counts(normalize=True))

# Save a raw copy of data
df.to_csv(
    r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\app_user_behavior_segmentation\data\raw\raw_data.csv",
    index=False
)

