import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.impute import SimpleImputer

# Load the raw Dataset
file_path = (r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\app_user_behavior_segmentation\data\raw\raw_data.csv")
df = pd.read_csv(file_path)

# Quick check
df.head()
df.shape

# Data Cleaning
################

# Missing Value Handling
df['rating_given'].fillna(df['rating_given'].median(), inplace=True)

# Remove Identifier Column
df_model = df.drop(columns=['user_id'])


# Feature Engineering
######################

# Encoding Categorical Features
categorical_cols = [
    'gender', 'country', 'device_type',
    'app_version', 'subscription_type',
    'marketing_source'
]

df_model = pd.get_dummies(
    df_model,
    columns=categorical_cols,
    drop_first=True
)


# Feature Scaling
##################
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_model)
scaled_df = pd.DataFrame(scaled_data, columns=df_model.columns)

# Final Check
print("After encoding df_model shape:", df_model.shape)
print("After scaling scaled_df shape:", scaled_df.shape)

print("\nEncoded data preview:")
print(df_model.head())

print("\nScaled data preview:")
print(scaled_df.head())


# Saved Cleaned + encoded dataset (MAIN ASSET)
df_model.to_csv(
    r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\app_user_behavior_segmentation\data\cleaned\cleaned_data.csv",
    index=False
)

# Scaled dataset (OPTIONAL, for speed)
scaled_df.to_csv(
    r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\app_user_behavior_segmentation\data\cleaned\scaled_for_clustering.csv",
    index=False
)

# Save scaler 
import joblib
joblib.dump(scaler, "standard_scaler.pkl")

