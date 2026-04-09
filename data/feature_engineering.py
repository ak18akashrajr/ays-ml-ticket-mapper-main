import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

def execute_feature_engineering():
    print("Loading synthetic tickets...")
    # Assume script is run from project root
    data_path = "data/tickets_synthetic.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Ensure you run from project root.")
    
    df = pd.read_csv(data_path)
    
    print(f"Data shape initially: {df.shape}")
    
    # Text Features
    print("Extracting text features...")
    df['raw_text'] = df['description'].str.lower().apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
    df['token_count'] = df['description'].apply(lambda x: len(str(x).split()))
    
    # Error code patterns (OS356..., ERR_, HTTP 5xx)
    error_pattern = r'(os\d+|err_|http\s*5\d\d|error\s*\d+)'
    df['has_error_code'] = df['description'].str.lower().apply(lambda x: 1 if re.search(error_pattern, str(x)) else 0)
    
    # Temporal Features
    print("Extracting temporal features...")
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour_of_day'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # assumes business hours 08:00 - 18:00
    df['is_business_hour'] = df.apply(lambda row: 1 if (row['hour_of_day'] >= 8 and row['hour_of_day'] <= 18 and row['is_weekend'] == 0) else 0, axis=1)
    
    # User Features
    print("Extracting user features...")
    user_counts = df['affected_user'].value_counts().to_dict()
    df['user_ticket_frequency'] = df['affected_user'].map(user_counts)
    
    severity_map = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    df['severity_encoded'] = df['severity'].map(severity_map)
    user_avg_sev = df.groupby('affected_user')['severity_encoded'].mean().to_dict()
    df['user_avg_severity_encoded'] = df['affected_user'].map(user_avg_sev)
    
    # Operational count features are already integers in the dataset
    
    print("Creating BERT Embeddings... This may take a moment.")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedder.encode(df['description'].tolist(), batch_size=32, show_progress_bar=True)
    df['bert_embedding'] = list(embeddings)
    
    print("Splitting data into Train/Val/Test (70/15/15)...")
    # Stratified split requires target, let's stratify by severity
    df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df['severity'], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['severity'], random_state=42)
    
    print(f"Train split: {df_train.shape[0]} samples")
    print(f"Val split: {df_val.shape[0]} samples")
    print(f"Test split: {df_test.shape[0]} samples")
    
    os.makedirs("data", exist_ok=True)
    df_train.to_pickle("data/features_train.pkl")
    df_val.to_pickle("data/features_val.pkl")
    df_test.to_pickle("data/features_test.pkl")
    
    print("Feature engineering complete. Artifacts saved in data/ directory.")

if __name__ == "__main__":
    execute_feature_engineering()
