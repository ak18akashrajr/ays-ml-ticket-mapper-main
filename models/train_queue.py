import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

def evaluate_top_k(y_true, y_prob, k):
    """Evaluate Top-K accuracy given true labels and predicted probabilities."""
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)

def build_features(df):
    """Extract and concatenate BERT embeddings and structured features."""
    structured_cols = [
        'hour_of_day',
        'is_business_hour',
        'escalation_count',
        'reassign_count',
        'token_count',
        'has_error_code',
        'user_ticket_frequency'
    ]
    
    # Extract BERT embeddings
    # Assuming bert_embedding column contains lists or arrays of length 768
    X_text = np.stack(df['bert_embedding'].values)
    
    # Extract structured features
    X_struct = df[structured_cols].values
    
    # Concatenate features
    X_combined = np.concatenate([X_text, X_struct], axis=1)
    return X_combined

def train_queue_model():
    print("Loading Data...")
    train_df = pd.read_pickle("data/features_train.pkl")
    val_df = pd.read_pickle("data/features_val.pkl")
    
    X_train = build_features(train_df)
    X_val = build_features(val_df)
    
    print(f"X_train shape: {X_train.shape}")
    
    # Queue labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['assigned_to'])
    y_val = le.transform(val_df['assigned_to'])
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("\n--- Model Evaluation ---")
    
    y_prob = model.predict_proba(X_val)
    
    top1_acc = evaluate_top_k(y_val, y_prob, k=1)
    top3_acc = evaluate_top_k(y_val, y_prob, k=3)
    
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    print("\nSaving Queue Router Model, Embedder, and Label Encoder...")
    os.makedirs("models", exist_ok=True)
    
    # Save the Random Forest
    with open("models/queue_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Save LabelEncoder
    with open("models/queue_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    # The implementation plan also requests downloading the sentence-transformer models to "models/queue_embedder/"
    print("Saving SentenceTransformer model locally...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    os.makedirs("models/queue_embedder", exist_ok=True)
    embedder.save("models/queue_embedder")
    
    # Save label classes purely for reference
    queue_map = {i: c for i, c in enumerate(le.classes_)}
    with open("models/queue_label_map.json", "w") as f:
        json.dump(queue_map, f)
        
    print("Successfully saved Model 3 artifacts.")

if __name__ == "__main__":
    train_queue_model()
