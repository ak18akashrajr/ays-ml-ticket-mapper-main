import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_roberta_probabilities(df):
    """Run inference with Model 1 (Severity) to get probability vectors."""
    print("Loading Model 1 (Severity) artifacts...")
    model_path = "models/severity_model"
    tok_path = "models/severity_tokenizer"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Severity model not found. Please run train_severity.py first.")
    
    tokenizer = RobertaTokenizer.from_pretrained(tok_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    def tokenize_function(examples):
        return tokenizer(examples['description'], padding="max_length", truncation=True, max_length=256)
    
    dataset = Dataset.from_pandas(df[['description']])
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)
    
    probs_list = []
    print("Running severity model inference to generate features...")
    with torch.no_grad():
        for batch in dataloader:
            inputs = {'input_ids': batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device)}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs_list.append(probs.cpu().numpy())
            
    probs_arr = np.concatenate(probs_list, axis=0)
    
    # RoBERTa output labels: 0=Critical, 1=High, 2=Medium, 3=Low
    df['severity_prob_Critical'] = probs_arr[:, 0]
    df['severity_prob_High'] = probs_arr[:, 1]
    df['severity_prob_Medium'] = probs_arr[:, 2]
    df['severity_prob_Low'] = probs_arr[:, 3]
    return df

def build_feature_matrix(df):
    feature_cols = [
        'severity_prob_Critical',
        'severity_prob_High',
        'severity_prob_Medium',
        'severity_prob_Low',
        'hour_of_day',
        'day_of_week',
        'is_business_hour',
        'is_weekend',
        'escalation_count',
        'reopen_count',
        'token_count',
        'user_ticket_frequency',
        'user_avg_severity_encoded'
    ]
    return df[feature_cols].copy(), feature_cols

def train_priority_model():
    # Load Data
    train_df = pd.read_pickle("data/features_train.pkl")
    val_df = pd.read_pickle("data/features_val.pkl")
    
    # Get Severity probabilities
    train_df = get_roberta_probabilities(train_df)
    val_df = get_roberta_probabilities(val_df)
    
    # Priority Label Encoding: P1=0, P2=1, P3=2, P4=3, P5=4
    pri_map = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
    y_train = train_df['priority'].map(pri_map).values
    y_val = val_df['priority'].map(pri_map).values
    
    X_train, feature_cols = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )
    
    # Evaluation
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='weighted')
    kappa = cohen_kappa_score(y_val, preds)
    
    # Precision and recall for P1 (label 0)
    p1_prec = precision_score(y_val, preds, average=None, labels=[0])[0]
    p1_rec = recall_score(y_val, preds, average=None, labels=[0])[0]
    
    print("\n--- Model Evaluation ---")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"P1 Precision: {p1_prec:.4f}")
    print(f"P1 Recall: {p1_rec:.4f}")
    
    # Save artifacts
    print("\nSaving Priority Model (Pickle) and SHAP Explainer...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/priority_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Standard SHAP Explainer
    explainer = shap.TreeExplainer(model)
    with open("models/priority_shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
        
    with open("models/priority_feature_names.json", "w") as f:
        json.dump(feature_cols, f)
        
    # Save label map
    label_map_inv = {v: k for k, v in pri_map.items()}
    with open("models/priority_label_map.json", "w") as f:
        json.dump(label_map_inv, f)
        
    print("Successfully saved Model 2 artifacts.")

if __name__ == "__main__":
    train_priority_model()
