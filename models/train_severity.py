import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Calculate recall for Critical class (which is label 0)
    # We will compute per-class recall and grab the 0th index
    recalls = recall_score(labels, predictions, average=None, labels=[0, 1, 2, 3])
    critical_recall = recalls[0] if len(recalls) > 0 else 0.0
    
    return {
        'weighted_f1': f1,
        'critical_recall': critical_recall
    }

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train_severity_model():
    print(f"Training on device: {device}")
    
    # Load Data
    train_df = pd.read_pickle("data/features_train.pkl")
    val_df = pd.read_pickle("data/features_val.pkl")
    
    # Prepare Labels
    label_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    train_df['label'] = train_df['severity'].map(label_map)
    val_df['label'] = val_df['severity'].map(label_map)
    
    # Compute Class Weights
    y_train = train_df['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class weights computed: {weights}")
    
    # Tokenizer
    print("Loading RobertaTokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    def tokenize_function(examples):
        return tokenizer(examples['description'], padding="max_length", truncation=True, max_length=256)
    
    # Create huggingface datasets
    train_dataset = Dataset.from_pandas(train_df[['description', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['description', 'label']])
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Load Model
    print("Loading RobertaForSequenceClassification...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
    model.to(device)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./models/severity_checkpoints',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="epoch",
        metric_for_best_model="weighted_f1",
        load_best_model_at_end=True,
        report_to="none"  # Do not report to wandb
    )
    
    # Initialize Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating on Validation Set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")
    
    # Save Model Artifacts for Local Usage
    print("Saving model and tokenizer to models/ directory...")
    os.makedirs("models/severity_model", exist_ok=True)
    os.makedirs("models/severity_tokenizer", exist_ok=True)
    
    model.save_pretrained("models/severity_model")
    tokenizer.save_pretrained("models/severity_tokenizer")
    
    label_map_inv = {v: k for k, v in label_map.items()}
    with open("models/severity_label_map.json", "w") as f:
        json.dump(label_map_inv, f)
    
    print("Successfully saved Model 1 artifacts.")

if __name__ == "__main__":
    train_severity_model()
