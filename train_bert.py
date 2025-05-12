import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhishingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

try:
    # Load balanced dataset
    df = pd.read_csv('data/intermediate/balanced_processed_dataset.csv')
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Balanced dataset missing 'text' or 'label' columns")
    logger.info(f"Loaded dataset: {df.shape[0]} rows")

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    logger.info(f"Train: {train_df.shape[0]}, Val: {val_df.shape[0]}, Test: {test_df.shape[0]}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = PhishingDataset(train_encodings, train_df['label'].tolist())
    val_dataset = PhishingDataset(val_encodings, val_df['label'].tolist())
    test_dataset = PhishingDataset(test_encodings, test_df['label'].tolist())

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    logger.info("Training completed")

    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = test_df['label'].values
    logger.info("Test Set Evaluation:\n" + classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))

    model.save_pretrained('models/bert_phishing')
    tokenizer.save_pretrained('models/bert_phishing')
    logger.info("Model and tokenizer saved to models/bert_phishing")
except Exception as e:
    logger.error(f"Error in training: {e}")