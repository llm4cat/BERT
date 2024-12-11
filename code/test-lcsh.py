import numpy as np
import pandas as pd
from transformers import BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset

# Step 1: Load data
data = pd.read_csv('./lcsh_m.csv', encoding='utf-8-sig')

# Combine title, toc, and abstract as model input
data['text'] = data['title'].fillna('') + ' ' + data['toc'].fillna('') + ' ' + data['abstract'].fillna('')

# Step 2: Extract binarized LCSH label columns
lcsh_labels = data.filter(regex='^lcsh_').values  # Extract all columns starting with "lcsh_"

# Step 3: Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels_lcsh, tokenizer, max_len):
        self.texts = texts
        self.labels_lcsh = labels_lcsh
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label_lcsh = self.labels_lcsh[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_lcsh, dtype=torch.float)  # Use float type for labels
        }

# Step 4: Initialize BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 5: Prepare test dataset
test_texts = data['text'].tolist()
test_labels_lcsh = lcsh_labels

# Step 6: Create test dataset object
test_dataset = TextDataset(test_texts, test_labels_lcsh, tokenizer, max_len=128)

# Step 7: Load the pre-trained model
model = BertForSequenceClassification.from_pretrained(
    './saved_model2',
    num_labels=test_labels_lcsh.shape[1],  # Use shape[1] to get the number of labels
    ignore_mismatched_sizes=True
)

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# Step 9: Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Unpack logits and labels
    predictions = torch.sigmoid(torch.tensor(logits))  # Convert logits to probabilities using sigmoid
    predictions = (predictions > 0.5).int()  # Binarize predictions with a threshold of 0.5

    # Convert logits and labels to numpy array format
    predictions = predictions.cpu().numpy()
    labels = labels  # If labels are already numpy arrays, no need to convert

    # Calculate evaluation metrics, with zero_division=0 to avoid warnings
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='samples', zero_division=0)
    recall = recall_score(labels, predictions, average='samples')
    f1 = f1_score(labels, predictions, average='samples')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Step 10: Use Trainer API to run the evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Pass compute_metrics to Trainer
)

# Step 11: Evaluate the model and print detailed metrics
test_results = trainer.predict(test_dataset)
print("Test Evaluation Results:", test_results.metrics)
