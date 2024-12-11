import numpy as np
import pandas as pd
from transformers import BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset

# Step 1: Load the data
data = pd.read_csv('./fast_m.csv', encoding='utf-8-sig')

# Combine title, toc, and abstract as input for the model
data['text'] = data['title'].fillna('') + ' ' + data['toc'].fillna('') + ' ' + data['abstract'].fillna('')

# Step 2: Extract binarized FAST label columns
fast_labels = data.filter(regex='^fast_').values  # Extract all columns starting with "fast_"

# Step 3: Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels_fast, tokenizer, max_len):
        self.texts = texts
        self.labels_fast = labels_fast
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label_fast = self.labels_fast[index]
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
            'labels': torch.tensor(label_fast, dtype=torch.float)  # Use float type
        }

# Step 4: Initialize BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 5: Split the validation dataset
test_texts = data['text'].tolist()
test_labels_fast = fast_labels

# Step 6: Create validation dataset object
test_dataset = TextDataset(test_texts, test_labels_fast, tokenizer, max_len=128)

# Step 7: Load the trained model
model = BertForSequenceClassification.from_pretrained(
    './saved_model1',  
    num_labels=test_labels_fast.shape[1],  # Use shape[1] to get the number of labels for multi-label classification
    ignore_mismatched_sizes=True
)

# Step 8: Define training parameters
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# Step 9: Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Unpack logits and labels
    predictions = torch.sigmoid(torch.tensor(logits))  # Apply sigmoid to get probabilities
    predictions = (predictions > 0.5).int()  # Binarize probabilities with a threshold of 0.5

    # Convert logits and labels to numpy arrays
    predictions = predictions.cpu().numpy()
    labels = labels  # No conversion needed as labels are already in numpy format

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='samples')
    recall = recall_score(labels, predictions, average='samples')
    f1 = f1_score(labels, predictions, average='samples')

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Step 10: Use Trainer API to perform evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Pass compute_metrics to Trainer
)

# Step 11: Evaluate model performance and print detailed metrics
test_results = trainer.predict(test_dataset)
print("Test Evaluation Results:", test_results.metrics)
metrics = compute_metrics((test_results.predictions, test_results.label_ids))
print(metrics)
