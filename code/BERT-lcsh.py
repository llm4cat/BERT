import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the data
data = pd.read_csv('./lcsh_m.csv', encoding='utf-8-sig')

# Combine title, toc, and abstract as model input
data['text'] = data['title'].fillna('') + ' ' + data['toc'].fillna('') + ' ' + data['abstract'].fillna('')

# Step 2: Extract binarized LCSH label columns
lcsh_labels = data.filter(regex='^lcsh_').values  # Extract all columns starting with "lcsh_"

# Step 3: Split the dataset
train_texts, temp_texts, train_labels_lcsh, temp_labels_lcsh = train_test_split(
    data['text'].tolist(),
    lcsh_labels,
    test_size=0.3,  # First split 30% as temporary data
    random_state=42
)

val_texts, test_texts, val_labels_lcsh, test_labels_lcsh = train_test_split(
    temp_texts, 
    temp_labels_lcsh, 
    test_size=0.5,  # Split half of the remaining 30% into validation and test sets
    random_state=42
)

# Step 4: Define custom dataset class
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
            'labels': torch.tensor(label_lcsh, dtype=torch.float)  # For multi-label problems, use float type
        }

# Step 5: Initialize BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 6: Create dataset objects
train_dataset = TextDataset(train_texts, train_labels_lcsh, tokenizer, max_len=128)
val_dataset = TextDataset(val_texts, val_labels_lcsh, tokenizer, max_len=128)
test_dataset = TextDataset(test_texts, test_labels_lcsh, tokenizer, max_len=128)

# Step 7: Initialize BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=train_labels_lcsh.shape[1]  # Set the number of output labels according to the number of binarized labels
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Unpack logits and labels
    predictions = torch.sigmoid(torch.tensor(logits))  # Convert to sigmoid to get probabilities
    predictions = (predictions > 0.5).int()  # Binarize predictions, 0.5 as threshold

    # Convert logits and labels to numpy array format
    predictions = predictions.cpu().numpy()
    labels = labels  # If labels are already numpy arrays, no need to convert

    # Calculate evaluation metrics, adding zero_division=0 to avoid warnings
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

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",  # Save after each epoch
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,  # Output logs every 100 steps
    load_best_model_at_end=True  # Load the best model at the end
)

# Step 9: Use Trainer API to train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Step 10: Train the model
trainer.train()

# Step 11: Model evaluation (Validation set)
eval_results = trainer.evaluate()
print("Validation Evaluation Results:", eval_results)

# Step 12: Use the trained model to evaluate the test set
test_results = trainer.predict(test_dataset)
print("Test Evaluation Results:", test_results.metrics)

# Save the model
model.save_pretrained('./saved_model2/')
