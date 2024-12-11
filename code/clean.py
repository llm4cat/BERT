import pandas as pd
import re

# Load data
file_path = './bibli.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Print data size before cleaning
print(f"Data size before cleaning: {data.shape}")

# Columns to clean
columns_to_clean = ['toc', 'title', 'abstract', 'lcsh_subject_headings', 'fast_subject_headings']

# Check if the columns to clean exist
for col in columns_to_clean:
    if col not in data.columns:
        print(f"Warning: Column {col} not found in the dataset.")

# Define the cleaning function: trim whitespace, convert to lowercase, remove unnecessary punctuation
def clean_text(text):
    if pd.isna(text):
        return text
    text = text.strip()  # Remove leading and trailing spaces
    text = re.sub(r'\.', '', text)  # Remove periods
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.lower()  # Convert to lowercase

# Clean the selected columns
for col in columns_to_clean:
    if col in data.columns:
        data[col] = data[col].apply(clean_text)

# Drop rows with missing values, limited to the specified columns
data.dropna(subset=columns_to_clean, inplace=True)

# Remove duplicates based on the specified columns
data.drop_duplicates(subset=columns_to_clean, inplace=True)

# Check data size to ensure the impact of cleaning operations
print(f"Data size after cleaning: {data.shape}")

# Display the first few rows of the cleaned data
print(data.head())

# Save the cleaned data to a new CSV file, ensuring 'utf-8-sig' encoding to avoid encoding issues
data.to_csv('./cleaned_data.csv', index=False, encoding='utf-8-sig')
