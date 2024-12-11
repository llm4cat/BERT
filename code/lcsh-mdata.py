import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load data
data = pd.read_csv('cleaned_data.csv', encoding='utf-8-sig')

# Extract the needed columns
columns_needed = ['toc', 'title', 'abstract', 'lcsh_subject_headings']
data_subset = data[columns_needed]

# Split the lcsh_subject_headings into a list by semicolon
def split_labels(label):
    if pd.isna(label):
        return []
    return [l.strip().lower() for l in label.split(';') if l.strip()]

data_subset['lcsh_subject_headings'] = data_subset['lcsh_subject_headings'].apply(split_labels)

# Count the frequency of LCSH labels
all_lcsh_labels = [label for sublist in data_subset['lcsh_subject_headings'] for label in sublist]
lcsh_label_freq = pd.Series(all_lcsh_labels).value_counts()

# Get the top 20 most frequent LCSH labels
top_20_lcsh_labels = lcsh_label_freq.head(20).index.tolist()

# Filter the data to retain only rows containing the top 20 LCSH labels
def filter_top_labels(labels, top_labels):
    return any(label in top_labels for label in labels)

# Filter the data to retain only rows containing the top 20 LCSH labels
filtered_data = data_subset[
    data_subset['lcsh_subject_headings'].apply(lambda x: filter_top_labels(x, top_20_lcsh_labels))
]

# Use MultiLabelBinarizer to binarize the LCSH labels
mlb_lcsh = MultiLabelBinarizer(classes=top_20_lcsh_labels)

# Binarized LCSH labels
lcsh_labels_binary = mlb_lcsh.fit_transform(filtered_data['lcsh_subject_headings'])

# Convert the binarized labels into a DataFrame and add label names as column names
lcsh_labels_df = pd.DataFrame(lcsh_labels_binary, columns=[f'lcsh_{label}' for label in mlb_lcsh.classes_])

# Combine toc, title, abstract with the binarized labels
final_data = pd.concat([filtered_data[['toc', 'title', 'abstract']].reset_index(drop=True), lcsh_labels_df], axis=1)

# Print the size of the filtered data
print(f"Size of filtered data: {filtered_data.shape[0]} rows")

# Display the result and save it as a new file
print(final_data.head())
final_data.to_csv('lcsh_m.csv', index=False, encoding='utf-8-sig')
