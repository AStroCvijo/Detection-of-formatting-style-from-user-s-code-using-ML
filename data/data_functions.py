import os
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------------------------------------------------
# Dataset functions
# --------------------------------------------------------------------------------------------------------------------------------------------

# Function to extract Coq (.v) files from an input folder to an output folder
def filter_coq_files(input_folder, output_folder):
    # Define Coq file extension
    coq_extensions = ['.v']

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through input folder to find and copy Coq files
    for root, _, files in os.walk(input_folder):
        for file in files:
            if any(file.endswith(ext) for ext in coq_extensions):
                # Define input and output file paths
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, file)
                # Copy the file to the output directory
                shutil.copy(input_file_path, output_file_path)
                print(f"Copied: {input_file_path} -> {output_file_path}")

# --------------------------------------------------------------------------------------------------------------------------------------------
# Data preprocessing functions
# --------------------------------------------------------------------------------------------------------------------------------------------

# Define sets of keywords and operators for token classification
KEYWORDS = {
    'Definition', 'Lemma', 'Proof', 'Qed', 'move', 'exact', 'rewrite',
    'let', 'in', 'fun', 'match', 'with', 'if', 'then', 'else', 'forall',
    'exists', 'induction', 'constructor', 'apply', 'assert', 'clear',
    'unfold', 'intros', 'split', 'simpl', 'rewrite', 'exact', 'apply',
}

OPERATORS = {'=', '=>', '<-', ':', ';', '.', '(', ')', '{', '}', '[', ']'}

# Function to determine token type based on its value
def get_token_type(token):
    if token in KEYWORDS:
        return 'keyword'
    elif token in OPERATORS:
        return 'operator'
    elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
        return 'identifier'
    else:
        return 'other'

# Function to tokenize Coq files, identifying tokens and their types
def tokenize_coq_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()

    # Regex pattern to split into tokens, spaces, and special characters
    token_pattern = r'(\s+|[a-zA-Z_][a-zA-Z0-9_]*|[^\s\w])'
    tokens = re.findall(token_pattern, code)

    # Classify tokens as text or spacing and add spacing tokens as necessary
    tokenized_with_spacing = []
    token_info = []
    for token in tokens:
        if token.isspace():
            # Differentiate newline from other whitespace
            if '\n' in token:
                tokenized_with_spacing.append('<NEWLINE>')
                token_info.append('spacing')
            else:
                tokenized_with_spacing.append('<SPACE>')
                token_info.append('spacing')
        else:
            token_type = get_token_type(token)
            tokenized_with_spacing.append(token)
            token_info.append(token_type)

    return tokenized_with_spacing, token_info

# Function to tokenize all Coq files in a directory and collect tokens and types
def tokenize_coq_files_in_directory(directory):
    all_tokens = []
    all_token_info = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.v'):
            file_path = os.path.join(directory, filename)
            tokens, token_info = tokenize_coq_file(file_path)
            all_tokens.extend(tokens)
            all_token_info.extend(token_info)

    return all_tokens, all_token_info

# Function to create token sequences and labels for training
def create_sequences_and_labels(tokens, token_info, seq_length):
    sequences = []
    labels = []
    
    for i in range(len(tokens) - seq_length):
        # Extract a sequence of tokens and determine the label for the next token
        sequence = tokens[i:i + seq_length]
        next_token = tokens[i + seq_length]
        label = next_token if next_token in ("<SPACE>", "<NEWLINE>") else "other"
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels

# Mapping of token labels to indices
label_to_index = {'<SPACE>': 0, '<NEWLINE>': 1, 'other': 2}

# Dataset class for Coq token sequences
class CoqTokenDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        # Convert labels to indices
        self.labels = [label_to_index[label] for label in labels] 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Function to split data into training, validation, and test sets
def split_data(sequences, labels, seed, test_size=0.2, val_size=0.1):
    # Initial split into train+validation and test sets
    train_val_seqs, test_seqs, train_val_labels, test_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=seed
    )
    # Further split train+validation into train and validation sets
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        train_val_seqs, train_val_labels, test_size=val_size / (1 - test_size), random_state=seed
    )
    return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels

# Function to create a vocabulary mapping for token indices
def build_vocab(tokens):
    unique_tokens = list(set(tokens))
    token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
    return token_to_index

# Function to convert token sequences to index sequences using the vocabulary
def tokens_to_indices(sequences, token_to_index):
    return [[token_to_index[token] for token in seq] for seq in sequences]
