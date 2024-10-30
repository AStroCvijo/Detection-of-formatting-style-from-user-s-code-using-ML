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

# Function for extracting coq files
def filter_coq_files(input_folder, output_folder):
    # Coq file extension
    cpp_extensions = ['.v']

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Check if the file has a coq extension
            if any(file.endswith(ext) for ext in cpp_extensions):
                # Full path to the input file
                input_file_path = os.path.join(root, file)
                # Full path to the output file
                output_file_path = os.path.join(output_folder, file)
                # Copy the file to the output folder
                shutil.copy(input_file_path, output_file_path)
                print(f"Copied: {input_file_path} -> {output_file_path}")

# --------------------------------------------------------------------------------------------------------------------------------------------
# Data preprocess functions
# --------------------------------------------------------------------------------------------------------------------------------------------

# Define sets of keywords and operators
KEYWORDS = {
    'Definition', 'Lemma', 'Proof', 'Qed', 'move', 'exact', 'rewrite',
    'let', 'in', 'fun', 'match', 'with', 'if', 'then', 'else', 'forall',
    'exists', 'induction', 'constructor', 'apply', 'assert', 'clear',
    'unfold', 'intros', 'split', 'simpl', 'rewrite', 'exact', 'apply',
}

OPERATORS = {
    '=', '=>', '<-', ':', ';', '.', '(', ')', '{', '}', '[', ']', 
}

# Function to get token type
def get_token_type(token):
    if token in KEYWORDS:
        return 'keyword'
    elif token in OPERATORS:
        return 'operator'
    elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
        return 'identifier'
    else:
        return 'other'

# Function to tokenize coq files
def tokenize_coq_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()

    # Regex to match Coq tokens, spaces, and newlines
    token_pattern = r'(\s+|[a-zA-Z_][a-zA-Z0-9_]*|[^\s\w])'

    # Find all the tokens and spacing information
    tokens = re.findall(token_pattern, code)

    # Insert special tokens for spacing and newlines
    tokenized_with_spacing = []
    token_info = []
    for token in tokens:
        if token.isspace():
            if '\n' in token:
                # Add a newline token
                tokenized_with_spacing.append('<NEWLINE>')
                token_info.append('spacing')
            else:
                # Add a spacing token
                tokenized_with_spacing.append('<SPACE>')
                token_info.append('spacing')
        else:
            # Append the token and its type
            token_type = get_token_type(token)
            tokenized_with_spacing.append(token)
            token_info.append(token_type)

    return tokenized_with_spacing, token_info

# Function to iterate over files in directory
def tokenize_coq_files_in_directory(directory):
    all_tokens = []
    all_token_info = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.v'):
            file_path = os.path.join(directory, filename)
            # Get tokens and info
            tokens, token_info = tokenize_coq_file(file_path)
            # Append tokens to the list
            all_tokens.extend(tokens)
            # Append token types to the list
            all_token_info.extend(token_info)

    return all_tokens, all_token_info

# Funtion for creating sequences and labels
def create_sequences_and_labels(tokens, token_info, seq_length):
    sequences = []
    labels = []
    
    # Iterate over token list to create sequences
    for i in range(len(tokens) - seq_length):
        # Get the current sequence of tokens
        sequence = tokens[i:i + seq_length]
        # Get the next token type as the label
        if (tokens[i + seq_length] == "<SPACE>" or tokens[i + seq_length] == "<NEWLINE>"):
            label = tokens[i + seq_length]
        else:
            label = "other"
        # Append the sequence and label
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels

# Mapping token labels to indices
label_to_index = {'<SPACE>': 0, '<NEWLINE>': 1, 'other': 2}

# Dataset class
class CoqTokenDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = [label_to_index[label] for label in labels]  # Convert labels to indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Split data into training, validation, and test sets
def split_data(sequences, labels, test_size=0.2, val_size=0.1):
    # First split into training+validation and test sets
    train_val_seqs, test_seqs, train_val_labels, test_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=42
    )
    # Then split train+validation into training and validation sets
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        train_val_seqs, train_val_labels, test_size=val_size / (1 - test_size), random_state=42
    )
    return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels

# Create vocabulary mapping for tokens
def build_vocab(tokens):
    unique_tokens = list(set(tokens))
    token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
    return token_to_index

# Convert sequences of tokens to sequences of indices
def tokens_to_indices(sequences, token_to_index):
    return [[token_to_index[token] for token in seq] for seq in sequences]

# --------------------------------------------------------------------------------------------------------------------------------------------