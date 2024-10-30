import os
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Import the function for filtering coq files
from data.data_functions import filter_coq_files

# Import the functions for preprocessing the data
from data.data_functions import tokenize_coq_files_in_directory
from data.data_functions import create_sequences_and_labels
from data.data_functions import build_vocab
from data.data_functions import tokens_to_indices
from data.data_functions import split_data
from data.data_functions import CoqTokenDataset

# Import the function for training and evaluating the model
from train.train import train
from train.eval import eval

# Import the model
from model.LSTM import LSTM

if __name__ == "__main__":

    # Extract coq files
    input_folder = "data/math-comp"
    output_folder = "data/coq_files"
    filter_coq_files(input_folder, output_folder)

    # Directory containing coq files
    directory = 'data/coq_files'

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Preprocess the dataset
    # --------------------------------------------------------------------------------------------------------------------------------------------

    # Tokenize the coq code
    tokens, token_info = tokenize_coq_files_in_directory(directory)

    # Create sequences of tokens and their labels
    seq_length = 6
    sequences, labels = create_sequences_and_labels(tokens, token_info, seq_length)

    # Build the vocabulary and convert to indexed sequences
    token_to_index = build_vocab(tokens)
    indexed_sequences = tokens_to_indices(sequences, token_to_index)

    # Split into train, validation, and test sets
    train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = split_data(indexed_sequences, labels)

    # Create Dataset and DataLoader for each split
    train_dataset = CoqTokenDataset(train_seqs, train_labels)
    val_dataset = CoqTokenDataset(val_seqs, val_labels)
    test_dataset = CoqTokenDataset(test_seqs, test_labels)

    # Create the dataloaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Data loaders initialized")

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Initialize and train the model
    # --------------------------------------------------------------------------------------------------------------------------------------------

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model Constants
    embedding_dim = 768
    hidden_dim = 768
    output_dim = 3  # Three classes: <SPACE>, <NEWLINE>, other

    # Initialize the model and move it to the device
    vocab_size = len(token_to_index)
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Train the model
    num_epochs = 30
    print("Starting training...")
    train(num_epochs, train_loader, test_loader, model, device, optimizer, criterion)

    # Calculate and print the overall accuracy
    accuracy = calculate_accuracy(train_loader, model, device)
    print(f"Overall Accuracy: {accuracy:.4f}")
