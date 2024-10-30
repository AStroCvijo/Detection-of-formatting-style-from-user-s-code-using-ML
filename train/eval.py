import torch
import torch.nn as nn
from model.LSTM import LSTM

# Function for calculating the accuracy
def eval(data_loader, model, device):
    # Set model to evaluation mode
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    # Calculate the correct and total predictions
    with torch.no_grad():
        for batch_sequences, batch_labels in data_loader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            outputs = model(batch_sequences)
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)
    
    # Set the model back to training mode
    model.train()
    
    return correct_predictions / total_predictions