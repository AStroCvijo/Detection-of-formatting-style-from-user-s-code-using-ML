import torch
import torch.nn as nn
from model.LSTM import LSTM

# Function to calculate accuracy
def eval(data_loader, model, device):
    # Set model to evaluation mode
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    # Iterate through data and calculate correct and total predictions
    with torch.no_grad():
        for batch_sequences, batch_labels in data_loader:
            # Move data to the device
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_sequences)
            
            # Get predicted labels
            _, predicted_labels = torch.max(outputs, dim=1)
            
            # Update counts of correct and total predictions
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)
    
    # Return model to training mode
    model.train()
    
    # Return accuracy as a proportion
    return correct_predictions / total_predictions
