import torch
import torch.nn as nn

# Define the n-gram model for text classification
class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(NGram, self).__init__()
        
        # Embedding layer to convert token indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Fully connected layer to map embeddings to output classes
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        # Get embeddings for input tokens and compute the average embedding for each input
        embedded = self.embedding(x).mean(dim=1)
        
        # Pass the averaged embeddings through the fully connected layer
        output = self.fc(embedded)
        return output
