import torch
import torch.nn as nn

# Define the Transformer-based model
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, nhead):
        super(Transformer, self).__init__()
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embedding_dim))  # Adjust max length as needed
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected output layer
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        # Embedding and add positional encoding
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        # Pass through transformer encoder (transpose for compatibility with torch Transformer)
        transformer_out = self.transformer_encoder(embedded.transpose(0, 1))
        
        # Output layer on the last time step (or average pool for full sequence output)
        output = self.fc(transformer_out[-1])
        
        return output
