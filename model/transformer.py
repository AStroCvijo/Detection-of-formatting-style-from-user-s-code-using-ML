import torch
import torch.nn as nn

# Define the Transformer-based model for sequence classification
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, nhead):
        super(Transformer, self).__init__()
        
        # Embedding layer to map token indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding to capture token position information
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embedding_dim))
        
        # Transformer encoder with specified number of layers and attention heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layer to produce the final output
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        # Apply embedding and add positional encoding to each token embedding
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        # Pass through Transformer encoder (transpose for correct input shape)
        transformer_out = self.transformer_encoder(embedded.transpose(0, 1))
        
        # Output layer on the last token's representation (alternatively, can average across tokens)
        output = self.fc(transformer_out[-1])
        
        return output
