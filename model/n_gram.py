import torch
import torch.nn as nn

# Define the n-gram model
class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(NGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        # Average the embeddings of the input tokens
        embedded = self.embedding(x).mean(dim=1)
        output = self.fc(embedded)
        return output