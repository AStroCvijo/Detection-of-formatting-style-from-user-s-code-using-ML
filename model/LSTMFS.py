import torch
import torch.nn as nn

# Define the LSTM model from scratch
class LSTMFS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional):
        super(LSTMFS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define weights for the LSTM cell
        self.W_ii = nn.Linear(embedding_dim, hidden_dim)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.W_if = nn.Linear(embedding_dim, hidden_dim)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.W_ig = nn.Linear(embedding_dim, hidden_dim)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.W_io = nn.Linear(embedding_dim, hidden_dim)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fully connected output layer
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.size()

        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        outputs = []

        # Loop through each time step in the sequence
        for t in range(seq_len):
            x_t = embedded[:, t, :]

            # Input gate
            i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h_t))
            # Forget gate
            f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h_t))
            # Cell gate
            g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h_t))
            # Output gate
            o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h_t))

            # Update cell state
            c_t = f_t * c_t + i_t * g_t
            # Update hidden state
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        # Concatenate outputs along the sequence length dimension
        outputs = torch.cat(outputs, dim=1)

        # Use only the last output for classification
        output = self.fc(outputs[:, -1, :])
        return output
