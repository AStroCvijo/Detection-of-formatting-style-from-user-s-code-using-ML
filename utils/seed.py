import random
import numpy as np
import torch
import torch.nn as nn

# Function for setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function for initializing weights
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)
