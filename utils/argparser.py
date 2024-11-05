import argparse

# Function for parsing arguments
def arg_parse():

    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('-m',   '--model',             type=str,   default = 'Transformer',  help="Model to use (e.g., LSTM, Transformer, N-gram)")
    parser.add_argument('-hs',  '--hidden_dim',        type=int,   default = 768,            help="Size of the models hidden layer")
    parser.add_argument('-ed',  '--embedding_dim',     type=int,   default = 768,            help="Size of the embedding space")
    parser.add_argument('-nl',  '--num_layers',        type=int,   default = 1,              help="Number of layers in the model")

    # LSTM specific arguments
    parser.add_argument('-bi',  '--bidirectional',     type=bool,  default = False,          help="Whether the LSTM is bidirectional (True/False)")

    # Transformer specific arguments
    parser.add_argument('-nh',  '--number_heads',      type=int,   default = 8,              help="Number of heads of the Transformer model")

    # Training arguments
    parser.add_argument('-e',   '--epochs',            type=int,   default = 30,             help="Number of epochs in training")
    parser.add_argument('-lr',  '--learning_rate',     type=float, default = 0.00002,        help="Learning rate in training")
    parser.add_argument('-mn',   '--model_name',       type=str,   default = 'model',        help="Name of the model when being saved")

    # Data arguments
    parser.add_argument('-sl',  '--sequence_length',   type=int,   default = 10,             help="Length of input sequences")
    parser.add_argument('-bs',  '--batch_size',        type=int,   default = 256,            help="Batch size for training")

    # Parse the arguments
    return parser.parse_args()
