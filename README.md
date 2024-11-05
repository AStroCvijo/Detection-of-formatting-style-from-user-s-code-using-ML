# Detection of formatting style from user's code using ML

Project made for the purpose of applying to JetBrains internship. <br>
Reproduction of the ["Learning to Format Coq Code Using Language Models"](https://arxiv.org/pdf/2006.16743v1) paper.

Date of creation: October, 2024

## Quickstart
1. Clone the repository:
    ```bash
    git clone https://github.com/AStroCvijo/detection_of_formatting_style.git
    ```

2. Download the [math-comp dataset](https://github.com/math-comp/math-comp), extract it, and paste the folder into the `detection_of_formatting_style\data` directory.

3. Navigate to the project directory:
    ```bash
    cd detection_of_formatting_style
    ```

4. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

5. Activate the virtual environment:
    - **Linux/macOS**:
      ```bash
      source venv/bin/activate
      ```
    - **Windows**:
      ```bash
      venv\Scripts\activate
      ```

6. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

7. Train the model using the default settings:
    ```bash
    python main.py
    ```

## Arguments guide 

### Model arguments
`-m or --model` Followed by the model you want to use: `LSTM`, `transformer`, or `n_gram`  
`-hs or --hidden_dim` Size of the models hidden layer
`-ed or --embedding_dim` Size of the embedding space
`-nl or --num_layers` Number of layers in the model

#### LSTM specific arguments
`-bi or --bidirectional` Will the LSTM be bidirectional

#### Transformer specific arguments
`-nh or --number_heads` Number of head of the Transformer model

### Training arguments
`-e or --epochs` Number of epochs in training  
`-lr or --learning_rate` Learning rate in training  

### Data arguments
`-sl or --sequence_length` Length of the sequences extracted from the data  
`-bs or --batch_size` Batch size  

## How to Use

 ### Training Example: 
`python main.py --train --model transformer --epochs 30 --learning_rate 0.0001 --sequence_length 6`

## Folder Tree

```
detection_of_formatting_style
├── data
│   ├── data_functions.py     # Contains functions for data preprocessing, loading, and transformation
│   └── math-comp             # The dataset folder
├── models
│   ├── transformer.py        # Transformer model implementation
│   ├── LSTM.py               # LSTM model implementation
│   └── n_gram.py             # n_gram model implementation
├── pretrained_models         # Directory for saving and loading pre-trained models
├── train
│   ├── eval.py               # Script to evaluate model performance
│   └── train.py              # Script to train and save models
├── utils
│   ├── argparser.py          # Contains argument parsing logic for CLI inputs
│   └── seed.py               # Contains functions for seed setting
└── main.py                   # Main script to run the project
```
