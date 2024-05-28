# BERT Implementation


<img src="images/architecture.png?raw=true" alt="Bert-Architecture" width="350"/>


This repository contains an implementation of the BERT (Bidirectional Encoder Representations from Transformers) model, which is a popular language model used for various natural language processing tasks.


## Files

### model.py

This file contains the implementation of the BERT model architecture, including the following components:

- `LayerNormalization`: A layer normalization module used in the residual connections.
- `FeedForwardBlock`: The feed-forward block used in each encoder block.
- `InputEmbeddings`: The input token embedding layer.
- `SegmentEmbeddings`: The segment embedding layer used to differentiate between sentence A and sentence B.
- `PositionalEncoding`: The positional encoding layer used to encode the position of each token in the sequence.
- `ResidualConnection`: The residual connection module used in each encoder block.
- `MultiHeadAttentionBlock`: The multi-head attention block used in each encoder block.
- `PreTrain`: The pre-training module for masked language modeling (MLM) and next sentence prediction (NSP) tasks.
- `EncoderBlock`: The encoder block consisting of a multi-head attention block and a feed-forward block.
- `Encoder`: The encoder module consisting of a stack of encoder blocks.
- `BERT`: The main BERT model class that combines the encoder, pre-training module, and various embedding layers.
- `build_bert`: A helper function to construct a BERT model with given hyperparameters.

### dataset.py

This file contains the implementation of the `WikiDataset` class, which is a custom PyTorch dataset for pre-training the BERT model on Wikipedia data. The dataset class includes methods for:

- Loading and preprocessing the data
- Creating masked language modeling (MLM) and next sentence prediction (NSP) examples
- Tokenizing the input sequences using a WordPiece tokenizer
- Padding and truncating sequences to a fixed length
- Masking and replacing tokens for the MLM task

### train.py

This file contains the implementation of the `BertTrainer` class, which is responsible for training the BERT model on the `WikiDataset`. The trainer class includes methods for:

- Initializing the model, dataset, and optimizer
- Training the model for a specified number of epochs
- Computing the MLM and NSP losses
- Evaluating the model's accuracy on the MLM and NSP tasks
- Saving and loading model checkpoints
- Logging training progress and metrics using TensorBoard

The file also includes some configuration settings for the training process, such as the batch size, learning rate, and number of epochs.

## Usage

1. Prepare your dataset by placing it in a CSV file with a column named 'text' containing the text data.
2. Modify the configuration settings in `train.py` according to your requirements.
3. Run `train.py` to start the training process.

During training, the script will print the progress, loss values, and accuracy metrics for both the MLM and NSP tasks. The trained model checkpoints will be saved in the `bert_checkpoints` directory, and the training logs will be saved in the `logs` directory for visualization using TensorBoard.

## Dependencies

The following Python packages are required to run this implementation:

- PyTorch
- NumPy
- Pandas
- tqdm
- Tokenizers
- Datasets

Note: The code assumes the availability of a GPU for faster training. If you don't have a GPU, you can modify the `device` variable in `model.py` and `train.py` to use the CPU instead.

## Acknowledgments

This implementation is based on the BERT paper by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).