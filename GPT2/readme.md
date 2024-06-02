# Language Models are Unsupervised Multitask Learners

<p align="center">
<img src="images/gpt2.png?raw=true" alt="GPT-Architecture" width="350"/>
</p>


This repository contains a PyTorch implementation of the GPT2 (Generative Pre-trained Transformer) model for text generation tasks. The model is trained on the Shakespeare dataset, but it can be easily adapted to other text datasets.

The code is entirely similar to the GPT code with some minor changes stated in paper: 
"Layer normalization (Ba et al., 2016)
was moved to the input of each sub-block, similar to a
pre-activation residual network (He et al., 2016) and an
additional layer normalization was added after the final selfattention block." 

## Files

### config.py

This file contains the configuration settings for the model and the training process. The `get_config()` function returns a dictionary with the following keys:

- `layers`: Number of decoder layers.
- `heads`: Number of attention heads.
- `batch_size`: Batch size for training.
- `num_epochs`: Number of training epochs.
- `lr`: Learning rate.
- `seq_len`: Maximum sequence length.
- `d_model`: Dimensionality of the model.
- `datasource`: Name of the dataset (currently set to 'shakespeare').
- `model_folder`: Folder to store the trained model weights.
- `model_basename`: Base name for the model weights file.
- `preload`: Specifies whether to load a pre-trained model before training (can be 'latest' or a specific epoch number).
- `tokenizer_file`: Name of the tokenizer file.
- `experiment_name`: Name of the experiment for TensorBoard logging.
- `column_name`: Name of the column containing the text data in the dataset.

The `get_weights_file_path()` function generates the file path for storing the model weights at a specific epoch. The `latest_weights_file_path()` function finds the latest weights file in the weights folder.

### dataset.py

This file contains the `ShakespeareDataset` class, which is a PyTorch `Dataset` subclass for loading and preprocessing the Shakespeare dataset. The `__getitem__()` method tokenizes the text data, applies padding, and returns a dictionary containing the decoder input, decoder mask, label, and source text.

The `causal_mask()` function creates a causal mask for the decoder self-attention, preventing the model from attending to future tokens during training.

### model.py

This file contains the implementation of the GPT model and its components. The main components are:

- `LayerNormalization`: Layer normalization module.
- `FeedForwardBlock`: Feed-forward block for the decoder.
- `InputEmbeddings`: Combined text and position embeddings.
- `ResidualConnection`: Residual connection module.
- `MultiHeadAttentionBlock`: Multi-head attention block for the decoder.
- `DecoderBlock`: Decoder block consisting of a self-attention block and a feed-forward block with residual connections.
- `Decoder`: Decoder module composed of multiple decoder blocks.
- `ProjectionLayer`: Final projection layer that maps the decoder output to the vocabulary size.
- `GPT`: Main GPT model that combines the input embeddings, decoder, and projection layer.

The `build_GPT()` function constructs and initializes the GPT model with the specified hyperparameters.

### train.py

This file contains the training loop and auxiliary functions for training the GPT model. The main functions are:

- `greedy_decode()`: Performs greedy decoding using the trained model.
- `get_all_sentences()`: Generator function to iterate over the sentences in the dataset.
- `get_or_build_tokenizer()`: Builds or loads a tokenizer for the dataset.
- `get_ds()`: Loads and preprocesses the dataset, creating the training dataloader.
- `get_model()`: Initializes the GPT model with the specified configuration.
- `train_model()`: Main training loop that iterates over the training dataset, computes the loss, performs backpropagation, and saves the model weights at the end of each epoch.

## Usage

1. Modify the config file.
2. Run the `train.py` script to start the training process: `python train.py`

During training, the script will:

- Load or build a tokenizer for the dataset.
- Initialize the GPT model with the specified configuration.
- Load a pre-trained model weights if specified in the configuration.
- Iterate over the training dataset, compute the loss, perform backpropagation, and update the model weights.
- Save the model weights at the end of each epoch.
- Log the training loss to TensorBoard.

After training, you can use the trained model for text generation tasks by loading the saved model weights and using the `greedy_decode()` function.

## Customization

To use this implementation with a different dataset, you can modify the `ShakespeareDataset` class in `dataset.py` and update the `get_ds()` function in `train.py` accordingly. Additionally, you may need to update the configuration settings in `config.py` to match your dataset and desired hyperparameters.

## Acknowledgments

This implementation is based on the GPT2 paper by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).