# GPT2 with LoRA (Low-Rank Adaptation) Implementation in PyTorch

<p align="center">
<img src="images/architecture.png?raw=true" alt="GPT-Architecture" width="350"/>
</p>

This repository contains a PyTorch implementation of the Low-Rank Adaptation (LoRA) technique for fine-tuning pre-trained language models, specifically GPT-2. LoRA is a parameter-efficient method for adapting large pre-trained models to specific downstream tasks, requiring significantly fewer trainable parameters compared to full fine-tuning.

## Overview

The code demonstrates the following:

1. Loading a pre-trained GPT-2 model and tokenizer from the Hugging Face Transformers library.
2. Applying LoRA to the attention layers of the GPT-2 model.
3. Creating a custom dataset and dataloader for training.
4. Fine-tuning the LoRA-adapted GPT-2 model on a text dataset.
5. Generating text samples using the fine-tuned model.
6. Saving the fine-tuned model for future use.

## Requirements

To run this code, you'll need the following:

- Python 3.6 or higher
- PyTorch
- Transformers (Hugging Face library)
- pandas
- tqdm

## Usage

1. Clone this repository.
2. Install the required dependencies.
3. Prepare your text dataset and update the `GPT2Dataset` class accordingly.
4. Run the notebook to fine-tune the LoRA-adapted GPT-2 model on your dataset.
5. Use the fine-tuned model for text generation or other downstream tasks.

## Acknowledgments

This implementation is based on the LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

The code utilizes the Hugging Face Transformers library for loading and working with pre-trained language models.
