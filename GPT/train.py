from model import build_GPT
from dataset import ShakespeareDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
import pandas as pd

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset as dataset
from datasets import DatasetDict

def greedy_decode(model, input, tokenizer, max_len, device):

    input_token = tokenizer.token_to_id(input)
    decoder_input = torch.tensor([input_token], dtype=torch.int64).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(decoder_input, decoder_mask) # (1, seq_len, d_model)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

    return decoder_input.squeeze(0)

def get_all_sentences(ds, column_name):
    # print(ds)
    for item in ds:
        yield item[column_name]

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, 'text'), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(f"{config['datasource']}", split='train')
    df = pd.read_csv('/content/drive/MyDrive/transformer/GPT/shakespeare.csv')
    ds_raw = dataset.from_pandas(df)
    dataset_dict = DatasetDict({
          'train': dataset,

      })['train']
    print(ds_raw)
    # Build tokenizer
    tokenizer = get_or_build_tokenizer(config, ds_raw)


    train_ds_size = len(ds_raw)
    train_ds_raw = ds_raw

    train_ds = ShakespeareDataset(train_ds_raw, tokenizer, config['seq_len'], config['column_name'])


    # Find the maximum length of each sentence in the source and target sentence
    max_len = 0


    for item in ds_raw:
        src_ids = tokenizer.encode(item[config['column_name']]).ids
        max_len = max(max_len, len(src_ids))

    print(f'Max length of source sentence: {max_len}')



    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)


    return train_dataloader, tokenizer

def get_model(config, vocab_len):
    model = build_GPT(vocab_len, config["seq_len"], d_model=config['d_model'], N=config['layers'], h=config['heads'], dropout=0.1, d_ff=4*config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, tokenizer = get_ds(config)
    print(tokenizer.get_vocab_size())
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            pos = torch.arange(0, decoder_input.shape[1], dtype=torch.long, device=device).unsqueeze(0) # shape (1, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            decoder_output = model.decode(decoder_input, decoder_mask, pos) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)ss

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    print(config)
    train_model(config)
