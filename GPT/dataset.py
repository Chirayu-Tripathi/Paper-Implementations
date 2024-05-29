import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len, column_name):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer = tokenizer
        self.column_name = column_name
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        target = self.ds[idx]
        text = target[self.column_name]
        text = text[:self.seq_len]
        # print(self.seq_len)
        # Transform the text into tokens
        input_tokens = self.tokenizer.encode(text).ids


        # Add padding to each sentence
        num_padding_tokens = self.seq_len - len(input_tokens) + 1 # plus one, because one token is left out from both input and label.

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if num_padding_tokens < 0 :
            raise ValueError("Sentence is too long")


        # Add padding tokens to the decoder input.
        decoder_input = torch.cat(
            [
                torch.tensor(input_tokens[:-1], dtype=torch.int64), # leave the last token, to calculate the loss against the label.
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only [EOS] token
        label = torch.cat(
            [
                torch.tensor(input_tokens[1:], dtype=torch.int64),
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        # print(decoder_input.size(0), self.seq_len)
        # Double check the size of the tensors to make sure they are all seq_len long

        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        # print(decoder_input)
        # print(label)
        return {
            "decoder_input": decoder_input,  # (seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0