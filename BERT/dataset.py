import random
import typing
# from collections import Counter
# from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
# from torchtext.vocab import vocab
# from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizerFast
import datasets
from datasets import Dataset as dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from tokenizers import BertWordPieceTokenizer
from tokenizers.trainers import WordPieceTrainer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class WikiDataset(Dataset):
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    MASK_PERCENTAGE = 0.15

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=False, tokenizer_path = None):

        df = pd.read_csv(path)
        self.ds: pd.Series = df['text']
        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]
        self.get_tokenizer(df.iloc[ds_from:ds_to], tokenizer_path)  # Loading the tokenizer we trained in previous step.

        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]

        self.df = self.get_dataset()
        # print(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()
        token_mask = torch.Tensor(item[self.TOKEN_MASK_COLUMN]).bool()

        mask_target = torch.Tensor(item[self.TARGET_COLUMN]).long()
        mask_target = mask_target.masked_fill_(token_mask, 0) # make all the non masked fields as zero, to facilitate loss calculation.

        # attention_mask = (inp == self.tokenizer.convert_tokens_to_ids(self.PAD)).unsqueeze(0) # add mask for padding, we don't want model to calculate attention on padding.
        attention_mask = (inp == self.tokenizer.token_to_id(self.PAD)).unsqueeze(0)
        if item[self.NSP_TARGET_COLUMN] == 0: # create binary output for next sentence prediction.
            t = [1, 0]
        else:
            t = [0, 1]

        nsp_target = torch.Tensor(t)
        return (
            inp.to(device), # (seq_len)
            attention_mask.to(device), # (1, seq_len)
            token_mask.to(device), # (seq_len)
            mask_target.to(device), # (seq_len)
            nsp_target.to(device) # (2)
        )

    def _batch_iterator(self, data, batch_size=10000):
        for i in tqdm(range(0, len(data), batch_size)):
            yield data[i : i + batch_size]["text"]

    def get_tokenizer(self, data, path):
        # check if tokenizer already exists.
        if path:
            self.tokenizer = Tokenizer.from_file(str(path))
            # self.tokenizer = AutoTokenizer.from_pretrained(path)
        # train a tokenizer.
        else:
          # Initialize a tokenizer
            self.tokenizer = BertWordPieceTokenizer()
            # trainer = WordPieceTrainer()
            # Customize training
            self.tokenizer.train_from_iterator(self._batch_iterator(data), special_tokens=[
                "[CLS]",
                "[PAD]",
                "[MASK]",
                "[UNK]",
                "[SEP]",
            ], min_frequency=2, vocab_size=5000)

            # Save tokenizer to disk
            self.tokenizer.save("tokenizer")

    def get_dataset(self) -> pd.DataFrame:
        sentences = []
        nsp = []
        sentence_lens = []

        # Split dataset on sentences
        for review in self.ds:
            review_sentences = review.split('. ')
            sentences += review_sentences
            self._update_length(review_sentences, sentence_lens)

        # selecting optimal length of the sentence i.e the sequence length, if anything is larger than this, it will be truncated and if anything is smaller than this it will be padded
        arr = np.array(sentence_lens)
        self.optimal_sentence_length = int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

        print("Preprocessing dataset")
        for review in tqdm(self.ds):
            review_sentences = review.split('. ')
            # print(review_sentences)
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    # True NSP item
                    # first, second = self.tokenizer(review_sentences[i]).tokens(), self.tokenizer(review_sentences[i + 1]).tokens()
                    first, second = self.tokenizer.encode(review_sentences[i]).tokens, self.tokenizer.encode(review_sentences[i + 1]).tokens
                    nsp.append(self._create_item(first, second, 1))

                    # False NSP item
                    first, second = self._select_false_nsp_sentences(sentences)
                    # first, second = self.tokenizer(first).tokens(), self.tokenizer(second).tokens()
                    first, second = self.tokenizer.encode(first).tokens, self.tokenizer.encode(second).tokens
                    nsp.append(self._create_item(first, second, 0))
        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths


    def _create_item(self, first: typing.List[str], second: typing.List[str], target: int = 1):
        # Create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())

        nsp_sentence = updated_first + [self.SEP] + updated_second[1:] # from because tokenizer gives CLS for second sentence as well.
        nsp_indices = [self.tokenizer.token_to_id(i) for i in nsp_sentence]
        # nsp_indices = self.tokenizer.encode(" ".join(nsp_sentence)).ids
        # nsp_indices = self.tokenizer.convert_tokens_to_ids(nsp_sentence)

        inverse_token_mask = first_mask + [True] + second_mask[1:]

        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)

        original_nsp_sentence = first[0] + [self.SEP] + second[0][1:]
        original_nsp_indices = [self.tokenizer.token_to_id(i) for i in original_nsp_sentence]
        # original_nsp_indices = self.tokenizer.encode(" ".join(original_nsp_sentence)).ids
        # original_nsp_indices = self.tokenizer.convert_tokens_to_ids(original_nsp_sentence)

        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indices,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target
            )
        else:
            return (
                nsp_indices,
                original_nsp_indices,
                inverse_token_mask,
                target
            )

    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        # Select sentence that is not present at index i+1.
        sentences_len = len(sentences)
        sentence_index = random.randint(0, sentences_len - 1)
        next_sentence_index = random.randint(0, sentences_len - 1)

        # To be sure that it's not real next sentence
        while next_sentence_index == sentence_index + 1:
            next_sentence_index = random.randint(0, sentences_len - 1)

        return sentences[sentence_index], sentences[next_sentence_index]

    def _preprocess_sentence(self, sentence: typing.List[str], should_mask: bool = True):
        inverse_token_mask = None
        if should_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
            sentence, inverse_token_mask = self._pad_sentence(sentence, inverse_token_mask)
        else:
            sentence =  self._pad_sentence(sentence)
        return sentence, inverse_token_mask

    def _mask_sentence(self, sentence: typing.List[str]):

        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]

        # Mask 15 % of all wordpiece tokens in each sentence.
        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)
            rand_num = random.random()
            # Change with mask token 80% of the time.
            if rand_num < 0.8:
                sentence[i] = self.MASK
            # keep the token as it is 10% of the time.
            elif rand_num < 0.9:
                pass
            # replace the token with some random token 10% of the time.
            else:
                # first 2000 tokens are unused tokens which can be used when we are fine-tuning on a specific task which will require adding tokens into vocabulary without splitting them using wordpiece tokenizer, the next 100 is just symbols.
                # j = random.randint(2100, self.tokenizer.vocab_size - 1)
                j = random.randint(2100, self.tokenizer.get_vocab_size() - 1)

                # sentence[i] = self.tokenizer.convert_ids_to_tokens(j)
                sentence[i] = self.tokenizer.id_to_token(j)
            inverse_token_mask[i] = False
        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool] = None):
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length] # truncate the sequence to match the optimal length.
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s) # expand the sequence to match the optimal length.

        # inverse token mask should be padded as well
        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[:self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (self.optimal_sentence_length - len_m)
        return s, inverse_token_mask




# ds = WikiDataset('/content/drive/MyDrive/transformer/wiki.csv', ds_from=0, ds_to=100,
#                       should_include_text=True, tokenizer_path = 'tokenizer')
# print(ds.df)