import time
# from datetime import datetime
import datetime
from pathlib import Path

import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import WikiDataset
from model import build_bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def percentage(batch_size: int, max_index: int, current_index: int):
    batched_max = max_index // batch_size
    return round(current_index / batched_max * 100, 2)


def nsp_accuracy(result: torch.Tensor, target: torch.Tensor):
    s = (result.argmax(1) == target.argmax(1)).sum()
    return round(float(s / result.size(0)), 2)


def token_accuracy(result: torch.Tensor, target: torch.Tensor, inverse_token_mask: torch.Tensor):
    r = result.argmax(-1).masked_select(~inverse_token_mask)
    t = target.masked_select(~inverse_token_mask)
    s = (r == t).sum()
    return round(float(s / (result.size(0) * result.size(1))), 2)


class BertTrainer:

    def __init__(self,
                 model,
                 dataset: WikiDataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 0.005,
                 epochs: int = 5,
                 ):
        self.model = model
        self.dataset = dataset

        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir

        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.ml_criterion = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1).to(device) # 1 is token id for PAD.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)

        self._splitter_size = 35

        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size

        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def print_summary(self):
        ds_len = len(self.dataset)

        print("Model Summary\n")
        print('=' * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {ds_len}")
        print(f"Max / Optimal sentence len: {self.dataset.optimal_sentence_length}")
        # print(f"Vocab size: {len(self.dataset.vocab)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batched dataset len: {self._batched_len}")
        print('=' * self._splitter_size)
        print()

    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            loss = self.train(self.current_epoch)
            self.save_checkpoint(self.current_epoch, step=-1, loss=loss)

    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")

        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0
        for i, value in enumerate(self.loader):
            index = i + 1
            # inp ---> (batch, seq_len), mask ---> (batch, 1, seq_len), inverse_token_mask ---> (batch, seq_len), token_target ---> (batch, seq_len), nsp_target ---> (batch, 2)
            inp, mask, inverse_token_mask, token_target, nsp_target = value
            mask = mask.unsqueeze(1) # (batch, 1, 1, seq_len) each head will face the same sequence so increase a dimention at pos 1.
            self.optimizer.zero_grad()

            token, nsp = self.model(inp, mask) # token ---> (batch, seq_len, vocab_size), nsp ---> (batch, 2)

            tm = inverse_token_mask.unsqueeze(-1).expand_as(token) # (batch, seq_len, vocab_size)

            token = token.masked_fill(tm, 0) # replace non masked position with zero for cross entropy loss.

            loss_token = self.ml_criterion(token.view(-1, 5000), token_target.view(-1)) # ([batch * seq_len, vocab_size], [batch * seq_len])
            loss_nsp = self.criterion(nsp, nsp_target)

            loss = loss_token + loss_nsp
            average_nsp_loss += loss_nsp
            average_mlm_loss += loss_token

            loss.backward()
            self.optimizer.step()

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                s = self.training_summary(elapsed, index, average_nsp_loss, average_mlm_loss)

                if index % self._accuracy_every == 0:
                    s += self.accuracy_summary(index, token, nsp, token_target, nsp_target, inverse_token_mask)

                print(s)

                average_nsp_loss = 0
                average_mlm_loss = 0
        return loss

    def training_summary(self, elapsed, index, average_nsp_loss, average_mlm_loss):
        passed = percentage(self.batch_size, self._ds_len, index)
        global_step = self.current_epoch * len(self.loader) + index

        print_nsp_loss = average_nsp_loss / self._print_every
        print_mlm_loss = average_mlm_loss / self._print_every

        s = f"{time.strftime('%H:%M:%S', elapsed)}"
        s += f" | Epoch {self.current_epoch + 1} | {index} / {self._batched_len} ({passed}%) | " \
             f"NSP loss {print_nsp_loss:6.2f} | MLM loss {print_mlm_loss:6.2f}"

        self.writer.add_scalar("NSP loss", print_nsp_loss, global_step=global_step)
        self.writer.add_scalar("MLM loss", print_mlm_loss, global_step=global_step)
        return s

    def accuracy_summary(self, index, token, nsp, token_target, nsp_target, inverse_token_mask):
        global_step = self.current_epoch * len(self.loader) + index
        nsp_acc = nsp_accuracy(nsp, nsp_target)
        token_acc = token_accuracy(token, token_target, inverse_token_mask)

        self.writer.add_scalar("NSP train accuracy", nsp_acc, global_step=global_step)
        self.writer.add_scalar("Token train accuracy", token_acc, global_step=global_step)

        return f" | NSP accuracy {nsp_acc} | Token accuracy {token_acc}"

    def save_checkpoint(self, epoch, step, loss):
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.datetime.utcnow().timestamp():.0f}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, CHECKPOINT_DIR + '/' + name)

        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print('=' * self._splitter_size)
        print()

    def load_checkpoint(self, path: Path):
        print('=' * self._splitter_size)
        print(f"Restoring model {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")
        print('=' * self._splitter_size)


EMB_SIZE = 16
HIDDEN_SIZE = 64  # 4 * EMB_SIZE
EPOCHS = 4
BATCH_SIZE = 32
NUM_HEADS = 1

CHECKPOINT_DIR = 'bert_checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

timestamp = datetime.datetime.utcnow().timestamp()

LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Prepare dataset")

    ds = WikiDataset('/content/drive/MyDrive/transformer/wiki.csv', ds_from=0, ds_to=100,
                    should_include_text=True, tokenizer_path = None)

    bert = build_bert(5000, 512, EMB_SIZE, NUM_HEADS, NUM_HEADS, 0.1, HIDDEN_SIZE).to(device)
    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        print_progress_every=20,
        print_accuracy_every=200,
        batch_size=BATCH_SIZE,
        learning_rate=0.0001,
        epochs=15
    )

    trainer.print_summary()
    trainer()