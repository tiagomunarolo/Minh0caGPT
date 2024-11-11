import torch
from torch import nn
from torch.utils.data import DataLoader
from keras.src.utils import pad_sequences
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Set the fraction of memory to use
torch.mps.set_per_process_memory_fraction(0.0)


class LSTMTextGenerator(nn.Module):

    def __init__(self,
                 tokenizer,
                 embedding_dim: int,
                 max_length: int,
                 hidden_size: int,
                 num_layers: int,
                 device='cpu'
                 ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.tokenizer.pad_token,
            device=self.device
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            device=self.device
        )

        self.l_out = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            device=self.device
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, state_h, state_c):
        x = self.embedding(x)
        x, (state_h, state_c) = self.lstm(x, (state_h, state_c))
        x = self.dropout(x)
        x = self.l_out(x)[:, -1, :]
        return x, state_h, state_c

    def generate_text(self, seed_text, max_length=20):
        self.eval()
        with torch.no_grad():
            state_h, state_c = self.init_hidden(batch_size=1)
            for _ in range(max_length):
                sequences = self.tokenizer.encode(seed_text)
                sequences = pad_sequences(
                    sequences=[sequences],
                    maxlen=self.max_length,
                    padding='post',
                    value=self.tokenizer.pad_token
                )
                x = torch.tensor(sequences, dtype=torch.long).to(self.device)
                state_h, state_c = state_h.to(self.device), state_c.to(self.device)
                output, state_h, state_c = self(x, state_h, state_c)
                output = torch.softmax(output, dim=-1)
                index = torch.argmax(output, dim=-1).item()
                word = self.tokenizer.decode([index])
                if index == self.tokenizer.eos_token:
                    seed_text += ' <eos>'
                    break
                seed_text += ' ' + word
            return seed_text

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        )

    def fit(self, data_loader: DataLoader, epochs: int, summary_writer: SummaryWriter):
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in tqdm(range(epochs), desc="Training"):
            self.train()
            loss_val = 0
            state_h, state_c = None, None

            for x, y in data_loader:
                if state_h is None or state_c is None:
                    state_h, state_c = self.init_hidden(batch_size=x.size(0))
                else:
                    state_h, state_c = state_h.detach(), state_c.detach()

                optimizer.zero_grad(set_to_none=True)
                predictions, state_h, state_c = self(x, state_h, state_c)
                # Reshape predictions and targets if only using the last time step
                loss = loss_fn(predictions, y.float())
                loss_val += loss.detach()
                loss.backward()
                optimizer.step()
        summary_writer.close()
        return self
