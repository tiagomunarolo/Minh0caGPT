import torch
from torch import nn
from time import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from keras.api.models import Model

# Disable debugging features to speed up training
torch.autograd.set_detect_anomaly(False)
# Set precision to medium
torch.set_float32_matmul_precision('medium')
# set random seed
torch.manual_seed(42)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device='cpu'):
        super(InputEmbedding, self).__init__()
        # Initialize embedding layer
        # Dimensions: (VOCAB_SIZE, 512) -> Each word is represented by a vector of size 512
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_sequence_length: int, dropout: float = 0.1, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # no required gradient
        position = torch.arange(0, max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_sequence_length, embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return self.dropout(x)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, device='cpu'):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.head_dim = embedding_dim // num_heads
        self.q_linear = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)

    def split_heads(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # (Batch, Seq Len, Embedding Len) -> (Batch, Num Heads, Seq Len, Head Dim)
        # Transpose: (Batch, Seq Len, Num Heads, Head Dim) -> (Batch, Num Heads, Seq Len, Head Dim)
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def scale_dot_product_attention(self, Q, K, V, mask):
        q = self.split_heads(self.q_linear(Q))
        k = self.split_heads(self.k_linear(K))
        v = self.split_heads(self.v_linear(V))
        out = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            out = out.masked_fill(~mask, -torch.inf)
        dot_product = torch.softmax(out, dim=-1)
        attention = torch.matmul(dot_product, v)
        return attention

    def concat_heads(self, x):
        batch, num_heads, seq_len, head_dim = x.size()
        return x.permute(0, 2, 1, 3).reshape(batch, seq_len, self.embedding_dim)

    def forward(self, Q, K, V, mask=None):
        attention = self.scale_dot_product_attention(Q, K, V, mask)
        out = self.concat_heads(attention)
        out = self.out(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1, device='cpu'):
        super(PositionWiseFeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim, device=device),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim, device=device)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.ff(x))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, device='cpu'):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttentionLayer(embedding_dim, num_heads, device=device)
        self.ff = PositionWiseFeedForward(embedding_dim, ff_dim, device=device)
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention = self.attention(x, x, x, mask)
        # Add and Normalize
        x = self.norm1(attention + x)
        # Feed Forward
        ff = self.ff(x)
        # Add and Normalize
        x = self.norm2(ff + x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, device='cpu'):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttentionLayer(embedding_dim, num_heads, device=device)
        self.cross_attention = MultiHeadAttentionLayer(embedding_dim, num_heads, device=device)
        self.ff = PositionWiseFeedForward(embedding_dim, ff_dim, device=device)
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        self.norm3 = nn.LayerNorm(embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, self_mask=None, cross_mask=None):
        attention = self.attention(x, x, x, self_mask)
        # Add and Normalize
        x = self.norm1(attention + x)
        # Cross Attention
        if encoder_outputs is not None:
            cross_attention = self.cross_attention(x, encoder_outputs, encoder_outputs, cross_mask)
            # Add and Normalize
            x = self.norm2(cross_attention + x)
        # Feed Forward
        ff = self.ff(x)
        # Add and Normalize
        x = self.norm3(ff + x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 max_sequence_length: int,
                 num_heads: int,
                 ff_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 device='cpu',
                 ):
        super(Encoder, self).__init__()
        self.emb = InputEmbedding(vocab_size, embedding_dim, device=device)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_sequence_length, dropout, device=device)
        self.layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads, ff_dim, dropout, device=device) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 max_sequence_length: int,
                 num_heads: int,
                 ff_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 device='cpu',
                 ):
        super(Decoder, self).__init__()
        self.emb = InputEmbedding(vocab_size, embedding_dim, device=device)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_sequence_length, dropout, device=device)
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, ff_dim, dropout, device=device) for _ in range(num_layers)])

    def forward(self, x, encoder_outputs=None, self_mask=None, cross_mask=None):
        x = self.emb(x).float()
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_outputs, self_mask, cross_mask)
        return x


class Gpt2TextGenerator(nn.Module):
    def __init__(self,
                 tokenizer,
                 max_length,
                 embedding_dim,
                 num_heads=8,
                 ff_dim=2048,
                 num_layers=6,
                 dropout_rate=0.1,
                 device='cpu'
                 ):
        super(Gpt2TextGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.num_layers = num_layers
        # Model Components
        self.mask = None
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(
            self.vocab_size,
            self.embedding_dim,
            self.max_length,
            self.num_heads,
            self.ff_dim,
            self.num_layers,
            dropout_rate,
            device=device
        )
        self.output = nn.Linear(self.embedding_dim, self.vocab_size, device=device)
        self.device = device

    def forward(self, x):
        x = self.decoder(x, None, self.mask, None)
        x = self.dropout(x)
        return self.output(x)

    def generate_text(self, seed_text, max_length=20):
        self.eval()
        self.mask = None
        with torch.no_grad():
            with torch.device(self.device):
                for _ in range(max_length):
                    sequences = self.tokenizer.encode(seed_text)
                    src = torch.tensor(sequences).unsqueeze(0)
                    if src.size(1) >= self.max_length:
                        break
                    output = self(src)[:, -1, :]
                    output = torch.softmax(output, dim=-1)
                    output = torch.argmax(output, dim=-1)
                    index = output.item()
                    word = self.tokenizer.decode([index])
                    if index == self.tokenizer.eos_token:
                        seed_text += ' <eos>'
                        break
                    seed_text += ' ' + word
                return seed_text

    def _log_prediction(self, epoch):
        text = self.generate_text(seed_text='<bos>')
        print('Epoch:', epoch, 'Prediction:', text)

    @staticmethod
    def _log_tensorboard(
            summary_writer: SummaryWriter,
            epochs: int,
            epoch_loss: float,
            current_epoch: int,
            batch_size: int,
            run_time: float):
        summary_writer.add_scalar(tag=f'Loss Per Batch (E={epochs}, B={batch_size})',
                                  scalar_value=epoch_loss / batch_size, global_step=current_epoch)
        summary_writer.add_scalar(tag=f'Time (E={epochs}, B={batch_size})',
                                  scalar_value=run_time,
                                  global_step=current_epoch)

    def fit(self,
            data_loader: DataLoader[TensorDataset],
            epochs: int,
            summary_writer: SummaryWriter
            ) -> Model:
        with torch.device(self.device):
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token, label_smoothing=0.1)
            optimizer = torch.optim.AdamW(params=self.parameters(), lr=0.001)
            for epoch in range(epochs):
                self.train()
                epoch_loss = 0
                start = time()
                for (x, y) in data_loader:
                    if self.mask is None:
                        self.mask = torch.tril(torch.ones(x.size(1), x.size(1), dtype=torch.bool))
                    optimizer.zero_grad(set_to_none=True)
                    predictions = self(x).view(-1, self.vocab_size)
                    loss = loss_fn(input=predictions, target=y.view(-1))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach()
                end = time()
                # Add to summary writer
                self._log_tensorboard(
                    summary_writer=summary_writer,
                    epochs=epochs,
                    current_epoch=epoch,
                    epoch_loss=epoch_loss,
                    batch_size=data_loader.batch_size,
                    run_time=end - start
                )
                print(f'Epoch ({self.device}): {epoch} | Loss: {epoch_loss / len(x)} | Time:  {end - start}')
                self._log_prediction(epoch=epoch)

            summary_writer.close()
            return self
