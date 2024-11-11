import os
import torch
from time import time
from keras.api.preprocessing.sequence import pad_sequences
from torch import nn

# set random seed
torch.manual_seed(17)
device = 'cpu'  # torch.device('mps' if torch.mps.is_available() else 'cpu')


class Config:
    EMBEDDING_DIM = 16
    BATCH_SIZE = 32
    NUM_HEADS = 2
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 2
    FF_DIM = EMBEDDING_DIM * 4
    EPOCHS = 1000


class Tokenizer:
    def __init__(self, words):
        self.word_index = {word: index + 1 for index, word in enumerate(words)}
        self.word_index['<pad>'] = 0
        self.index_word = {index: word for word, index in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = [self.word_index[word] for word in text.lower().split()]
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = ' '.join([self.index_word[index] for index in sequence])
            texts.append(text)
        return texts


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(InputEmbedding, self).__init__()
        # Initialize embedding layer
        # Dimensions: (VOCAB_SIZE, 512) -> Each word is represented by a vector of size 512
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_sequence_length: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # no required gradient
        self.positional_encoding = torch.zeros(max_sequence_length, embedding_dim)
        self.positional_encoding.requires_grad = False
        # For each position in the sequence
        for pos in range(max_sequence_length):
            # For each dimension in the vector
            for i in range(0, embedding_dim, 2):
                sin_input = torch.tensor(pos / (10000 ** ((2 * i) / embedding_dim)))
                self.positional_encoding[pos, i] = torch.sin(sin_input)
                cos_input = torch.tensor(pos / (10000 ** ((2 * i) / embedding_dim)))
                self.positional_encoding[pos, i + 1] = torch.cos(cos_input)

    def forward(self, x):
        positional_wise_encoded = self.positional_encoding[:x.size(1), :]
        x = x + positional_wise_encoded
        return self.dropout(x)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.q_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def split_heads(self, x):
        batch_size = x.size(0)
        # (Batch, Seq Len, Embedding Len) -> (Batch, Num Heads, Seq Len, Head Dim)
        # Transpose: (Batch, Seq Len, Num Heads, Head Dim) -> (Batch, Num Heads, Seq Len, Head Dim)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def scale_dot_product_attention(self, Q, K, V, mask):
        _q = self.q_linear(Q)
        _k = self.k_linear(K)
        _v = self.v_linear(V)
        q = self.split_heads(_q)
        k = self.split_heads(_k)
        v = self.split_heads(_v)
        scale_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        out = torch.matmul(q, k.transpose(-2, -1)) / scale_factor
        if mask is not None:
            out = out.masked_fill(~mask, -torch.inf)
        dot_product = torch.softmax(out, dim=-1)
        attention = torch.matmul(dot_product, v)
        return attention

    def concat_heads(self, x):
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.embedding_dim)

    def forward(self, Q, K, V, mask=None):
        attention = self.scale_dot_product_attention(Q, K, V, mask)
        out = self.concat_heads(attention)
        out = self.out(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttentionLayer(embedding_dim, num_heads)
        self.ff = PositionWiseFeedForward(embedding_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
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
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttentionLayer(embedding_dim, num_heads)
        self.cross_attention = MultiHeadAttentionLayer(embedding_dim, num_heads)
        self.ff = PositionWiseFeedForward(embedding_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
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
                 dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.emb = InputEmbedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_sequence_length, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

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
                 dropout: float = 0.1):
        super(Decoder, self).__init__()
        self.emb = InputEmbedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_sequence_length, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_outputs=None, self_mask=None, cross_mask=None):
        x = self.emb(x).float()
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_outputs, self_mask, cross_mask)
        return x


class TransformerTextGenerator(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 max_sequence_length: int,
                 num_heads: int,
                 ff_dim: int,
                 num_layers: int,
                 dropout: float = 0.1
                 ):
        super(TransformerTextGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = nn.Dropout(dropout)
        self.decoder = Decoder(
            vocab_size,
            embedding_dim,
            max_sequence_length,
            num_heads,
            ff_dim,
            num_layers,
            dropout
        )
        self.output = nn.Linear(embedding_dim, vocab_size)

    def generate_text(self, tokenizer, seed_text, max_length=20):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                sequences = tokenizer.texts_to_sequences([seed_text])
                src = torch.tensor(sequences)
                if src.size(1) >= self.max_sequence_length:
                    break
                output = self(src)[:, -1, :]
                output = torch.softmax(output, dim=-1)
                output = torch.argmax(output, dim=-1)
                index = output.item()
                word = tokenizer.index_word[index]
                if word == '<eos>':
                    break
                seed_text += ' ' + word
            return seed_text

    def forward(self, x, mask=None):
        x = self.decoder(x, None, mask, None)
        x = self.dropout(x)
        return self.output(x)


def get_mock_data():
    corpus = [
        "The quick brown fox jumps over fences.",
        "Machine learning models improve with more data.",
        "Python programming is fun and highly versatile.",
        "Transformers revolutionized the field of NLP tasks.",
        "Artificial intelligence continues to evolve rapidly."
    ]
    import string
    # remove punctuation
    corpus = [line.translate(str.maketrans('', '', string.punctuation)) for line in corpus]
    corpus = [f'<bos> {line.lower()} <eos>' for line in corpus]
    words = list(set(' '.join(corpus).split()))
    words.sort()

    tokenizer = Tokenizer(words=words)
    sequences = tokenizer.texts_to_sequences(corpus)
    sequences = pad_sequences(sequences, maxlen=10, padding='post')
    sequences = torch.tensor(sequences, dtype=torch.long)
    return sequences.size(1), sequences, tokenizer


def build_casual_mask(seq_len):
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))


def train(model, sequences, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    seq_len = sequences.size(1) - 1
    mask = build_casual_mask(seq_len)

    for _ in range(epochs):
        src = sequences[:, :-1]
        trg = sequences[:, 1:]
        optimizer.zero_grad()
        out = model(src, mask)
        loss = loss_fn(out.transpose(1, 2), trg)
        loss.backward()
        optimizer.step()
        print('Epoch:', _, 'Loss:', loss.item())


def build_transformer_block(sequence_len: int, tokenizer: Tokenizer) -> TransformerTextGenerator:
    config = Config()
    return TransformerTextGenerator(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=config.EMBEDDING_DIM,
        max_sequence_length=sequence_len - 1,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT_RATE
    )


def main():
    start = time()
    with torch.device(device):
        sequence_len, sequences, tokenizer = get_mock_data()
        transformer = build_transformer_block(sequence_len, tokenizer)
        train(transformer, sequences, Config.EPOCHS)
        text = transformer.generate_text(tokenizer, seed_text="<bos>")
        print(text)
    end = time()
    print(round(end - start, 2), "seconds")


if __name__ == "__main__":
    main()
