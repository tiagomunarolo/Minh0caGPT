from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    LSTM: str = "lstm"
    GPT: str = "gpt"


@dataclass(frozen=True)
class ConfigGPT:
    """
    Configuration class for the model and data preparation
    """
    # Model params
    EPOCHS = 100
    BATCH_SIZE = 256  # Number of samples processed in each iteration
    EMBEDDING_DIM = 128  # Dimension of the word embedding vectors
    NUM_HEADS = 8  # Number of attention heads
    FF_DIM = EMBEDDING_DIM * 4  # Dimension of the feed-forward network
    NUM_TRANSFORMER_LAYERS = 4  # Number of transformer layers


class ConfigLSTM:
    """
    Configuration class for LSTM model and data preparation
    """
    # Model params
    EPOCHS = 100
    BATCH_SIZE = 32  # Number of samples processed in each iteration
    EMBEDDING_DIM = 128  # Dimension of the word embedding vectors
    HIDDEN_LSTM_SIZE = 128  # Hidden size of the LSTM layers
    NUM_LSTM_LAYERS = 1  # Number of LSTM layers


@dataclass
class Config:
    """
    Configuration class for the model and data preparation
    """
    # Data params
    MOCKED_DATA = False  # Whether to use mocked data or not
    # Model params
    MODEL = ModelType.LSTM  # Specify that a GPT or LSTM model will be used
    FORCE_RETRAIN = True  # Whether to force retraining the model even if it was previously trained
    # NLP params
    MAX_LENGTH = 128  # Maximum length of input sequences
    SAMPLING = 0.1  # Percentage or number of sentences to sample from the dataset
    # Tokenizer params
    TOKENIZER_TYPE = 'keras'
    CONFIG_LSTM = ConfigLSTM()
    CONFIG_GPT = ConfigGPT()
    # Device for training
    DEVICE = 'mps'
