import numpy as np
from typing import Tuple, Union, Optional

from tensorflow.python.keras.utils.np_utils import to_categorical

from tokenizer.tokenizer import CustomTokenizer
from keras.src.legacy.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, Dataset
from containers.container import Container
from dependency_injector.wiring import inject, Provide
from .mocked_corpus import get_mock_data
from .data_loader import TorchDataSet
from .text_loader import TextLoader


@inject
def prepare_data(
        sampling: Optional[Union[int, float]] = Provide[Container.config.sampling],
        batch_size: int = Provide[Container.config.batch_size],
        max_len: Optional[int] = Provide[Container.config.max_len],
        model_type: str = Provide[Container.config.model],
        tokenizer: CustomTokenizer = Provide[Container.tokenizer],
) -> DataLoader[Dataset]:
    """
    Prepares data for training by loading, cleaning, and tokenizing text sequences.

    Args:
        sampling (Union[int, float, None]): Sampling fraction or absolute number of samples.
        batch_size (int): Batch size for data loading.
        max_len (int, optional): Maximum sequence length for padding.
        model_type (str): Model type ("gpt" or other) for custom data preparation.
        tokenizer (Union[tiktoken.Encoding, Tokenizer]): Tokenizer instance for custom data preparation.

    Returns:
        Tuple[tiktoken.model, DataLoader[Dataset]]: Tokenizer and DataLoader with prepared data.
    """

    # Step 1: Load and clean text data
    text_sequences = _load_and_clean_texts(sampling)

    # Step 2: Initialize tokenizer, and tokenize sequences
    quantile_80 = int(np.quantile([len(text.split()) for text in text_sequences], q=0.8)) + 1
    padding_length = min(quantile_80, max_len)
    tokenizer.fit(texts=text_sequences)
    sequences = [tokenizer.encode(text, padding=True, padding_length=padding_length) for text in text_sequences]
    # Prepare data based on model type
    if model_type.lower() == "gpt":
        sequences = np.array(sequences)
        x, y = sequences[:, :-1], sequences[:, 1:]  # GPT shifts input by one token
    else:
        x, y = _prepare_lstm_data(sequences, padding_length, vocab_size=tokenizer.vocab_size)
    data_loader = _build_data_loader(x, y, batch_size)
    del text_sequences, sequences, x, y
    return data_loader


@inject
def _load_and_clean_texts(
        sampling: Optional[Union[int, float]],
        mock: bool = Provide[Container.config.mock_data]) -> list[str]:
    """
    Loads and cleans text sequences based on the provided sampling.

    Args:
        sampling (Union[int, float, None]): Sampling fraction or count.

    Returns:
        list: Cleaned text sequences.
    """
    if mock:
        return get_mock_data()
    text_sequences = TextLoader().get_texts(sampling=sampling)
    text_sequences = TextLoader.clean_texts(text_sequences, min_words=25)
    text_sequences = [f'<bos> {text} <eos>' for text in text_sequences]
    return text_sequences


def _prepare_lstm_data(sequences: np.ndarray, max_len: int, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for an LSTM model, generating input-output pairs from tokenized sequences.

    Args:
        sequences (np.ndarray): Tokenized and padded text sequences.
        max_len (int): Maximum length for each sequence.
        vocab_size (int): Size of the vocabulary.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Input and target arrays for LSTM.
    """
    len_size = len(sequences)
    x = np.zeros((len_size * (max_len - 1), max_len - 1), dtype=np.int32)
    y = np.zeros((len_size * (max_len - 1), 1), dtype=np.int32)
    current = 0

    for sequence in sequences:
        for index in range(1, len(sequence)):
            if sequence[index] == 0:
                break  # Stop at padding
            x[current, :index] = sequence[:index]
            y[current] = sequence[index]
            current += 1

    # Remove rows where all values are zero (padding)
    x = x[~np.all(x == 0, axis=1)]
    y = y[~np.all(y == 0, axis=1)]
    y = to_categorical(y, num_classes=vocab_size)
    return x, y


@inject
def _build_data_loader(
        x: np.ndarray, y: np.ndarray,
        batch_size: int, device: str = Provide[Container.config.device]) -> DataLoader:
    """
    Builds a DataLoader from the provided input-output pairs.

    Args:
        x (np.ndarray): Input data array.
        y (np.ndarray): Target data array.
        batch_size (int): Batch size for loading data.
        device (str): Device for data loading.

    Returns:
        DataLoader: DataLoader with batched data.
    """
    data = TorchDataSet(x=x, y=y, device=device)
    drop_last = False if batch_size > len(x) else True
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=0
    )
