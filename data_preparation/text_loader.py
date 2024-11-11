import pandas as pd
import nltk
import string
import unicodedata
from pathlib import Path
from typing import Optional

# Constants for file paths
DATA_FILE: Path = Path(__file__).resolve().parent / "obras_machado_de_assis.csv"


class TextLoader:
    """
    A class to handle loading and processing of text data from CSV files.
    """

    def __init__(self, file_path: Optional[Path] = None) -> None:
        """
        Initializes the TextLoader with a CSV file path.

        Args:
            file_path (Optional[Path]): The path to the CSV file containing the text data.
                                        If None, the default file path is used.
        """
        self.file_path = file_path or DATA_FILE

    def get_texts(self, sampling: bool | float = False) -> list[str]:
        """
        Reads the text data from the CSV file.

        Args:
            sampling (bool): If True, only use a 10% sample of the text data.

        Returns:
            list[str]: A list of strings containing all the text from the CSV file.
        """
        text_data = pd.read_csv(self.file_path, usecols=['texto'])
        if sampling:
            sampling = sampling if isinstance(sampling, (float, int)) else 0.1
            sample_size = int(len(text_data) * sampling)
            text_data = text_data[:sample_size]
        return text_data['texto'].tolist()

    @staticmethod
    def clean_texts(text_sequences: list[str], remove_stop_words: bool = False, min_words: int = 25) -> list[str]:
        """
        Cleans the input text by converting to lowercase, removing accents and punctuation (except dots).

        Args:
            text_sequences (list[str]): The input texts to be cleaned.
            remove_stop_words (bool): Flag to remove or not stop_words
            min_words (int): Minimum number of words per text

        Returns:
            list[str]: The cleaned texts list.
        """
        # Convert to lowercase
        for index in range(len(text_sequences)):
            text = text_sequences[index]
            text = text.lower()

            # Remove accents
            text = unicodedata.normalize('NFD', text).encode('ASCII', 'ignore').decode('ASCII')

            # Remove punctuation and numbers
            ignore = (string.punctuation + string.digits).replace('.', '')
            text = ''.join([t for t in text if t not in ignore])
            text_list = text.split('\n')

            if remove_stop_words:
                # remove stop words
                stop_words = nltk.corpus.stopwords.words('portuguese')
                for i, t in enumerate(text_list):
                    text_list[i] = ' '.join([w for w in t.split(" ") if w not in stop_words])
            text_sequences[index] = ' '.join(text_list).strip()

        text_sequences = '.'.join(text_sequences).split('.')
        text_sequences = [t.strip() for t in text_sequences if len(t.split()) > min_words]
        return text_sequences
