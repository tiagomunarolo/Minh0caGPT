import tiktoken
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.api.utils import pad_sequences

# Constants for tokenizer types and special tokens
TOKENIZER_TYPES = ('keras', 'tiktoken')
SPECIAL_TOKENS = {
    'pad': '<pad>',
    'unk': '<unk>',
    'bos': '<bos>',
    'eos': '<eos>'
}


class CustomTokenizer:
    def __init__(self, tokenizer_type: str = 'keras', tiktoken_model: str = 'gpt2'):
        if tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")

        self._tokenizer_type = tokenizer_type
        self._tiktoken_model = tiktoken_model
        self._tokenizer = None
        self.vocab_size = None
        self.pad_token = None
        self.bos_token = None
        self.eos_token = None

    def _setup_special_tokens(self):
        """
        Sets up special tokens for the tokenizer.
        """
        if self._tokenizer_type == 'keras':
            self._tokenizer.index_word[0] = SPECIAL_TOKENS['pad']
            self._tokenizer.word_index[SPECIAL_TOKENS['pad']] = 0
            self.bos_token = self._tokenizer.word_index.get(SPECIAL_TOKENS['bos'], len(self._tokenizer.word_index) + 1)
            self.eos_token = self._tokenizer.word_index.get(SPECIAL_TOKENS['eos'], len(self._tokenizer.word_index) + 2)
            self.pad_token = self._tokenizer.word_index[SPECIAL_TOKENS['pad']]
        else:
            self.bos_token = self.vocab_size + 1
            self.eos_token = self.vocab_size + 2
            self.pad_token = self.vocab_size + 3
            self.vocab_size += len(SPECIAL_TOKENS)

    def fit(self, texts: list[str]) -> None:
        """
        Fits the tokenizer to a list of input text strings.

        Args:
            texts (list[str]): List of input text strings.

        Returns:
            None
        """
        if not texts:
            raise ValueError("Input text list is empty.")

        if self._tokenizer_type == 'keras':
            filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
            self._tokenizer = Tokenizer(filters=filters)
            self._tokenizer.fit_on_texts(texts)
            self.vocab_size = len(self._tokenizer.word_index) + 1
        else:
            self._tokenizer = tiktoken.get_encoding(self._tiktoken_model)
            self.vocab_size = self._tokenizer.n_vocab

        self._setup_special_tokens()

    def encode(self, text: str, padding: bool = False, padding_length: int = 0) -> list[int]:
        """
        Encodes a text string into a sequence of tokens.

        Args:
            text (str): Input text string.
            padding (bool): Flag to pad the sequence to a specific length.
            padding_length (int): Length to pad the sequence to.

        Returns:
            list[int]: List of token indices.
        """
        if not text:
            raise ValueError("Input text cannot be empty.")
        if padding and padding_length <= 0:
            raise ValueError("Padding length must be greater than zero.")

        sequence = []
        if self._tokenizer_type == 'keras':
            sequence = self._tokenizer.texts_to_sequences([text])[0]
        else:
            split_text = text.split()
            sequence = ([self.bos_token] if split_text[0] == SPECIAL_TOKENS['bos'] else []) + \
                       self._tokenizer.encode(' '.join(split_text[
                                                       1 if split_text[0] == SPECIAL_TOKENS['bos'] else 0:-1 if
                                                       split_text[-1] == SPECIAL_TOKENS['eos'] else None])) + \
                       ([self.eos_token] if split_text[-1] == SPECIAL_TOKENS['eos'] else [])

        if padding:
            sequence = pad_sequences([sequence], maxlen=padding_length, padding='post', value=self.pad_token)[0]
        return sequence

    def decode(self, sequence: list[int]) -> str:
        """
        Decodes a sequence of tokens into a text string.

        Args:
            sequence (list[int]): List of token indices.

        Returns:
            str: Decoded text string.
        """
        decoded_tokens = []
        for token in sequence:
            if token == self.pad_token:
                continue
            if token == self.bos_token:
                decoded_tokens.append(SPECIAL_TOKENS['bos'])
            elif token == self.eos_token:
                decoded_tokens.append(SPECIAL_TOKENS['eos'])
            else:
                token_str = self._tokenizer.decode(
                    [token]) if self._tokenizer_type == 'tiktoken' else self._tokenizer.index_word.get(token,
                                                                                                       SPECIAL_TOKENS[
                                                                                                           'unk'])
                decoded_tokens.append(token_str)

        return ' '.join(decoded_tokens) if self._tokenizer_type == 'keras' else ''.join(decoded_tokens)
