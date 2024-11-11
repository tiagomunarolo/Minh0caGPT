def get_mock_data():
    """
    Returns a list of mock text sequences.
    """
    text_sequences = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models improve with more data",
        "Python programming is fun and highly versatile",
        "Transformers revolutionized the field of NLP tasks",
        "Artificial intelligence continues to evolve rapidly"
    ]
    text_sequences = [f'<bos> {text} <eos>'.lower() for text in text_sequences]
    return text_sequences
