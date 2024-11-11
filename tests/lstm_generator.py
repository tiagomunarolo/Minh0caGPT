import numpy as np
from gensim.models import Word2Vec
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.api.preprocessing.sequence import pad_sequences
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense
from keras.api.optimizers import Adam

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models improve with more data",
    "Python programming is fun and highly versatile",
    "Transformers revolutionized the field of NLP tasks",
    "Artificial intelligence continues to evolve rapidly"
]
corpus = [line.lower() for line in corpus if line]  # Remove empty sequences

# 1. Train Word2Vec model on the corpus
embedding_dim = 100  # Size of word vectors
sentences = [line.split() for line in corpus]
word2vec = Word2Vec(sentences=sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# 2. Tokenize the text using Keras' Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1  # Plus one for padding

# 3. Create the embedding matrix using Word2Vec vectors
embedding_matrix = np.zeros((total_words, embedding_dim))
for word, index in word_index.items():
    if word in word2vec.wv:
        embedding_matrix[index] = word2vec.wv[word]

# 4. Prepare input sequences for LSTM model
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform length
input_sequences = pad_sequences(input_sequences, padding='pre')

# Split data into predictors (X) and labels (y)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode the labels (for softmax output layer)
y = to_categorical(y, num_classes=total_words)

# 5. Build the LSTM model using pre-trained Word2Vec embeddings
model = Sequential()
model.add(Embedding(total_words, embedding_dim, input_length=X.shape[1],
                    weights=[embedding_matrix], trainable=False))  # Word2Vec embeddings
model.add(LSTM(150))  # LSTM with 150 units
model.add(Dense(total_words, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Print model summary
model.summary()

# 6. Train the LSTM model
epochs = 500  # Number of epochs for training
history = model.fit(X, y, epochs=epochs, verbose=1)


# 7. Function to generate text using the trained LSTM model
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# 8. Test text generation
seed_text = "machine"
next_words = 10
print(generate_text(seed_text, next_words, X.shape[1]))
