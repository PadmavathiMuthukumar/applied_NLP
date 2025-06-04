# EXPERIMENT 3
## NAME: PADMAVATHI M
## REGISTER NUMBER: 212223040141
## AIM:
To design and implement a Siamese Neural Network using LSTM to detect sentence similarity. The model uses cosine similarity to compute how similar two input sentences are and is trained on labeled sentence pairs.

## PROCEDURE:
1. Import Required Libraries
2. Sample Dataset of Sentence Pairs
3. Preprocess and Vectorize Sentences
4.  Build the Siamese Network Using LSTM
5.  Compile and Train the Model
6.  Test on New Sentence Pairs

## PROGRAM:
```


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# Sentence pairs (labels: 1 for similar, 0 for dissimilar)
sentence_pairs = [
    ("How are you?", "How do you do?", 1),
    ("How are you?", "What is your name?", 0),
    ("What time is it?", "Can you tell me the time?", 1),
    ("What is your name?", "Tell me the time?", 0),
    ("Hello there!", "Hi!", 1),
]

# Separate into two sets of sentences and their labels
sentences1 = [pair[0] for pair in sentence_pairs]
sentences2 = [pair[1] for pair in sentence_pairs]
labels = np.array([pair[2] for pair in sentence_pairs])

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences1 + sentences2)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
max_len = 100  # Max sequence length
X1 = pad_sequences(tokenizer.texts_to_sequences(sentences1), maxlen=max_len)
X2 = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=max_len)

# Input layers for two sentences
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len,))

# Embedding layer
embedding_dim = 1000
embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)

# Shared LSTM layer
shared_lstm = Bidirectional(LSTM(512))

# Process the two inputs using the shared LSTM
encoded_1 = shared_lstm(embedding(input_1))
encoded_2 = shared_lstm(embedding(input_2))

# Calculate the L1 distance between the two encoded sentences
def l1_distance(vectors):
    x, y = vectors
    return K.abs(x - y)

l1_layer = Lambda(l1_distance)
l1_distance_output = l1_layer([encoded_1, encoded_2])

# Add a dense layer for classification (similar/dissimilar)
output = Dense(1, activation='sigmoid')(l1_distance_output)

# Create the Siamese network model
siamese_network = Model([input_1, input_2], output)
siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
siamese_network.summary()

# Train the model
siamese_network.fit([X1, X2], labels, epochs=12, batch_size=2)

# Test with a new sentence pair
test_sentences1 = ["How are you?"]
test_sentences2 = ["How do you do?"]

test_X1 = pad_sequences(tokenizer.texts_to_sequences(test_sentences1), maxlen=max_len)
test_X2 = pad_sequences(tokenizer.texts_to_sequences(test_sentences2), maxlen=max_len)

# Predict similarity
similarity = siamese_network.predict([test_X1, test_X2])
print(f"Similarity Score: {similarity[0][0]}")

```
## OUTPUT:
![image](https://github.com/user-attachments/assets/00cc745f-29f7-4034-9558-f59fd5b67f72)

![image](https://github.com/user-attachments/assets/2306b446-6e8e-4fd8-a1c5-6fd25280426c)

## RESULT:
The Siamese LSTM network was successfully trained on sentence pairs.

