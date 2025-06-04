# EXPERIMENT 2
## NAME: PADMAVATHI M
## REGISTER NUMBER: 212223040141

## AIM:
To develop a basic machine translation system using the K-Nearest Neighbors (KNN) algorithm that translates English sentences into French by finding the most similar sentences from a bilingual dictionary using cosine similarity.

## PROCEDURE:
1. Import Required Libraries
2. Create a Small Bilingual Dictionary
3. Prepare the Data
4. Vectorize the Sentences Using TF-IDF
5.  Define a Function for Translation Using KNN
6.  Test the Translation System

## PROGRAM:
```
!pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Sample bilingual dictionary
english_sentences = [
    "hello", "how are you", "good morning", "good night", "thank you",
    "see you later", "what is your name", "my name is John", "where is the library",
    "I like to read books"
]

french_sentences = [
    "bonjour", "comment ça va", "bonjour", "bonne nuit", "merci",
    "à plus tard", "quel est ton nom", "mon nom est John", "où est la bibliothèque",
    "j'aime lire des livres"
]
vectorizer = TfidfVectorizer()
english_vectors = vectorizer.fit_transform(english_sentences)
def knn_translate(input_sentence, k=1):
    input_vector = vectorizer.transform([input_sentence])

    # Compute cosine similarity between the input sentence and all sentences in the dictionary
    similarities = cosine_similarity(input_vector, english_vectors).flatten()

    # Get indices of the top-k similar sentences
    top_k_indices = similarities.argsort()[-k:][::-1]

    # Retrieve and display the French translations for the most similar sentences
    translations = [french_sentences[i] for i in top_k_indices]
    return translations
# Test sentences
test_sentences = ["good evening", "where is the library", "thank you very much"]

# Translate each test sentence
for sentence in test_sentences:
    translations = knn_translate(sentence, k=1)  # Use k=1 for the closest translation
    print(f"English: {sentence} -> French: {translations[0]}")
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/2c1c7c05-9827-4c48-bcb1-bc6098d59dd3)

![image](https://github.com/user-attachments/assets/3c46c207-a257-4c09-8727-90668037c49f)


## RESULT:
The system successfully uses the KNN algorithm with cosine similarity to find the closest matching English sentence and returns the corresponding French translation.

