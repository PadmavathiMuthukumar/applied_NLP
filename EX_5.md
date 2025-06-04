# EXPERIMENT 5
## NAME: PADMAVATHI M
## REGISTER NUMBER: 212223040141

## AIM:
To preprocess a given text corpus, train a Word2Vec model on the processed data, and find the embedding of a specific word. Additionally, identify the two words most similar to the chosen word using the trained model along with their similarity scores.

## PROCEDURE:
1. Collect and Load Text Corpus
2. Text Preprocessing
3. Train Word2Vec Model
4. Find Embedding Vector for a Word
5. Find Most Similar Words

## PROGRAM:
```
# Import necessary libraries
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample corpus
corpus = [
    "Natural language processing is a field of artificial intelligence.",
    "It enables computers to understand human language.",
    "Word embedding is a representation of words in a dense vector space.",
    "Gensim is a library for training word embeddings in Python.",
    "Machine learning and deep learning techniques are widely used in NLP."
]

# Preprocess the text: Tokenize, remove punctuation and stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

# Apply preprocessing to the corpus
processed_corpus = [preprocess_text(sentence) for sentence in corpus]

# Train a Word2Vec model
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=2, min_count=1, sg=1)  # sg=1 uses Skip-gram

# Save the model for future use
model.save("word2vec_model.model")

# Test the model by finding the embedding of a word
word = "vector"
if word in model.wv:
    print(f"Embedding for '{word}':\n{model.wv[word]}")
else:
    print(f"'{word}' not found in vocabulary.")

# Find similar words
similar_words = model.wv.most_similar(word, topn=2)
print(f"Words similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")

```

## OUTPUT:
![image](https://github.com/user-attachments/assets/3c04aec0-24f8-4c98-bcb2-e8a3b95f83ab)

## RESULT:
These results show that the model has successfully learned semantic relationships in the small training corpus:
