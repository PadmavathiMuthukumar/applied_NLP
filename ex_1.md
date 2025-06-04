# EXPERIMENT 1
## NAME: PADMAVATHI M
## REGISTER NUMBER: 212223040141
# AIM:
To apply the Naive Bayes algorithm to perform sentiment analysis on a customized dataset containing labeled text data (positive and negative sentiments).

# PROCEDURE:
1. Import Required Libraries
2. Create or Load the Customized Dataset
3. Preprocessing the Data
4. Vectorize the Text Data
5. Train the Naive Bayes Model
6. Make Predictions and Evaluate

# PROGRAM:
```
# Import necessary libraries
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download NLTK data (stopwords)
nltk.download('stopwords')

# Sample dataset of sentences with their sentiments (1 = positive, 0 = negative)
data = [
    ("I love this product, it works great!", 1),
    ("This is the best purchase I have ever made.", 1),
    ("Absolutely fantastic service and amazing quality!", 1),
    ("I am very happy with my order, will buy again.", 1),
    ("This is a horrible experience.", 0),
    ("I hate this so much, it broke on the first day.", 0),
    ("Worst product I have ever used, total waste of money.", 0),
    ("I am disappointed with this product, it didn't work as expected.", 0)
]

# Separate sentences and labels
sentences = [pair[0] for pair in data]
labels = np.array([pair[1] for pair in data])

# Split dataset into training and testing sets
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=42)

# Text Preprocessing
# Tokenization, removing stopwords and converting text into numerical data using CountVectorizer

# Instead of using a set, use a list for stop_words
stop_words = stopwords.words('english') # Changed this line to create a list

# Initialize CountVectorizer (this will convert text into a bag-of-words representation)
vectorizer = CountVectorizer(stop_words=stop_words)

# Fit the vectorizer on the training data and transform both training and test sets
X_train = vectorizer.fit_transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

# Initialize the Naive Bayes Classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict sentiments for the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Test the model with new sentences
test_sentences = ["I am happy to comment!", "This is a terrible product."]
test_X = vectorizer.transform(test_sentences)

# Predict sentiments for new sentences
predictions = nb_classifier.predict(test_X)

# Output predictions
for sentence, sentiment in zip(test_sentences, predictions):
    print(f"Sentence: '{sentence}' => Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/f938b77f-dd79-4f72-8e9d-0d991bc2dc2d)

## RESULT:
The Naive Bayes model was successfully applied to a customized dataset for sentiment analysis.


