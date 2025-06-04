# EXPERIMENT 6
## NAME: PADMAVATHI M
## REGISTER NUMBER: 212223040141

## AIM:
To implement an Auto-Correction system that suggests the most likely correct word for a misspelled word by calculating the edit distance (Levenshtein Distance) from a set of valid dictionary words.

## PROCEDURE:
1. Import Required Libraries
2. Define the Edit Distance Function
3. Build or Load a Dictionary
4. Auto-Correction Logic
5. Test the Auto-Corrector

## PROGRAM:
```
# Install NLTK if needed
!pip install nltk
import nltk
nltk.download('words')
from nltk.corpus import words
import re
from collections import Counter
# Use the NLTK words corpus as our vocabulary
word_list = words.words()
word_freq = Counter(word_list)  # Count frequencies, though here it's a simple corpus with each word appearing once

# Define a set of all known words
WORD_SET = set(word_list)
# Define a function to calculate minimum edit distance
def edit_distance(word1, word2):
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                dp[i][j] = j  # Cost of insertions
            elif j == 0:
                dp[i][j] = i  # Cost of deletions
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change cost
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                   dp[i][j - 1],      # Insertion
                                   dp[i - 1][j - 1])  # Substitution
    return dp[-1][-1]

# Define a function to calculate word probability
def word_probability(word, N=sum(word_freq.values())):
    return word_freq[word] / N if word in word_freq else 0
# Suggest corrections based on edit distance and probability
def autocorrect(word):
    # If the word is correct, return it as is
    if word in WORD_SET:
        return word

    # Find candidate words within an edit distance of 1 or 2
    candidates = [w for w in WORD_SET if edit_distance(word, w) <= 2]

    # Choose the candidate with the highest probability
    corrected_word = max(candidates, key=word_probability, default=word)

    return corrected_word
# Test the function with common misspellings
test_words = ["speling", "korrect", "exampl", "wrld"]

for word in test_words:
    print(f"Original: {word} -> Suggested: {autocorrect(word)}")
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/32ade7af-3bbe-4ade-9ca7-f3b1b8fca370)

![image](https://github.com/user-attachments/assets/0dcfc975-94dd-4d5e-89b4-c6fa6148465c)

![image](https://github.com/user-attachments/assets/e29e96d0-280d-4728-9352-ab04df8bbea6)

## RESULT:
These results show that the auto-correct system can accurately suggest the correct spelling for common misspellings using edit distance.
