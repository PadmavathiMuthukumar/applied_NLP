# EXPERIMENT - 4
## NAME: PADMAVATHI M
## REGISTER NUMBER: 212223040141

## AIM:
To create a simple language translation application that translates English sentences into French using a pre-trained Transformer model (Helsinki-NLP/opus-mt-en-fr).

## PROCEDURE:
1. Install Required Libraries
2. Import Required Modules
3. Load the Pre-trained Model and Tokenizer
4. Create a Translation Function
5. Test with Sample Sentences

## PROGRAM:
```
!pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer for English-to-French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
import torch

def translate_text(text: str, max_length: int = 40) -> str:
    # Tokenize the input text and convert to input IDs
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate translation using the model
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated IDs back to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text
# Sample sentences in English
english_sentences = [
    "Hello, how are you?",
    "This is an experiment in machine translation.",
    "Transformers are powerful models for natural language processing tasks.",
    "Can you help me with my homework?",
    "I love learning new languages."
]

# Translate each sentence
for sentence in english_sentences:
    translation = translate_text(sentence)
    print(f"Original: {sentence}")
    print(f"Translated: {translation}\n")
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/c7c08a09-87db-4bd4-a43d-7492f97ff341)

![image](https://github.com/user-attachments/assets/7f20731d-22b6-4136-9cc9-7db00be18c5d)

![image](https://github.com/user-attachments/assets/f60e113e-16a3-4478-a89b-a9772649436c)

## RESULT:
The model successfully translated English sentences into French using a pre-trained Transformer model (Helsinki-NLP/opus-mt-en-fr). The output is accurate and demonstrates the power of transfer learning in NLP.


