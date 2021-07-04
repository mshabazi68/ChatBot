import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words=[]

classes = []

documents = []

ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:

    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)

        words.extend(word)        
        #add documents in the corpus

        documents.append((word, intent['tag']))

        # add to our classes list

        if intent['tag'] not in classes:

            classes.append(intent['tag'])

print(documents)