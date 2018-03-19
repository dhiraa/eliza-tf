import json
import random
import numpy as np
from overrides import overrides
from dataset.dataset_interface import *
from utils.nlp import *


class Intents(DatasetInterface):
    def __init__(self, path='data/intents.json'):
        self.path = path

        self.classes = []
        self.words = []
        self.documents = []
        self.ignore_words = []

        self.intents = None

    @overrides
    def get_training_data(self):
        # load json file
        with open(self.path) as json_data:
            self.intents = json.load(json_data)

        # loop through each sentense in our intents patterns
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our word list
                self.words.extend(w)
                # add to documents in our corpus
                self.documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # stem and lower each word and remove duplicates
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))

        # remove duplicates
        self.classes = sorted(list(set(self.classes)))

        # create training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        pickle.dump({
            'words': self.words,
            'classes': self.classes,
            'train_x': train_x,
            'train_y': train_y
            }, open('training_data', 'wb'))

        return train_x, train_y