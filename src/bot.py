import nltk
import sys
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

import pickle
import json
import numpy as np

from utils.nlp import *
from utils.print_helpers import *

class ContextualBot:
    def __init__(self, dataset, model):
        self.model = model

        self.classes = dataset.classes
        self.words = dataset.words
        self.documents = dataset.documents
        self.ignore_words = dataset.ignore_words

        self.intents = dataset.intents

        self.context = {}
    #
    # def classify(self, sentence):
    #     # generate probabilities from the model
    #     results = self.model.predict([bow(sentence, self.words)])[0]
    #     # filter out predictions below a threshold
    #     results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    #     # sort by strength of probability
    #     results.sort(key=lambda x: x[1], reverse=True)
    #     return_list = []
    #     for r in results:
    #         return_list.append((self.classes[r[0]], r[1]))
    #     # return tuple of intent and probability
    #     return return_list
    #
    # def response(self, sentence, userID='123', show_details=False):
    #     results = self.classify(sentence)
    #     # if we have a classification then find the matching intent tag
    #     if results:
    #         # loop as long as there are matches to process
    #         while results:
    #             for i in self.intents['intents']:
    #                 # find a tag matching the first results
    #                 if i['tag'] == results[0][0]:
    #                     # a random response from the intent
    #                     return random.choice(i['responses'])
    #
    #             results.pop(0)

    ERROR_THRESHOLD = 0.25

    def classify(self, sentence):
        # generate probabilities from the model
        results = self.model.predict([bow(sentence, self.words)])
        # filter out predictions below a threshold
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        print_debug(return_list)
        # return tuple of intent and probability
        return return_list

    def response(self, sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print('context:', i['context_set'])
                            self.context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                                (userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[
                                    userID]):
                            if show_details: print('tag:', i['tag'])
                            # a random response from the intent
                            return random.choice(i['responses'])

                results.pop(0)

    def start(self):
        while(True):
            user = input(">")
            print_info(self.response(user))
            if user == "bye":
                sys.exit(0)

# ===
# data = pickle.load(open("training_data", "rb"))
# words = data['words']
# classes = data['classes']
# train_x = data['train_x']
# train_y = data['train_y']
#
# # import our chat-bot intents file
# with open('intents.json') as json_data:
#     intents = json.load(json_data)
#
# # load saved model
# net = tflearn.input_data(shape=[None, len(train_x[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# net = tflearn.regression(net)
# model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# model.load('./model.tflearn')
#
# print (classify('is your shop open today?'))
# print (classify('are you open today?'))
# print (classify('do you take cash?'))
# print (classify('what kind of mopeds do you rent?'))
# print (classify('Goodbye, see you later'))
