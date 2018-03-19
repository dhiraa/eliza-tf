import os


# TensorFlow stuff
import numpy as np
import tflearn
import tensorflow as tf

from dataset.dataset_interface import DatasetInterface
from utils.print_helpers import *

class DNN:
    def __init__(self, dataset):
        self.dataset: DatasetInterface = dataset
        self.model = None

        self.train_x, self.train_y = self.dataset.get_training_data()

        self.model_fn()

        if not os.path.exists("model"):
            os.makedirs("model")

    def model_fn(self):
        # reset underlying graph data
        tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, len(self.train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

        print_info("Checking for model.tflearn...")
        if os.path.exists('model/model.tflearn.meta'):
            print_info("Loading the model...")
            self.model.load('model/model.tflearn')


    def train(self):
        # Start training (apply gradient descent algorithm)
        self.model.fit(self.train_x,
                       self.train_y,
                       n_epoch=1000,
                       batch_size=8,
                       show_metric=True)
        self.model.save('model/model.tflearn')


    def predict(self, data):
        return self.model.predict(data)[0]
