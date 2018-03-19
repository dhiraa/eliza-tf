import pickle

class DatasetInterface:
    def __init__(self):
        ""

    def get_training_data(self):
        raise NotImplementedError

    def get_testing_data(self):
        raise NotImplementedError

    def get_validation_data(self):
        raise NotImplementedError