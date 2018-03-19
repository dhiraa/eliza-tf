import argparse
import sys
import pickle
sys.path.append("src/")
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

from bot import ContextualBot
from dataset.intents import Intents
from model.dnn import DNN

def run(opt):
    dataset = Intents(path="data/intents.json")
    model = DNN(dataset)
    bot = ContextualBot(dataset, model)

    if opt.mode == "train":
        model.train()
    elif opt.mode == "chat":
        bot.start()
    elif opt.mode =="chatterbot":
        english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
        english_bot.set_trainer(ChatterBotCorpusTrainer)

        english_bot.train("chatterbot.corpus.english")
        pickle.dump(file=open("bot_brain.eliza", "wb"), obj=english_bot)


if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Run experiments on available models and datasets")


    optparse.add_argument('-mode', '--mode',
                          choices=['train', 'chat', 'chatterbot'],
                          required=True,
                          help="'train', 'chat', 'chatterbot'"
                          )

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Model directory needed for training')

    optparse.add_argument('-dsn', '--dataset-name', action='store',
                          dest='dataset_name', required=False,
                          help='Name of the Dataset to be used')

    optparse.add_argument('-din', '--data-iterator-name', action='store',
                          dest='data_iterator_name', required=False,
                          help='Name of the DataIterator to be used')

    optparse.add_argument('-bs', '--batch-size',  type=int, action='store',
                          dest='batch_size', required=False,
                          default=1,
                          help='Batch size for training, be consistent when retraining')

    optparse.add_argument('-ne', '--num-epochs', type=int, action='store',
                          dest='num_epochs', required=False,
                          help='Number of epochs')

    optparse.add_argument('-mn', '--model-name', action='store',
                          dest='model_name', required=False,
                          help='Name of the Model to be used')

    opt = optparse.parse_args()
    if (opt.mode == 'retrain' or opt.mode == 'predict') and not opt.model_dir:
        optparse.error('--model-dir argument is required in "retrain" & "predict" mode.')
    else:
        run(opt)