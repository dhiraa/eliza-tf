from flask import Flask, render_template, request, jsonify
import os
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import pickle

app = Flask(__name__)

#Setup chatten bot for first time run
english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
english_bot.set_trainer(ChatterBotCorpusTrainer)

if not os.path.exists("_TRAINED"):
	english_bot.train("chatterbot.corpus.english")
	open("_TRAINED", "wb")


@app.route("/")
def hello():
	return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
	message = str(request.form['messageText'])

	# kernel now ready for use
	while True:
		if message == "quit":
			exit()
		else:
			bot_response =  "eliza >>> " + str(english_bot.get_response(message))
			print(bot_response)
			return jsonify({'status':'OK','answer':bot_response})

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
