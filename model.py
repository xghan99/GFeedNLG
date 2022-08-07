import pandas as pd
import markovify
import re
import nltk
import json
import pickle

nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('headlines.csv')


class POSifiedText(markovify.NewlineText):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = ["::".join(tag) for tag in nltk.pos_tag(words)]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence


text_model = POSifiedText(df['title'], state_size=3)

model_json = text_model.to_json()
with open('model_data.json', 'w') as outfile:
    json.dump(model_json, outfile)
