# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import pickle
import markovify
import nltk
import re
import json
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

df = pd.read_csv('headlines.csv')
with open('model_data.json') as json_file:
    model_json = json.load(json_file)

reconstituted_model = markovify.NewlineText.from_json(model_json)

app = dash.Dash(__name__)
app.title = 'Goody Feed Headline Generator'
server = app.server
app.layout = html.Div([
    html.Link(rel="stylesheet", href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css"),
    html.Div(html.H1("GoodyFeed Title Generator")),
    html.Div(id = 'output', children = 'Generate a headline'),
    html.Div(html.Button('Generate', id='generate', n_clicks=0),id="button-container"),
    html.Div(html.A(html.I(className="fa fa-github"), href = "https://github.com/xghan99/GFeedNLG", id="github"))
])

@app.callback(
    Output('output','children'),
    Input('generate', 'n_clicks'),
)

def generate_text(n_clicks):
    status = True
    while status:
        text = reconstituted_model.make_sentence(max_overlap_ratio = 0.5)
        if text:
            result = [i.split("::")[0] for i in text.split(" ")]
            title = " ".join(result)
            if title not in df["title"]:
                return title
            else:
                continue
        else:
            continue

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
