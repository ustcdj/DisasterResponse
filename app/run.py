from app import app

import json
import plotly
import pandas as pd
import pickle

# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
# from sklearn.externals import joblib
from sqlalchemy import create_engine

import random
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

import utility
from utility import tokenize, StartingVerbExtractor, TextLenghExtractor

app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # data for graph 1
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)

    # data for graph 2
    category_counts = df.drop('id', axis=1).sum(axis=0, numeric_only=True).sort_values(ascending=False)
    category_names = list(category_counts.index)

    # data for graph 3 (wordcloud)
    text = df['message']
    text = " ".join(message for message in df['message'])
    wordcloud = WordCloud(max_words=50, stopwords=set(stopwords.words('english')), background_color='white', max_font_size=50).generate(text)
    df_wordcloud = pd.DataFrame.from_dict(wordcloud.words_, orient='index', columns=['count'])
    random.seed(5)
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(50)]
    weights = df_wordcloud['count'] * 70

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
                # graph 2
                {'data': [Bar(x=category_names, y=category_counts)],

                 'layout': {'title': 'Counts of Message Category (sorted)',
                            'yaxis': {'title': "Count"},
                            'xaxis': {'title': "Category"}
                            }
                },

                # graph 1
                {'data': [Bar(x=genre_names, y=genre_counts)],

                 'layout': {'title': 'Distribution of Message Genres',
                            'yaxis': {'title': "Count"},
                            'xaxis': {'title': "Genre"}
                            }
                },

                # graph 3
                {'data': [Scatter(x=[random.random() for i in range(50)],
                                  y=[random.random() for i in range(50)],
                            #      y=[random.choices(range(50), k=50)],
                                  mode='text',
                                  text=df_wordcloud.index,
                                  marker={'opacity': 0.3},
                                  textfont={'size': weights,
                                            'color': colors})],

                 'layout': {'title': 'Top 50 Words in All Messages',
                            'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}
                            }
                }
             ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

'''
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
'''
