import json_lines
import pandas as pd
from pickle import load
import re


def load_formatted_data():
    filename = '../data/newsroom_formatted.pkl'
    stories = load(open(filename, 'rb'))
    print(stories.head(), stories.shape)


def save_as_df(stories, file_path):
    df = pd.DataFrame(stories)
    clipped = df.iloc[:90000]
    clipped.to_pickle(file_path)


def read_raw_data(file_path):
    stories = []
    with open(file_path, 'rb') as f:
        for item in json_lines.reader(f):
            text = ''
            clean_story = ''
            for k in text.join(item['text'].replace('\n', ' ').split('.')[:5]).split("\n"):
                clean_story += re.sub(r"[^a-zA-Z0-9]+", ' ', k)
            stories.append({'headline': item['title'],
                            'story': clean_story})
    return stories


# read newsroom-dev data (first 5 lines of article)
stories = read_raw_data('D:\dev-stats.jsonl')

# make data frame from stories and save
save_as_df(stories, '../data/newsroom_formatted.pkl')

# read pickle file to check if saved correctly
load_formatted_data()