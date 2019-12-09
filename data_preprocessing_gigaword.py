import pandas as pd
from pickle import load


def make_df():
    f_titles = open('D:/train/train.title.txt', 'r')
    f_articles = open('D:/train/train.article.txt', 'r')

    titles = f_titles.read()
    articles = f_articles.read()

    titles = titles.split('\n')
    articles = articles.split('\n')

    stories = []
    for title, article in zip(titles, articles):
        stories.append({'headline': title, 'story': article})

    df = pd.DataFrame(stories)
    clipped = df.iloc[:90000]
    clipped.to_pickle('../data/gigaword_formatted.pkl')


def load_formatted_data():
    filename = '../data/gigaword_formatted.pkl'
    stories = load(open(filename, 'rb'))
    print(stories.head(), stories.shape)


if __name__ == '__main__':
    # make_df()
    load_formatted_data()
