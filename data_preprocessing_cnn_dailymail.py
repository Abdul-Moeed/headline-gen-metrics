# REFERENCE: https://machinelearningmastery.com/prepare-news-articles-text-summarization/

from os import listdir
from pickle import dump, load
import fire
import pandas as pd


def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, headlines = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    headlines = [h.strip() for h in headlines if len(h) > 0]
    return story, headlines


# load all stories in a directory
def load_stories(directory):
    stories = list()
    count = 0
    max_count = 92000
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, headlines = split_story(doc)
        # save only first highlight
        headline = headlines[0]
        # store
        stories.append({'story': story, 'headline': headline})

        if count == max_count:
            break
        count += 1
    return stories


# clean a list of lines
def clean_lines(lines):
    max_lines = 5
    count = 0
    cleaned = ''
    skip_line = False
    skip_next_line = False
    skip_triggers = ['By', 'PUBLISHED:', 'UPDATED:']

    for idx, line in enumerate(lines):
        if len(line) == 0 or line.strip() == '|':
            continue

        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index + len('(CNN) -- '):]

        # strip author names, edit dates if found (for DailyMail)
        if line.strip() in skip_triggers:
            skip_line = True
            skip_next_line = True

        if skip_line:
            skip_line = False
        elif skip_next_line:
            skip_next_line = False
        else:
            cleaned += line + ' '
            if count > max_lines:
                break
            else:
                count += 1
    return cleaned


# print stories
def print_stories(stories):
    for example in stories:
        print(example['headline'])
        print(example['story'])
        print('===============================')


def parse(path=None, save_as=None):
    if not path or not save_as:
        return

    # load stories
    directory = path
    stories = load_stories(directory)

    # clean stories, pick first 5 lines of each
    for idx, example in enumerate(stories):
        stories[idx]['story'] = clean_lines(example['story'].split('\n'))

    # save stories
    filename = '../data/' + save_as + '_dataset.p'
    dump(stories, open(filename, 'wb'))


def make_df(dataset_name=None):
    filename = '../data/'+dataset_name+'_dataset.p'
    stories = load(open(filename, 'rb'))
    df = pd.DataFrame(stories)
    df.to_pickle('../data/'+dataset_name+'_formatted.pkl')


def load_formatted_data(dataset_name=None):
    filename = '../data/' + dataset_name + '_formatted.pkl'
    stories = load(open(filename, 'rb'))
    print(stories.head())


"""
    Usage: <method> <args> e.g. make_df cnn
    Methods:
        - parse: Parse raw data and save in pickle file as cnn_dataset.p
        - make_df: Create df from the parsed file
        - load_formatted_data: Load formatted df
"""
if __name__ == '__main__':
    fire.Fire()
