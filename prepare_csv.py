from prepare_doc import read_json
import csv

with open('Human_Evaluation_Summarization.csv', mode='w') as outfile:
    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Dataset', 'Text', 'H1', 'H2',
                     'Similarity/Relevance (H1, H2)',
                     'Similarity/Relevance (Text, H1)',
                     'Similarity/Relevance (Text, H2)'])


def write_csv(reviews, summaries, generated, dataset):
    with open('Human_Evaluation_Summarization.csv', mode='a') as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for review, summary, generated in zip(reviews, summaries, generated):
            writer.writerow([dataset, review, summary, generated])


datasets = {'amazon': 'samples-amazon.json',
            'cnn-dailymail': 'samples_cnn-dm.json',
            'newsroom': 'samples-newsroom.json',
            'gigaword': 'samples-gigaword.json'}
for name, filename in datasets.items():
    reviews, summaries, generated = read_json(filename)
    write_csv(reviews, summaries, generated, name)
