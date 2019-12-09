import pandas as pd
import os
import json
from scipy.stats import pearsonr

human_eval_path = 'data/human-eval'
aes_path = 'data/angular.json'
rouge_path = 'data/rouge.json'
dataset_mask = {"Amazon": 0, "CNN_DM": 1, "Newsroom": 2, "Gigaword": 3}
col_mask = {"gen_text": 6, "gt_text": 5, "gen_gt": 4}


def load_human_evals(path):
    human_evals = []
    for file in os.listdir(path):
        df = pd.read_excel(os.path.join(path, file))
        df = df.drop(df.index[0])
        human_evals.append(df)

    return human_evals


def load_json_data(path):
    with open(path, 'r') as read_file:
        data = json.load(read_file)

    return data


def calculate_human_average(human_evals, col):
    result = []
    for row in range(human_evals[0].shape[0]):
        sum = 0
        for df in human_evals:
            sum += df.iat[row, col]
        result.append(sum / 5)

    return result


def analyse(rouge, human_evals, col):
    human_avg = calculate_human_average(human_evals, col_mask[col])

    # hypothesis testing
    corr, p_val = pearsonr(human_avg, rouge)

    return round(corr, 4), p_val

def analyse_single(rouge, human_evals, col, dataset, dataset_len):
    human_eval_avg = []
    for row in range(dataset_mask[dataset] * dataset_len, dataset_mask[dataset] * dataset_len + dataset_len):
        sum = 0
        for df in human_evals:
            sum += df.iat[row, col_mask[col]]
        human_eval_avg.append(sum / 5)

    # angular scores
    rouge_slice = rouge[dataset_mask[dataset] * dataset_len : dataset_mask[dataset] * dataset_len + dataset_len]

    # hypothesis testing
    corr, p_val = pearsonr(human_eval_avg, rouge_slice)

    return round(corr, 4), round(p_val, 5)


if __name__ == '__main__':
    # load human evaluations
    human_evals = load_human_evals(human_eval_path)

    # load rouge scores
    rouge_data = load_json_data('data/rouge_f_scores.json')

    # separate rouge1, rouge2, rougeL
    rouge_1_gen_text, rouge_1_gt_text, rouge_1_gen_gt = [], [], []
    rouge_2_gen_text, rouge_2_gt_text, rouge_2_gen_gt = [], [], []
    rouge_L_gen_text, rouge_L_gt_text, rouge_L_gen_gt = [], [], []

    for dataset in rouge_data:
        for i in range(20):
            rouge_1_gen_text.append(dataset['gen_text'][i]['rouge-1'])
            rouge_2_gen_text.append(dataset['gen_text'][i]['rouge-2'])
            rouge_L_gen_text.append(dataset['gen_text'][i]['rouge-l'])

            rouge_1_gt_text.append(dataset['gt_text'][i]['rouge-1'])
            rouge_2_gt_text.append(dataset['gt_text'][i]['rouge-2'])
            rouge_L_gt_text.append(dataset['gt_text'][i]['rouge-l'])

            rouge_1_gen_gt.append(dataset['gen_gt'][i]['rouge-1'])
            rouge_2_gen_gt.append(dataset['gen_gt'][i]['rouge-2'])
            rouge_L_gen_gt.append(dataset['gen_gt'][i]['rouge-l'])

    '''
    # Overall rouge scores
    # rouge 1
    print("OVERALL")
    print("ROUGE-1")
    corr, p_val = analyse(rouge_1_gen_text, human_evals, "gen_text")
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = analyse(rouge_1_gt_text, human_evals, "gt_text")
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = analyse(rouge_1_gen_gt, human_evals, "gen_gt")
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")

    print("ROUGE-2")
    corr, p_val = analyse(rouge_2_gen_text, human_evals, "gen_text")
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = analyse(rouge_2_gt_text, human_evals, "gt_text")
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = analyse(rouge_2_gen_gt, human_evals, "gen_gt")
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")

    print("ROUGE-L")
    corr, p_val = analyse(rouge_L_gen_text, human_evals, "gen_text")
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = analyse(rouge_L_gt_text, human_evals, "gt_text")
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = analyse(rouge_L_gen_gt, human_evals, "gen_gt")
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")
    '''

    for dataset, index in dataset_mask.items():
        print(dataset)
        print("ROUGE-1")
        corr, p_val = analyse_single(rouge_1_gen_text, human_evals, "gen_text", dataset, 20)
        print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
        corr, p_val = analyse_single(rouge_1_gt_text, human_evals, "gt_text", dataset, 20)
        print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
        corr, p_val = analyse_single(rouge_1_gen_gt, human_evals, "gen_gt", dataset, 20)
        print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
        print("...")

        print("ROUGE-2")
        corr, p_val = analyse_single(rouge_2_gen_text, human_evals, "gen_text", dataset, 20)
        print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
        corr, p_val = analyse_single(rouge_2_gt_text, human_evals, "gt_text", dataset, 20)
        print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
        corr, p_val = analyse_single(rouge_2_gen_gt, human_evals, "gen_gt", dataset, 20)
        print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
        print("...")

        print("ROUGE-L")
        corr, p_val = analyse_single(rouge_L_gen_text, human_evals, "gen_text", dataset, 20)
        print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
        corr, p_val = analyse_single(rouge_L_gt_text, human_evals, "gt_text", dataset, 20)
        print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
        corr, p_val = analyse_single(rouge_L_gen_gt, human_evals, "gen_gt", dataset, 20)
        print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
        print("================================================")