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


# Generated vs. text comparison
def full_dataset_analysis(human_evals, aes_data, col):
    # human average for gen-text
    human_avg = calculate_human_average(human_evals, col_mask[col])

    # angular scores
    angular = []
    datasets = ["Amazon", "CNN_DM", "Newsroom", "Gigaword"]
    for idx, dataset in enumerate(datasets):
        angular.extend(aes_data[idx][dataset][col])

    # hypothesis testing
    corr, p_val = pearsonr(human_avg, angular)

    return round(corr, 4), round(p_val, 8)


def single_dataset_analysis(human_evals, aes_data, col, dataset, dataset_len):
    human_eval_avg = []
    for row in range(dataset_mask[dataset] * dataset_len, dataset_mask[dataset] * dataset_len + dataset_len):
        sum = 0
        for df in human_evals:
            sum += df.iat[row, col_mask[col]]
        human_eval_avg.append(sum / 5)

    # angular scores
    angular = aes_data[dataset_mask[dataset]][dataset][col]

    # hypothesis testing
    corr, p_val = pearsonr(human_eval_avg, angular)

    return round(corr, 4), round(p_val, 5)


def rouge_analysis():
    rouge_data = load_json_data('data/rouge_f_scores.json')
    print(rouge_data[0])

    # separate rouge1, rouge2, rougeL
    rouge_1, rouge_2, rouge_L = [], [], []




if __name__ == '__main__':
    rouge_analysis()
    exit(0)

    # load human evaluations
    human_evals = load_human_evals(human_eval_path)

    # load angular scores
    aes_data = load_json_data(aes_path)

    # Overall
    print("OVERALL")
    corr, p_val = full_dataset_analysis(human_evals, aes_data, "gen_text")
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = full_dataset_analysis(human_evals, aes_data, "gt_text")
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = full_dataset_analysis(human_evals, aes_data, "gen_gt")
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")

    # For each dataset

    # Amazon
    print("AMAZON")
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_text", "Amazon", 20)
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gt_text", "Amazon", 20)
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_gt", "Amazon", 20)
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")

    # CNN_DailyMail
    print("CNN_DAILYMAIL")
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_text", "CNN_DM", 20)
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gt_text", "CNN_DM", 20)
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_gt", "CNN_DM", 20)
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")

    # Newsroom
    print("NEWSROOM")
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_text", "Newsroom", 20)
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gt_text", "Newsroom", 20)
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_gt", "Newsroom", 20)
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
    print("================================================")

    # Gigaword
    print("GIGAWORD")
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_text", "Gigaword", 20)
    print("Generated-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gt_text", "Gigaword", 20)
    print("GT-Text ->   r-value: {}, p-value: {}".format(corr, p_val))
    corr, p_val = single_dataset_analysis(human_evals, aes_data, "gen_gt", "Gigaword", 20)
    print("Generated-GT ->   r-value: {}, p-value: {}".format(corr, p_val))
