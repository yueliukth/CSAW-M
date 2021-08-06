import numpy as np
import pandas as pd
import scipy.stats as stats
import os

import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import data_handler
import helper
import models
from . import utility
import globals


# ---------------- functions related to evaluation metrics ----------------
def calc_kendall_rank_correlation(all_preds, all_labels):
    """Gets the kendall's tau-b rank correlation coefficient.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html

    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.
    Returns
    -------
    correlation: float
        The tau statistic.
    pvalue: float
        The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
    """

    tau, p_value = stats.kendalltau(all_preds, all_labels)
    return tau


def calc_class_absolute_error(all_preds, all_labels):
    """Gets the average mean absolute error (AMAE).

    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.
    Returns
    -------
    amae: float
        The AMAE.
    """

    label_set = list(set(all_labels))
    all_mae = []
    for label in label_set:
        index_list = [i for i, x in enumerate(all_labels) if x == label]
        pred_list = [all_preds[i] for i in index_list]
        label_list = [all_labels[i] for i in index_list]
        mae = mean_absolute_error(pred_list, label_list)
        all_mae.append(mae)
    return np.average(all_mae)


def calc_precision_recall_f1(all_preds, all_labels, bins1, bins2):
    """Gets the F1 score.

    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.
    bins1: list
         A list of masking levels that are considered positive in predictions, for example  [7,8] or [1,2]
    bins2: list
        A list of masking levels that are considered positive in labels, for example  [7,8] or [1,2]

    Returns
    -------
    precision: float
    recall: float
    f1: float
        The F1 score.
    """

    all_preds = ['t' if pred in bins1 else 'o' for pred in all_preds]  # 't': target, 'o': others
    all_labels = ['t' if label in bins2 else 'o' for label in all_labels]

    precision = precision_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')  # look for 't'
    recall = recall_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    return precision, recall, f1


def calc_aucs(all_preds, all_labels):
    """Gets AUC score for downstream task.

    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.

    Returns
    -------
    auc: float
        The AUC score.
    """

    auc = roc_auc_score(all_labels, all_preds)
    return auc


def calc_oddsratio_downstream(all_preds, all_labels, bin_list):
    """Gets odds ratio for downstream task.

    Parameters
    ----------
    all_preds: list
        A list of predicted values, quartile integers [1,2,3,4].
    all_labels: list
        A list of labels.
    bin_list: list
        A list of quartiles that we want to compare against the first quartile

    Returns
    -------
    odds ratio: float
        The odds ratio.
    """

    df = pd.DataFrame({'all_preds': all_preds, "all_labels": all_labels})
    df = df.sort_values(by='all_preds').reset_index(drop=True)
    index_list = df[df['all_preds'].isin(bin_list)].index.tolist()
    df.loc[index_list, 'temp_pred'] = 1
    index_list = df[df['all_preds'].isin([1])].index.tolist()
    df.loc[index_list, 'temp_pred'] = 0
    num_a = df[(df['temp_pred'] == 1) & (df['all_labels'] == 1)].shape[0]
    num_b = df[(df['temp_pred'] == 1) & (df['all_labels'] == 0)].shape[0]
    num_c = df[(df['temp_pred'] == 0) & (df['all_labels'] == 1)].shape[0]
    num_d = df[(df['temp_pred'] == 0) & (df['all_labels'] == 0)].shape[0]

    table = np.array([[num_a, num_b], [num_c, num_d]])
    try:
        oddsratio, pvalue = stats.fisher_exact(table, 'greater')
        return oddsratio
    except:
        return 'nan'


# ---------------- functions to be called for evaluation ----------------
def calc_loss(loss_type, logits, targets):
    if loss_type == 'one_hot':
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, targets)
    elif loss_type == 'multi_hot':
        loss_fn = torch.nn.BCEWithLogitsLoss()  # takes logits and targets both of shape (B, 7)
        return loss_fn(logits, targets)
    else:
        raise NotImplementedError('Loss type not implemented')


def calc_metrics(val_loader, model, loss_type, confusion=False):
    globals.logger.info('\n\033[1mCalculating val metrics...\033[10m')
    all_preds = []
    all_labels = []
    all_image_names = []
    all_bin_probs = []
    all_scores = []

    with torch.no_grad():
        for i_batch, batch in enumerate(val_loader):
            image_names = batch['image_name']
            image_batch, labels, targets = utility.extract_batch(batch, loss_type)

            logits = model(image_batch)
            loss = calc_loss(loss_type, logits, targets).item()  # val or test loss
            preds_list, probs_2d_list = utility.logits_to_preds(logits, loss_type)  # preds in range [0-7]

            # calculate continuous scores and bin probs
            b_size = image_batch.shape[0]
            for idx in range(b_size):  # loop over each item in batch
                probs_1d_list = utility.tensor_to_list(probs_2d_list[idx])  # for current image in batch
                if loss_type == 'one_hot':
                    bin_probs = probs_1d_list  # probs_1d_list is already bin probs for one-hot model
                    score = utility.softmax_score(bin_probs)
                elif loss_type == 'multi_hot':
                    bin_probs, score = utility.multi_hot_score(probs_1d_list)  # probs_1d_list is cumulative probs for multi-hot
                else:
                    raise NotImplementedError('loss_type not implemented')

                all_bin_probs.append(helper.as_str(bin_probs, sep=';'))  # convert the bin probs list to str and append
                all_scores.append(score)

            # append labels and pred
            all_preds.extend(preds_list)
            all_labels.extend(utility.tensor_to_list(labels))  # labels in range [0-7]
            all_image_names.extend(image_names)
            print(f'Metrics calculation done for batch: {i_batch}')

    # convert all predictions and labels from [0-7] back to [1-8]
    all_preds = [data_handler.convert_label(pred, direction='from_train') for pred in all_preds]
    all_labels = [data_handler.convert_label(label, direction='from_train') for label in all_labels]

    # calculating association metrics
    kendall = calc_kendall_rank_correlation(all_preds, all_labels)
    # average mean abs error over classes
    amae = calc_class_absolute_error(all_preds, all_labels)  # average over classes, not dominated by majority
    # precision, recall, f1 for low and high bins
    low_bin_precision, low_bin_recall, low_bin_f1 = calc_precision_recall_f1(all_preds, all_labels, bins1=[1, 2], bins2=[1, 2])
    high_bin_precision, high_bin_recall, high_bin_f1 = calc_precision_recall_f1(all_preds, all_labels, bins1=[7, 8], bins2=[7, 8])

    return_dict = {
        'loss': loss,
        'kendall': kendall,
        'amae': amae,
        'low_bin_precision': low_bin_precision,
        'low_bin_recall': low_bin_recall,
        'low_bin_f1': low_bin_f1,
        'high_bin_precision': high_bin_precision,
        'high_bin_recall': high_bin_recall,
        'high_bin_f1': high_bin_f1,
        'all_image_names': all_image_names,
        'all_preds': all_preds,
        'all_bin_probs': all_bin_probs,
        'all_scores': all_scores,
        'all_labels': all_labels
    }

    if confusion:
        globals.logger.info(f'Calculating confusion matrix for all_pred len: {len(all_preds)}, all_labels len: {len(all_labels)}')
        matrix = confusion_matrix(y_true=all_labels, y_pred=all_preds)
        # matrix_normalized = confusion_matrix(y_true=all_labels, y_pred=all_preds, normalize='true')
        return_dict['matrix'] = matrix
        # return_dict['matrix_normalized'] = matrix_normalized
    return return_dict


def evaluate_model(test_csv, model_name, loss_type, step, params, save_preds_to=None):
    # load model
    model = models.init_and_load_model_for_eval(model_name, loss_type, step)
    # prepare data
    test_list = helper.read_csv_to_list(test_csv)
    globals.logger.info(f'Using test_csv: {test_csv} with lines: {len(test_list)}\n')

    test_dataset_params = {
        'data_list': test_list,
        'data_folder': params['data']['test_folder'],
        'img_size': params['train']['img_size'],
        'imread_mode': params['data']['imread_mode'],
        'line_parse_type': params['data']['line_parse_type'],
        'csv_sep_type': params['data']['csv_sep_type']
    }
    test_dataloader_params = {
        'num_workers': params['train']['n_workers'],
        'batch_size': params['train']['batch_size'],
        'shuffle': False
    }
    globals.logger.info(f'Initializing test data loader...')
    test_loader = data_handler.init_data_loader(test_dataset_params, test_dataloader_params)

    globals.logger.info('Calculating metrics...')
    results_dict = calc_metrics(test_loader, model, loss_type, confusion=True)

    # print results
    globals.logger.info('\n----------------- TEST RESULTS -----------------')
    for k, v in results_dict.items():
        if k not in ['all_image_names', 'all_preds', 'all_bin_probs', 'all_scores', 'all_labels']:  # do not print these
            if k == 'matrix':
                globals.logger.info(f'{k}:\n{v}')
            else:
                globals.logger.info(f'{k}: {round(v, 4)}')

    # write predictions and score to file, if wanted
    if save_preds_to:
        aggregate = zip(results_dict['all_image_names'], results_dict['all_bin_probs'], results_dict['all_preds'])
        header = f"Filename;{helper.as_str([f'Prob_bin_{i}' for i in range(1, 9)], sep=';')};Final_pred"

        helper.make_dir_if_not_exists(os.path.dirname(save_preds_to))
        with open(save_preds_to, 'w') as f:
            f.write(f'{header}\n')
            for item in aggregate:
                f.write(f"{helper.as_str(item, sep=';')}\n")  # convert items in zip to a line for csv file
        globals.logger.info(f'Saved results to: {save_preds_to}')

