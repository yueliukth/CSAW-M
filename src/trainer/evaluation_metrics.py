import numpy as np
import pandas as pd
import scipy.stats as stats

import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import data_handler
from . import utility
import globals


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

    tau, p_value = stats.kendalltau(all_preds, all_labels, 'b')
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


def calc_loss(loss_type, logits, targets):
    if loss_type == 'one_hot':
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, targets)
    elif loss_type == 'multi_hot':
        loss_fn = torch.nn.BCEWithLogitsLoss()  # takes logits and targets both of shape (B, 7)
        return loss_fn(logits, targets)
    else:
        raise NotImplementedError('Loss type not implemented')


def calc_val_metrics(val_loader, model, loss_type, confusion=False):
    globals.logger.info('\n\033[1mCalculating val metrics...\033[10m')
    all_preds = []
    all_labels = []
    all_image_names = []

    with torch.no_grad():
        for i_batch, batch in enumerate(val_loader):
            image_names = batch['image_name']
            image_batch, labels, targets = utility.extract_batch(batch, loss_type)

            logits = model(image_batch)
            val_loss = calc_loss(loss_type, logits, targets).item()
            preds_list = utility.logits_to_preds(logits, loss_type)[0]  # preds in range [0-7]
            labels_list = utility.tensor_to_list(labels)  # labels in range [0-7]

            # for confusion matrix
            all_preds.extend(preds_list)
            all_labels.extend(labels_list)
            all_image_names.extend(image_names)
            print(f'Metrics calculation done for batch: {i_batch}')

    # convert all predictions and labels from [0-7] back to [1-8]
    all_preds = [data_handler.convert_label(pred, direction='from_train') for pred in all_preds]
    all_labels = [data_handler.convert_label(label, direction='from_train') for label in all_labels]

    # calculating association metrics
    kendall = calc_kendall_rank_correlation(all_preds, all_labels)
    # mean abs error
    abs_error = calc_class_absolute_error(all_preds, all_labels)  # average over classes, not dominated by majority
    # precision, recall, f1 for low and high bins
    low_bin_precision, low_bin_recall, low_bin_f1 = calc_precision_recall_f1(all_preds, all_labels, bins1=[1, 2], bins2=[1, 2])
    high_bin_precision, high_bin_recall, high_bin_f1 = calc_precision_recall_f1(all_preds, all_labels, bins1=[7, 8], bins2=[7, 8])

    return_dict = {
        'val_loss': val_loss,
        'kendall': kendall,
        'abs_error': abs_error,
        'low_bin_precision': low_bin_precision,
        'low_bin_recall': low_bin_recall,
        'low_bin_f1': low_bin_f1,
        'high_bin_precision': high_bin_precision,
        'high_bin_recall': high_bin_recall,
        'high_bin_f1': high_bin_f1,
        'all_image_names': all_image_names,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

    if confusion:
        globals.logger.info(f'Calculating confusion matrix for all_pred len: {len(all_preds)}, all_labels len: {len(all_labels)}')
        matrix = confusion_matrix(y_true=all_labels, y_pred=all_preds)
        matrix_normalized = confusion_matrix(y_true=all_labels, y_pred=all_preds, normalize='true')
        return_dict['matrix'] = matrix
        return_dict['matrix_normalized'] = matrix_normalized
    return return_dict
