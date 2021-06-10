import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score


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


def calc_f1_score(all_preds, all_labels, bins1, bins2):
    """Gets the F1 score.

    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.
    bin1: list
         A list of masking levels that are considered positive in predictions, for example  [7,8] or [1,2]
    bin2: list
        A list of masking levels that are considered positive in labels, for example  [7,8] or [1,2]

    Returns
    -------
    f1: float
        The F1 score.
    """

    all_preds = ['t' if pred in bins1 else 'o' for pred in all_preds] # 't': target, 'o': others
    all_labels = ['t' if label in bins2 else 'o' for label in all_labels]

    f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    return f1


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