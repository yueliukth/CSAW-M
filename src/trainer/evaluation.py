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

def make_master_pred_csv(mapping_csv_path, label_csv_path, multihot_pred_path, softmax_pred_path):
    # target final columns are
    # ['Filename', 'Label', 'Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5',
    # 'If_cancer', 'If_interval_cancer', 'If_large_invasive_cancer', 'If_composite',
    # 'Dicom_image_laterality', 'Dicom_window_center', 'Dicom_window_width', 'Dicom_photometric_interpretation',
    # 'Libra_percent_density', 'Libra_dense_area', 'Libra_breast_area',
    # 'final_pred_multihot_1', 'final_score_multihot_1',
    # 'final_pred_multihot_2', 'final_score_multihot_2',
    # 'final_pred_multihot_3', 'final_score_multihot_3',
    # 'final_pred_multihot_4', 'final_score_multihot_4',
    # 'final_pred_multihot_5', 'final_score_multihot_5',
    # 'final_pred_softmax_1', 'final_score_softmax_1',
    # 'final_pred_softmax_2', 'final_score_softmax_2',
    # 'final_pred_softmax_3', 'final_score_softmax_3',
    # 'final_pred_softmax_4', 'final_score_softmax_4',
    # 'final_pred_softmax_5', 'final_score_softmax_5']

    df1 = pd.read_csv(mapping_csv_path, delimiter=';', dtype={'sourcefile': str})
    df = pd.read_csv(label_csv_path, delimiter=';', dtype={'sourcefile': str})
    df = reduce(lambda left, right: pd.merge(left, right, on=['Filename'],
        how='inner'), [df1, df]).reset_index(drop=True)

    df1 = pd.read_csv(multihot_pred_path, delimiter=',', dtype={'sourcefile': str})
    df1 = df1.rename(columns={'Filename': 'Filename_original'})
    column_list = [column for column in df1.columns.tolist() if 'final_pred_' in column or 'final_score_' in column]
    new_column_list = ['_'.join(column.split('_')[:-1]) + '_multihot_' + column.split('_')[-1] for column in
                       column_list]
    new_column_dict = dict(zip(column_list, new_column_list))
    df1 = df1.rename(columns=new_column_dict)
    df = reduce(lambda left, right: pd.merge(left, right, on=['Filename_original'],
        how='inner'), [df, df1]).reset_index(drop=True)

    df1 = pd.read_csv(softmax_pred_path, delimiter=',', dtype={'sourcefile': str})
    df1 = df1.rename(columns={'Filename': 'Filename_original'})
    column_list = [column for column in df1.columns.tolist() if 'final_pred_' in column or 'final_score_' in column]
    new_column_list = ['_'.join(column.split('_')[:-1]) + '_softmax_' + column.split('_')[-1] for column in column_list]
    new_column_dict = dict(zip(column_list, new_column_list))
    df1 = df1.rename(columns=new_column_dict)
    df = reduce(lambda left, right: pd.merge(left, right, on=['Filename_original'],
        how='inner'), [df, df1]).reset_index(drop=True)
    del df['Filename_original']

    # change bin range from 0-7 to 1-8, if needed
    column_list = [column for column in df.columns.tolist() if 'final_pred_' in column]
    for column in column_list:
        if np.max(df[column].tolist()) == 7 or np.min(df[column].tolist()) == 0:
            df[column] = df[column] + 1
            df[column] = df[column].apply(int)

    return df

def df_score_to_bin(df):
    num = int(df.shape[0] / 8)
    column_list = [column for column in df.columns.tolist() if 'score_' in column]
    for column in column_list + ['Libra_percent_density', 'Libra_dense_area', 'Libra_breast_area']:
        df = df.sort_values(by=column).reset_index(drop=True)
        str2 = column + '_bin'
        df.loc[:num, str2] = 1
        df.loc[num:num * 2, str2] = 2
        df.loc[num * 2:num * 3, str2] = 3
        df.loc[num * 3:num * 4, str2] = 4
        df.loc[num * 4:num * 5, str2] = 5
        df.loc[num * 5:num * 6, str2] = 6
        df.loc[num * 6:num * 7, str2] = 7
        df.loc[num * 7:, str2] = 8
    return df


def make_ranks_from_private(df):
    df_with_ranks = df.copy()
    # sort expert annoations to get [1, 475]
    df_with_ranks['Expert_1'] = df_with_ranks['Expert_1'].rank()
    df_with_ranks['Expert_2'] = df_with_ranks['Expert_2'].rank()
    df_with_ranks['Expert_3'] = df_with_ranks['Expert_3'].rank()
    df_with_ranks['Expert_4'] = df_with_ranks['Expert_4'].rank()
    df_with_ranks['Expert_5'] = df_with_ranks['Expert_5'].rank()

    # get median as label
    column_list = [column for column in df_with_ranks.columns.tolist() if 'Expert' in column]
    df_with_ranks['Label'] = df_with_ranks[column_list].median(axis=1).round(decimals=8)
    df_with_ranks['Label'] = df_with_ranks['Label'].apply(int)

    # sort onehot and multihot predictions to get [1, 475]
    column_list = [i for i in df_with_ranks.columns.tolist() if 'score' in i]
    for column in column_list:
        df_with_ranks[column.replace('score', 'pred')] = df_with_ranks[column].rank()
    return df_with_ranks



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


def calc_precision_and_recall_network(all_preds, all_labels, bins1, bins2):
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

    all_labels = ['t' if label in bins1 else 'o' for label in all_labels]  # 't': target, 'o': others
    all_preds = ['t' if pred in bins2 else 'o' for pred in all_preds]
    precision = precision_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')  # look for 't'
    recall = recall_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    return precision, recall, f1

def calc_aucs(all_preds_prob, all_auc_labels):
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

    auc = roc_auc_score(all_auc_labels, all_preds_prob)
    return auc

def calc_oddsratio_downstream(all_preds, all_labels, bin_list, metric):
    """Gets odds ratio for downstream task.
    Parameters
    ----------
    all_preds: list
        A list of predicted values, percentile integers [1,2,3,4,5,6,7,8].
    all_labels: list
        A list of labels, binary labels for downstream task.
    bin_list: list
        A list of percentiles that we want to compare against the first 2 percentiles
    Returns
    -------
    odds ratio: float
        The odds ratio.
    """

    df = pd.DataFrame({'all_preds': all_preds, "all_labels": all_labels})
    df = df.sort_values(by='all_preds').reset_index(drop=True)

    column = 'all_preds'
    index_list = df[df[column].isin(bin_list)].index.tolist()
    df.loc[index_list, 'temp_pred'] = 1
    index_list = df[df[column].isin([1, 2])].index.tolist()
    df.loc[index_list, 'temp_pred'] = 0
    num_a = df[(df['temp_pred'] == 1) & (df['all_labels'] == 1)].shape[0]
    num_b = df[(df['temp_pred'] == 1) & (df['all_labels'] == 0)].shape[0]
    num_c = df[(df['temp_pred'] == 0) & (df['all_labels'] == 1)].shape[0]
    num_d = df[(df['temp_pred'] == 0) & (df['all_labels'] == 0)].shape[0]

    table = np.array([[num_a, num_b], [num_c, num_d]])
    try:
        oddsratio, pvalue = stats.fisher_exact(table, 'greater')
        if metric == 'pvalue':
            return pvalue
        elif metric == 'oddsratio':
            return oddsratio
    except:
        return 'nan'


def get_metric(metric, all_preds, all_labels, bin_list1, bin_list2):
    if metric == 'kendall':
        return calc_kendall_rank_correlation(all_preds, all_labels)
    elif metric == 'amae':
        return calc_class_absolute_error(all_preds, all_labels)
    elif metric == 'lowbinf1':
        return calc_precision_and_recall_network(all_preds, all_labels, [1, 2], [1, 2])[-1]
    elif metric == 'highbinf1':
        return calc_precision_and_recall_network(all_preds, all_labels, [7, 8], [7, 8])[-1]
    elif metric == 'lowbinf1_private':
        return calc_precision_and_recall_network(all_preds, all_labels, bin_list1, bin_list2)[-1]
    elif metric == 'highbinf1_private':
        return calc_precision_and_recall_network(all_preds, all_labels, bin_list1, bin_list2)[-1]

# ---------------- functions related to plotting results ----------------

def highlight_min(s):
    if s.dtype == np.object:
        is_min = [False for _ in range(s.shape[1])]
    else:
        is_min = s == s.min()

    return ['background: lightgreen' if cell else ''
            for cell in is_min]


def highlight_max(s):
    if s.dtype == np.object:
        is_max = [False for _ in range(s.shape[1])]
    else:
        is_max = s == s.max()

    return ['background: lightgreen' if cell else ''
            for cell in is_max]


def get_table_metric(df, metric, if_highlight, bin_list1, bin_list2):
    table = [['', 'Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5', 'Softmax', 'Multi-hot']]

    # rows to predict median and each radiologist
    for label_column in ['GT-Median', 'Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5']:
        pred_temp_list = [label_column]
        if label_column == 'GT-Median':
            label_column = 'Label'
        # radiologists
        for column in ['Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5']:
            label_list = df[label_column].tolist()
            pred_list = df[column].tolist()
            pred_temp_list.append(get_metric(metric, pred_list, label_list, bin_list1, bin_list2))
        # models
        for str1 in ['softmax', 'multihot']:
            column_list = [column for column in df.columns.tolist() if 'pred_' + str1 in column]
            temp_list = []
            for column in column_list:
                label_list = df[label_column].tolist()
                pred_list = df[column].tolist()
                temp_list.append(get_metric(metric, pred_list, label_list, bin_list1, bin_list2))
            pred_temp_list.append(str("%.4f" % np.average(temp_list)) + ' +- ' + str("%.4f" % np.std(temp_list)))
        table.append(pred_temp_list)

    table_df = pd.DataFrame(table[1:], columns=table[0])

    for i in range(1, 6):
        table_df['Expert_' + str(i)] = table_df['Expert_' + str(i)].apply(float)
    table_df['Softmax_mean'] = table_df['Softmax'].str.split(" ").str[0].apply(float)
    table_df['Multi-hot_mean'] = table_df['Multi-hot'].str.split(" ").str[0].apply(float)
    table_df = table_df[
        ['', 'Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5', 'Softmax_mean', 'Multi-hot_mean', 'Softmax',
         'Multi-hot']]
    table_df = table_df.apply(pd.to_numeric, errors='ignore')
    table_df = table_df.round(4)

    if not if_highlight:
        return table_df
    else:
        if metric == 'amae':
            table_df = table_df.replace(0, np.nan)
            return table_df.style.highlight_min(color='lightgreen', axis=1)
        else:
            table_df = table_df.replace(1, np.nan)
            return table_df.style.highlight_max(color='lightgreen', axis=1)


def plot_corr_map(corr, save_path=None, masking=True, vmax=None, cmap='Blues', fmt=".2f"):
    mask = np.zeros_like(corr)
    mask[[2, 3, 3, 4, 4, 4, 5, 5, 5, 5],
         [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        if masking:
            ax = sns.heatmap(corr, mask=mask, square=True, annot=True, cmap=cmap, vmax=vmax, fmt=fmt,
                annot_kws={"fontsize": 8},
                yticklabels=['GT - Median', 'Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5'],
                xticklabels=['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5', 'One-hot', 'Multi-hot'])
        else:
            ax = sns.heatmap(corr, square=True, annot=True, cmap=cmap, vmax=vmax, fmt=fmt, annot_kws={"fontsize": 13},
                yticklabels=['GT - Median', 'Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5'],
                xticklabels=['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5', 'One-hot', 'Multi-hot'])

        ax.tick_params(labeltop=True, labelbottom=False, rotation=25, axis='x', labelsize=12)
        ax.tick_params(labeltop=True, labelbottom=False, axis='y', labelsize=12)
        plt.vlines(5, 0, 5, colors='white', linestyles='dashed', linewidth=2)
        plt.hlines(1, 0, 7, colors='white', linestyles='dashed', linewidth=2)
    if save_path != None:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

def get_table_metric_downstream_auc(df, if_highlight=True):
    table = [['AUC', 'If_interval_cancer', 'If_large_invasive_cancer', 'If_composite']]
    # rows to use softmax and multihot
    for str1 in ['softmax', 'multihot']:
        if str1 == 'softmax':
            pred_temp_list = ['onehot']
        else:
            pred_temp_list = [str1]
        column_list = [column for column in df.columns.tolist() if 'score_' + str1 in column]
        for label_str in ['If_interval_cancer', 'If_large_invasive_cancer', 'If_composite']:
            label_list = df[label_str].tolist()
            temp_list = []
            for column in column_list:
                pred_list = df[column].tolist()
                temp_list.append(calc_aucs(pred_list, label_list))

            pred_temp_list.append(str("%.4f" % np.average(temp_list)) + ' +- ' + str("%.4f" % np.std(temp_list)))
        table.append(pred_temp_list)

    # rows to use libra densites
    for str1 in ['Libra_percent_density', 'Libra_dense_area', 'Libra_breast_area']:
        pred_temp_list = [str1]
        for label_str in ['If_interval_cancer', 'If_large_invasive_cancer', 'If_composite']:
            label_list = df[label_str].tolist()
            pred_list = df[str1].tolist()
            pred_temp_list.append(calc_aucs(pred_list, label_list))
        table.append(pred_temp_list)

    table_df = pd.DataFrame(table[1:], columns=table[0])

    table_df['If_interval_cancer_mean'] = table_df['If_interval_cancer'].str.split(" ").str[0].apply(float)
    table_df['If_large_invasive_cancer_mean'] = table_df['If_large_invasive_cancer'].str.split(" ").str[0].apply(float)
    table_df['If_composite_mean'] = table_df['If_composite'].str.split(" ").str[0].apply(float)

    index_list = [2, 3, 4]
    table_df.loc[index_list, 'If_interval_cancer_mean'] = table_df.loc[index_list, 'If_interval_cancer']
    table_df.loc[index_list, 'If_large_invasive_cancer_mean'] = table_df.loc[index_list, 'If_large_invasive_cancer']
    table_df.loc[index_list, 'If_composite_mean'] = table_df.loc[index_list, 'If_composite']

    table_df = table_df.apply(pd.to_numeric, errors='ignore')
    table_df = table_df.round(4)
    if not if_highlight:
        return table_df
    else:
        return table_df.style.highlight_max(color='lightgreen', axis=0)


def get_table_metric_downstream_oddsratio(df, target='interval', metric='oddsratio'):
    table = [['Interval OR', 'softmax', 'multi-hot', 'Libra_percent_density', 'Libra_dense_area', 'Libra_breast_area'],
             ['', 1, 1, 1, 1, 1]]
    # row for 2nd quartile Interval
    if target == 'interval':
        label_list = df['If_interval_cancer'].tolist()
    else:
        label_list = df['If_composite'].tolist()

    for quartile_idx in [2, 3, 4]:
        pred_temp_list = [str(quartile_idx) + 'nd quartile']
        for str1 in ['softmax', 'multihot']:
            column_list = [column for column in df.columns.tolist() if 'score_' + str1 in column and 'bin' in column]
            temp_list = []
            for column in column_list:
                pred_list = df[column].tolist()
                temp_list.append(calc_oddsratio_downstream(all_preds=pred_list, all_labels=label_list,
                    bin_list=[quartile_idx * 2 - 1, quartile_idx * 2], metric=metric))
            pred_temp_list.append(str("%.4f" % np.average(temp_list)) + ' +- ' + str("%.4f" % np.std(temp_list)))

        for column in ['Libra_percent_density_bin', 'Libra_dense_area_bin', 'Libra_breast_area_bin']:
            pred_list = df[column].tolist()
            pred_temp_list.append(calc_oddsratio_downstream(all_preds=pred_list, all_labels=label_list,
                bin_list=[quartile_idx * 2 - 1, quartile_idx * 2], metric=metric))

        table.append(pred_temp_list)

    table_df = pd.DataFrame(table[1:], columns=table[0])
    table_df['softmax_mean'] = table_df['softmax'].str.split(" ").str[0].apply(float)
    table_df['multi-hot_mean'] = table_df['multi-hot'].str.split(" ").str[0].apply(float)
    table_df.iloc[0] = int(1)
    table_df = table_df.apply(pd.to_numeric, errors='ignore')
    table_df = table_df[
        ['softmax', 'multi-hot', 'softmax_mean', 'multi-hot_mean', 'Libra_percent_density', 'Libra_dense_area',
         'Libra_breast_area']]
    table_df = table_df.round(4)
    if metric == 'oddsratio':
        return table_df.style.highlight_max(color='lightgreen', axis=0)
    elif metric == 'pvalue':
        return table_df.style.highlight_min(color='lightgreen', axis=0)


def get_oddsratio_plots(df, save_path=None, target='interval', metric='oddsratio'):
    table = [['Interval OR', 'softmax', 'multi-hot', 'Libra_percent_density', 'Libra_dense_area', 'Libra_breast_area'],
             ['', 1, 1, 1, 1, 1]]
    plot_df = pd.DataFrame(columns=['quartile', 'prediction', 'odds_ratio'])
    count = 0
    # row for 2nd quartile Interval
    if target == 'interval':
        label_list = df['If_interval_cancer'].tolist()
    elif target == 'largeinvasive':
        label_list = df['If_large_invasive_cancer'].tolist()
    else:
        label_list = df['If_composite'].tolist()

    for quartile_idx in [2, 3, 4]:
        for str1 in ['softmax', 'multihot']:
            column_list = [column for column in df.columns.tolist() if 'score_' + str1 in column and 'bin' in column]

            for column in column_list:
                plot_df.loc[count, 'quartile'] = 1
                plot_df.loc[count, 'prediction'] = column
                plot_df.loc[count, 'odds_ratio'] = 1
                count += 1

                pred_list = df[column].tolist()
                or_value = calc_oddsratio_downstream(all_preds=pred_list, all_labels=label_list,
                    bin_list=[quartile_idx * 2 - 1, quartile_idx * 2], metric=metric)
                plot_df.loc[count, 'quartile'] = quartile_idx
                plot_df.loc[count, 'prediction'] = column
                plot_df.loc[count, 'odds_ratio'] = or_value
                count += 1

        for column in ['Libra_percent_density_bin', 'Libra_dense_area_bin']:
            plot_df.loc[count, 'quartile'] = 1
            plot_df.loc[count, 'prediction'] = column
            plot_df.loc[count, 'odds_ratio'] = 1
            count += 1
            pred_list = df[column].tolist()
            or_value = calc_oddsratio_downstream(all_preds=pred_list, all_labels=label_list,
                bin_list=[quartile_idx * 2 - 1, quartile_idx * 2], metric=metric)
            plot_df.loc[count, 'quartile'] = quartile_idx
            plot_df.loc[count, 'prediction'] = column
            plot_df.loc[count, 'odds_ratio'] = or_value
            count += 1

    index_list = plot_df[plot_df['prediction'].str.contains('softmax')].index.tolist()
    plot_df.loc[index_list, 'model'] = 'One-hot'

    index_list = plot_df[plot_df['prediction'].str.contains('multihot')].index.tolist()
    plot_df.loc[index_list, 'model'] = 'Multi-hot'

    index_list = plot_df[plot_df['prediction'].str.contains('Libra_percent_density_bin')].index.tolist()
    plot_df.loc[index_list, 'model'] = 'Percent density'

    index_list = plot_df[plot_df['prediction'].str.contains('Libra_dense_area_bin')].index.tolist()
    plot_df.loc[index_list, 'model'] = 'Dense area'

    plot_df = plot_df.drop_duplicates()

    plot_df['model+quartile'] = plot_df['model'] + plot_df['quartile'].apply(str)

    index_list = plot_df[plot_df['model+quartile'].str.contains('Percent')].index.tolist()
    plot_df.loc[index_list, 'x_order'] = 3
    index_list = plot_df[plot_df['model+quartile'].str.contains('Dense')].index.tolist()
    plot_df.loc[index_list, 'x_order'] = 4
    index_list = plot_df[plot_df['model+quartile'].str.contains('One-hot')].index.tolist()
    plot_df.loc[index_list, 'x_order'] = 1
    index_list = plot_df[plot_df['model+quartile'].str.contains('Multi-hot')].index.tolist()
    plot_df.loc[index_list, 'x_order'] = 2
    plot_df = plot_df.sort_values(by=['x_order', 'quartile']).reset_index(drop=True)

    sns.set_style("white")
    palette = ['dimgray'] * 100
    g = sns.catplot(col='model', x='quartile', y="odds_ratio", palette=palette, kind='point', data=plot_df, height=3.5,
        aspect=.7, sharey=True)
    g.despine(left=True)

    if target == 'interval':
        g.set_axis_labels("", "Odds ratio (interval cancer)", size=16)
    elif target == 'largeinvasive':
        g.set_axis_labels("", "Odds ratio (large invasive cancer)", size=16)
    else:
        g.set_axis_labels("", "Odds ratio (CEP)", size=16)
    (g.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], size=16)
     .set_titles("{col_name}", size=16)
     .despine(left=True))
    g.set(yticks=[0, 1, 2, 3, 4])
    g.set_yticklabels([0, 1, 2, 3, 4], size=16)
    g.set(ylim=(0, 4.2))
    g.fig.subplots_adjust(wspace=1, hspace=0)

    g.tight_layout()
    ax1, ax2, ax3, ax4 = g.axes[0]
    ax1.axhline(1, ls='--', color='gray', alpha=0.5)
    ax2.axhline(1, ls='--', color='gray', alpha=0.5)
    ax3.axhline(1, ls='--', color='gray', alpha=0.5)
    ax4.axhline(1, ls='--', color='gray', alpha=0.5)
    if save_path != None:
        g.savefig(save_path, bbox_inches='tight')
    plt.show()
    return plot_df


def make_variations_violin_plot(df, save_path=None):
    expert1_masking = df['Expert_1'].tolist()
    expert2_masking = df['Expert_2'].tolist()
    expert3_masking = df['Expert_3'].tolist()
    expert4_masking = df['Expert_4'].tolist()
    expert5_masking = df['Expert_5'].tolist()
    label_masking = df['Label'].tolist()
    percent_density = df['Libra_percent_density'].tolist()
    dense_area = df['Libra_dense_area'].tolist()

    len_annotations = len(expert1_masking)
    experts_id = ['GT - Median']*len_annotations + ['Expert 1']*len_annotations + ['Expert 2']*len_annotations + ['Expert 3']*len_annotations + ['Expert 4']*len_annotations + ['Expert 5']*len_annotations
    experts_masking = label_masking + expert1_masking+expert2_masking+expert3_masking+expert4_masking+expert5_masking
    experts_percent_density = percent_density*6
    experts_dense_area = dense_area*6

    violin_df = pd.DataFrame(list(zip(experts_id, experts_masking, experts_percent_density, experts_dense_area)), columns=['Expert ID', 'Masking', 'Percent density', 'Dense area'])
    sns.set(font_scale = 1.25)
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(16, 5))
        ax = sns.violinplot(x="Expert ID", y="Percent density", hue='Masking', data=violin_df, linewidth=0.5,palette=sns.color_palette("Blues", 14)[5:11])
        ax.legend(title='Masking level', loc='upper right', bbox_to_anchor=(1.18, 0.95))
        plt.xlabel('')
        plt.grid(b=True)
        if save_path != None:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


def get_metric_seperate_masking_levels(df, gt_column, rows, columns, metric):
    all_list = []
    for row in rows:
        temp_list = []
        for column in columns:
            if column == 'Masking 1-2':
                new_df = df[df[gt_column].isin([1, 2])]
            elif column == 'Masking 3-4':
                new_df = df[df[gt_column].isin([3, 4])]
            elif column == 'Masking 5-6':
                new_df = df[df[gt_column].isin([5, 6])]
            elif column == 'Masking 7-8':
                new_df = df[df[gt_column].isin([7, 8])]

            if 'Expert' in row or 'Label' in row:
                temp_list.append(get_metric(metric, new_df[row].tolist(), new_df[gt_column].tolist(), None, None))
            else:
                if row == 'One-hot':
                    five_rows = [column for column in df.columns.tolist() if 'final_pred_softmax' in column]
                elif row == 'Multi-hot':
                    five_rows = [column for column in df.columns.tolist() if 'final_pred_multihot' in column]
                kendall_list = []
                for five_row in five_rows:
                    kendall_list.append(
                        get_metric(metric, new_df[five_row].tolist(), new_df[gt_column].tolist(), None, None))

                temp_list.append(np.average(kendall_list))
        all_list.append(temp_list)
    return all_list


def plot_metric_seperate_masking_levels(corr, save_path=None, vmax=None, cmap='Blues', fmt=".2f", which_bold=None):
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 6))

        ax = sns.heatmap(np.transpose(corr), square=True, annot=True, cmap=cmap, vmax=vmax, fmt=fmt,
            annot_kws={"fontsize": 13}, cbar_kws={"shrink": .72},
            xticklabels=['GT - Median', 'Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5', 'One-hot',
                         'Multi-hot'],
            yticklabels=columns)

        ax.tick_params(labeltop=True, labelbottom=False, rotation=25, axis='x', labelsize=12)
        ax.tick_params(labeltop=True, labelbottom=False, rotation=0, axis='y', labelsize=12)
        if not which_bold == None:
            ax.get_xticklabels()[which_bold].set_fontweight("bold")
        if save_path != None:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')


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


def calc_metrics(val_loader, model, loss_type, confusion=False, only_get_preds=False):
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
            if not only_get_preds:
                loss = calc_loss(loss_type, logits, targets).item()  # val or test loss
            else:
                loss = None
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
            all_labels.extend(utility.tensor_to_list(labels) if 'none' not in labels else labels)  # labels in range [0-7]
            all_image_names.extend(image_names)
            print(f'Metrics calculation done for batch: {i_batch}')

    # convert all predictions and labels from [0-7] back to [1-8]
    all_preds = [data_handler.convert_label(pred, direction='from_train') for pred in all_preds]
    all_labels = [data_handler.convert_label(label, direction='from_train') if label != 'none' else label for label in all_labels]

    # calc metrics
    if not only_get_preds:
        # calculating association metrics
        kendall = calc_kendall_rank_correlation(all_preds, all_labels)
        # average mean abs error over classes
        amae = calc_class_absolute_error(all_preds, all_labels)  # average over classes, not dominated by majority
        # precision, recall, f1 for low and high bins
        low_bin_precision, low_bin_recall, low_bin_f1 = calc_precision_recall_f1(all_preds, all_labels, bins1=[1, 2], bins2=[1, 2])
        high_bin_precision, high_bin_recall, high_bin_f1 = calc_precision_recall_f1(all_preds, all_labels, bins1=[7, 8], bins2=[7, 8])
    else:
        kendall = amae = None
        low_bin_precision = low_bin_recall = low_bin_f1 = high_bin_precision = high_bin_recall = high_bin_f1 = None

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


def evaluate_model(model_name, loss_type, step, params, only_get_preds=False, save_preds_to=None):
    # load model
    model = models.init_and_load_model_for_eval(model_name, loss_type, step)

    # prepare data
    test_csv = params['data']['test_csv']

    if test_csv != 'none':
        test_list = helper.read_csv_to_list(test_csv)
        globals.logger.info(f'Using test_csv: {test_csv} with lines: {len(test_list)}\n')
    else:  # just data_folder is provided and the filenames are extracted in the MammoDataset
        test_list = None
        globals.logger.info(f'No test_csv provided, set test_list to {test_list}\n')

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
    if_confusion = not only_get_preds
    results_dict = calc_metrics(test_loader, model, loss_type, confusion=if_confusion, only_get_preds=only_get_preds)

    # print results
    if not only_get_preds:
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

