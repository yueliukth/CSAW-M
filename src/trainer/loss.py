import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import scipy.stats as stats

from . import utility
import globals


def weighted_multi_hot_loss(logits, targets, pos_weights, neg_weights):
    probs = torch.sigmoid(logits)  # targets and logits shape: (B, 7)
    pos_loss = pos_weights * targets * torch.log(probs)  # pos_weights and neg_weights: torch.Size([7])
    neg_loss = neg_weights * (1 - targets) * torch.log(1 - probs)
    total_loss = -(pos_loss + neg_loss)
    mean_loss = torch.mean(total_loss)
    return mean_loss


def calc_kendall_rank_correlation(all_labels, all_preds):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    # Kendall’s tau is a measure of the correspondence between two rankings.
    # Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement.
    # This is the tau-b or c version of Kendall’s tau which accounts for ties.
    # tau-a version fails if the dataset contains a high portion of ties
    # We should monitor both tau and p value
    tau, p_value = stats.kendalltau(all_labels, all_preds)  # default tau-b
    return tau, p_value


def calc_spearman_rank_correlation(all_labels, all_preds):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    # Positive correlations imply that as x increases, so does y.
    # Negative correlations imply that as x increases, y decreases.
    # The p-value roughly indicates the probability of an uncorrelated system producing datasets that
    # have a Spearman correlation at least as extreme as the one computed from these datasets.
    # The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.
    corr, p_value = stats.spearmanr(all_labels, all_preds)  # which applies fractional ranks to ties
    return corr, p_value


def calc_class_absolute_error(all_labels, all_preds):
    diffs = [0 for _ in range(8)]
    for i in range(len(all_labels)):
        pred, label = all_preds[i], all_labels[i]
        abs_diff = abs(pred - label)
        diffs[label] += abs_diff

    all_counts = [np.sum([(all_labels[index] == j) for index in range(len(all_labels))]) for j in range(8)]
    mean_diffs = [diffs[ind] / all_counts[ind] for ind in range(8)]

    avg_mae = np.mean(mean_diffs)

    min_mae = np.min(mean_diffs)
    min_mae_pos = mean_diffs.index(min_mae)

    max_mae = np.max(mean_diffs)
    max_mae_pos = mean_diffs.index(max_mae)
    return {
        'avg_mae': avg_mae,
        'min_mae': min_mae,
        'min_mae_pos': min_mae_pos,
        'max_mae': max_mae,
        'max_mae_pos': max_mae_pos
    }
    # return np.mean(mean_diffs), np.max(mean_diffs), np.min(mean_diffs)  # average/max over all classes, each one treated equally


def calc_precision_and_recall(all_labels, all_preds, gt_hyper_bin, pred_hyper_bin):
    # bins = [0, 1] if group == 'low' else [6, 7]
    all_labels = ['t' if label in gt_hyper_bin else 'o' for label in all_labels]  # 't': target, 'o': others
    all_preds = ['t' if pred in pred_hyper_bin else 'o' for pred in all_preds]

    precision = precision_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')  # look for 't'
    recall = recall_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='binary', pos_label='t')
    return precision, recall, f1


def calc_metric_for_batch(preds_list, labels_list, metric, bin_threshold=None):
    batch_size = len(preds_list)
    if bin_threshold is not None:
        equalities = [abs(preds_list[i] - labels_list[i]) <= bin_threshold for i in range(len(preds_list))]
    else:
        equalities = [preds_list[i] == labels_list[i] for i in range(len(preds_list))]

    absolute_diff = np.sum([abs(preds_list[i] - labels_list[i]) for i in range(len(preds_list))])

    if metric == 'average_acc':
        return float(np.sum(equalities)) / batch_size
    elif metric == 'equalities':
        return np.sum(equalities)
    elif metric == 'absolute_diff':
        return absolute_diff
    elif metric == 'mean_absolute_diff':
        return float(absolute_diff) / batch_size
    else:
        raise NotImplementedError


def calc_loss(loss_type, logits, targets, class_weights=None, pos_weights=None, neg_weights=None, sample_weights=None):
    if loss_type == 'one_hot':
        if class_weights is not None:  # weighted cross entropy
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(globals.get_current_device())
            loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
            return loss_fn(logits, targets)

        # elif samples_weighted:
        #     raise NotImplementedError('This part of code needs model_mode')
        #     loss_fn = nn.CrossEntropyLoss(reduction='none')  # returns a vector of losses
        #     loss = loss_fn(logits, targets)  # targets are labels themselves
        #     with torch.no_grad():  # compute sample weights based one prediction distance
        #         preds = torch.tensor(logits_to_preds(logits, loss_type), dtype=torch.long, device=globals.get_current_device())
        #         distance_weights = (torch.abs(preds - targets) + 1).float()  # for distance 0, this factor becomes 1
        #     loss = (loss * distance_weights / distance_weights.sum()).sum()  # weighted average
        #     return loss

        else:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, targets)

    elif loss_type == 'multi_hot':
        if pos_weights is None:  # no weight, usual multi-hot loss
            if sample_weights is not None:  # used for interval cancer weights
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss = loss_fn(logits, targets)  # given tensor of size (N, 7)
                loss = torch.sum(loss, dim=1)  # make it of size (N), sum over losses for each label
                res = loss * sample_weights.float()
                return torch.sum(res) / torch.sum(sample_weights)  # return the weighted avg

            else:
                loss_fn = torch.nn.BCEWithLogitsLoss()  # takes logits and targets both of shape (B, 7)
                return loss_fn(logits, targets)

        else:
            pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(globals.get_current_device())
            neg_weights = torch.tensor(neg_weights, dtype=torch.float32).to(globals.get_current_device())
            return weighted_multi_hot_loss(logits, targets, pos_weights, neg_weights)
    else:
        raise NotImplementedError('Loss type not implemented')


def calc_val_metrics(val_loader, model, model_mode, loss_type, confusion=False, bin_threshold=None, skip_associations=False):
    globals.logger.info('\n\033[1mCalculating val metrics...\033[10m')
    total_equalities = 0
    total_absolute_diffs = 0

    all_preds = []
    all_labels = []
    all_image_names = []

    with torch.no_grad():
        for i_batch, batch in enumerate(val_loader):
            image_names = batch['image_name']
            image_batch, targets, labels, _, _ = utility.extract_batch(batch, loss_type)

            logits = model(image_batch)
            val_loss = calc_loss(loss_type, logits, targets).item()
            preds_list, labels_list = utility.logits_to_preds(logits, model_mode, loss_type)[0], utility.tensor_to_list(labels)

            # for confusion matrix
            all_preds.extend(preds_list)
            all_labels.extend(labels_list)
            all_image_names.extend(image_names)

            batch_equalities = calc_metric_for_batch(preds_list, labels_list, metric='equalities', bin_threshold=bin_threshold)
            batch_absolute_diffs = calc_metric_for_batch(preds_list, labels_list, metric='absolute_diff')

            total_equalities += batch_equalities
            total_absolute_diffs += batch_absolute_diffs
            print(f'Done for batch: {i_batch}')

    # calculating association metrics usually takes time
    if not skip_associations:
        spearman, _ = calc_spearman_rank_correlation(all_labels, all_preds)
        kendall, _ = calc_kendall_rank_correlation(all_labels, all_preds)
        print(f'Calculated spearman and kendall for all_labels of len: {len(all_labels)}, all_preds: {len(all_preds)}')
    else:
        spearman = None
        kendall = None
    abs_error_results = calc_class_absolute_error(all_labels, all_preds)  # average over classes, not dominated by majority
    # precision, recall, f1
    low_bin_precision, low_bin_recall, low_bin_f1 = calc_precision_and_recall(all_labels, all_preds, gt_hyper_bin=[0, 1], pred_hyper_bin=[0, 1])
    high_bin_precision, high_bin_recall, high_bin_f1 = calc_precision_and_recall(all_labels, all_preds, gt_hyper_bin=[6, 7], pred_hyper_bin=[6, 7])

    return_dict = {
        **abs_error_results,
        'val_loss': val_loss,
        'spearman': spearman,
        'kendall': kendall,
        'low_bin_precision': low_bin_precision,
        'low_bin_recall': low_bin_recall,
        'low_bin_f1': low_bin_f1,
        'high_bin_precision': high_bin_precision,
        'high_bin_recall': high_bin_recall,
        'high_bin_f1': high_bin_f1,
        'all_image_names': all_image_names,
        'all_preds': all_preds
    }

    if confusion:
        globals.logger.info(f'Calculating confusion matrix for all_pred len: {len(all_preds)}, all_labels len: {len(all_labels)}')
        matrix = confusion_matrix(y_true=all_labels, y_pred=all_preds)
        matrix_normalized = confusion_matrix(y_true=all_labels, y_pred=all_preds, normalize='true')
        return_dict['matrix'] = matrix
        return_dict['matrix_normalized'] = matrix_normalized
    return return_dict
