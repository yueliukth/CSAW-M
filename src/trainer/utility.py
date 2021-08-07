import torch
import numpy as np
from copy import deepcopy

import globals


# --------- utility functions for dealing with tensors and batches ---------
def tensor_to_list(tensor):
    return [t.item() for t in tensor]  # 1d tensor as list


def extract_batch(batch, loss_type):
    image_batch = batch['image'].to(globals.get_current_device())
    if 'none' not in batch['label']:  # if there is 'none' in label, it means there is no label
        labels = batch['label'].to(globals.get_current_device())  # label is always scalar for each image
        # target (used in torch loss function) is the same as label (scalar) for softmax, but is multi_hot_label for multi-hot model
        if loss_type == 'multi_hot':
            targets = batch['multi_hot_label'].to(globals.get_current_device())
        elif loss_type == 'one_hot':
            targets = labels
        else:
            raise NotImplementedError('loss_type not implemented')
    else:
        labels, targets = batch['label'], batch['label']  # list of 'none' values in case there is no label
    return image_batch, labels, targets


# --------- functions for scanning making predictions from one-hot or multi-hot models
def scan_thresholded(thresh_row):
    predicted_label = 0
    for ind in range(thresh_row.shape[0]):  # start scanning from left to right
        if thresh_row[ind] == 1:
            predicted_label += 1
        else:  # break the first time we see 0
            break
    return predicted_label


def logits_to_preds(logits, loss_type):
    with torch.no_grad():
        if loss_type == 'multi_hot':
            probs = torch.sigmoid(logits)
            thresholded = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))  # apply threshold 0.5
            preds = []
            batch_size = thresholded.shape[0]
            for i in range(batch_size):  # for each item in batch
                thresholded_row = thresholded[i, :]  # apply threshold to probabilities to replace floats with either 1's or 0's
                predicted_label = scan_thresholded(thresholded_row)  # scan from left to right and make the final prediction
                preds.append(predicted_label)

        else:  # softamx followed by argmax
            probs = torch.softmax(logits, dim=1)
            preds_tensor = torch.argmax(probs, dim=1)  # argmax in dim 1 over 8 classes
            preds = [pred.item() for pred in preds_tensor]
        return preds, probs  # preds is 1d list, probs 2d tensor


# --------- functions for calculating continuous masking score from bin probabilities ---------
def softmax_score(probs_as_list):
    # score = np.sum([((i + 1) * probs_as_list[i]) for i in range(len(probs_as_list))]) / 8
    score = np.sum([(i * probs_as_list[i]) for i in range(len(probs_as_list))]) / 7
    return score


def is_non_increasing(lst):
    return all(x >= y for x, y in zip(lst, lst[1:]))


def make_monotonic(cdf_list):
    monotonic = []
    for i in range(7):
        max_cdf = max(cdf_list[i:])
        monotonic.append(max_cdf)
    return monotonic


def make_probs(lst):
    extended = deepcopy(lst)
    extended.insert(0, 1)  # cdf: 1 in the beginning
    extended.append(0)  # cdf: 0 at last

    probs = []
    for i in range(0, len(extended) - 1):
        probs.append(extended[i] - extended[i + 1])
    return probs


def multi_hot_score(cdf_list):
    monotonic = make_monotonic(cdf_list)  # we get list of 7 cdf values
    probs = make_probs(monotonic)
    score = softmax_score(probs)  # now that we have prob for each bin, we can use the nice formula
    return probs, score
