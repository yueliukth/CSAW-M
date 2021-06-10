import torch
import numpy as np
from copy import deepcopy

import globals


def tensor_to_list(tensor):
    return [t.item() for t in tensor]  # 1d tensor as list


def extract_batch(batch, loss_type):
    # image and label
    image_batch = batch['image'].to(globals.get_current_device())
    labels = batch['label'].to(globals.get_current_device())  # label is always scalar for each image - e.g. 1, 5
    targets = (batch['multi_hot_label'] if loss_type == 'multi_hot' else batch['label']).to(globals.get_current_device())
    # annotator index
    if 'none' in batch['ann_ind']:  # a list of ['none', 'none', ..., 'none']
        ann_inds = None
    else:
        ann_inds = batch['ann_ind'].to(globals.get_current_device())
    # sample weights
    if 'none' in batch['int_cancer_weight']:  # same as above  # todo: change int_cancer_weight name to sample weight
        int_cancer_weights = None
    else:
        int_cancer_weights = batch['int_cancer_weight'].to(globals.get_current_device())
    return image_batch, targets, labels, ann_inds, int_cancer_weights


def scan_thresholded(thresh_row):
    predicted_label = 0
    for ind in range(thresh_row.shape[0]):
        if thresh_row[ind] == 1:
            predicted_label += 1
        else:  # break the first time we see 0
            break
    return predicted_label


def logits_to_preds(logits, model_mode, loss_type):
    with torch.no_grad():
        if loss_type == 'multi_hot':
            probs = torch.sigmoid(logits)
            thresholded = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))  # apply threshold

            preds = []
            batch_size = thresholded.shape[0]
            if model_mode == 'sep_anns':
                for i in range(batch_size):
                    preds_for_image = []
                    for j in range(5):  # for each annotator
                        thresholded_row = thresholded[i, j, :]
                        preds_for_image.append(scan_thresholded(thresholded_row))

                    # median_pred = int(np.median(preds_for_image))  # todo: decide if mean or median or both
                    final_pred = round(np.mean(preds_for_image))
                    preds.append(final_pred)  # final prediction for the image
            else:
                for i in range(batch_size):
                    thresholded_row = thresholded[i, :]
                    predicted_label = scan_thresholded(thresholded_row)
                    preds.append(predicted_label)

        else:
            if model_mode == 'sep_anns':
                probs = torch.softmax(logits, dim=2)  # logits (N, 5, 8)
                all_anns_preds = torch.argmax(probs, dim=2)  # all_anns_preds (N, 5)
                # preds_tensor = torch.median(all_anns_preds, dim=1)[0]  # preds (N)  # todo: decide if mean or median or both
                preds_tensor = torch.round(torch.mean(all_anns_preds.float(), dim=1)).type(torch.int)  # round to int and cast type   # preds (N)
                preds = tensor_to_list(preds_tensor)
            else:
                probs = torch.softmax(logits, dim=1)
                preds_tensor = torch.argmax(probs, dim=1)  # argmax in dim 1 over 8 classes
                preds = [pred.item() for pred in preds_tensor]
        return preds, probs  # preds is 1d list, probs 2d tensor


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


def multi_hot_score_old(pred, probs):
    if pred < 7:
        prob = probs[pred]
        shifted = pred + prob * 2
    elif pred == 7:
        prob = probs[-1]
        shifted = pred + (prob - 0.5) * 2
    else:
        raise NotImplementedError('Score for that prediction not implemented')
    return shifted / 8


def multi_hot_score(cdf_list):
    monotonic = make_monotonic(cdf_list)  # we get list of 7 cdf values
    probs = make_probs(monotonic)
    score = softmax_score(probs)  # now that we have prob for each bin, we can use the nice formula
    return probs, score
