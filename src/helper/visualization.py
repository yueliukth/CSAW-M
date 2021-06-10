import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from PIL import Image

import helper
import globals


def visualize_csv(csv_file, data_folder, out_path, files_suffix, filename_col_ind, label_col_ind, selected_bins=np.arange(7)):
    lines = helper.read_csv_to_list(csv_file)
    globals.logger.info(f'Reading {len(lines)} lines from: {csv_file}: done')

    for i_line, line in enumerate(lines):
        splits = line.split(',')
        filename, i_bin = splits[filename_col_ind], splits[label_col_ind]

        if files_suffix == '.dcm':  # files_suffix indicates if the filenames in the csv file end with .dcm, in which case the name should be replaced
            filename = filename.replace('.dcm', '.png')

        if int(i_bin) not in selected_bins:  # only for selected bins
            continue

        out_bin_path = os.path.join(out_path, f'bin_{i_bin}')
        source_filepath = os.path.join(data_folder, filename)
        dest_filepath = os.path.join(out_bin_path, filename)

        helper.make_dir_if_not_exists(out_bin_path)
        shutil.copyfile(source_filepath, dest_filepath)
        if i_line % 50 == 0:
            globals.logger.info(f'Done for file {i_line}')


# function for displaying image arrays
def vis_array(img):
    Image.fromarray(img).show()
    helper.waited_print('')


def show_as_csv(array):
    for row in array:
        str_list = [str(value) for value in row]
        print(','.join(str_list))


def draw_bars_from_list(the_list, title=None, x_ticks_labels=None, y_label=None):
    plt.style.use('seaborn-whitegrid')
    xs = np.arange(len(the_list))
    ys = the_list
    plt.bar(xs, ys)

    if title is not None:
        plt.title(title)
    if x_ticks_labels is not None:
        plt.xticks(ticks=xs, labels=x_ticks_labels)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()


def average_curve(steps, means, stds, title, xlabel, ylabel, min_loc):
    plt.style.use('seaborn-whitegrid')
    # xs = np.arange(len(means))

    plt.plot(steps, means, '-', color='black')
    plt.fill_between(steps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color='gray', alpha=0.2)
    plt.plot(min_loc[0], min_loc[1], 'or')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def draw_distribution_from_csv(csv_file, desired_col, bin_or_score, save_path=None, sep=','):
    ys = pd.read_csv(csv_file, sep)[desired_col].tolist()
    if bin_or_score == 'score':
        plt.hist(ys, density=True, bins=30)  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data')
    else:
        ys = np.bincount(ys)  # count repetitions per bin
        labels = [f'bin_{i}' for i in range(8)]
        xs = np.arange(len(labels))
        # width = 1
        plt.bar(xs, ys, align='center')
        plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
        plt.yticks(ys)
    # save
    plt.savefig(save_path)
    print(f'Saved to: {save_path}')
