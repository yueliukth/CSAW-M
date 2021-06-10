import os
import glob
import shutil

import yaml
import pandas as pd
import numpy as np

import globals
import helper


def read_params(file=os.path.join('..', 'params.yaml')):
    with open(file) as f:
        params = yaml.safe_load(f)
    return params


def read_file_to_list(filename):
    lines = []
    if os.path.isfile(filename):
        with open(filename) as f:
            lines = f.read().splitlines()
    return lines


def read_csv_to_list(csv_file, exclude_header=True):
    lines = read_file_to_list(csv_file)
    if exclude_header:
        lines = lines[1:]
    return lines


def read_csv_to_bins_list(csv_file):
    lines = read_csv_to_list(csv_file)
    bins_list = [[] for _ in range(8)]
    for line in lines:
        splits = line.split(',')
        filename, label = splits[0], int(splits[1])
        bins_list[label].append(filename)
    return bins_list


def write_bins_to_csv(bins_list, csv_file):
    with open(csv_file, 'w') as file:
        file.write('filename,label\n')
        for i_bin in range(len(bins_list)):
            for filename in bins_list[i_bin]:
                file.write(f'{filename},{i_bin}\n')


def write_list_to_file(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write(f'{item}\n')


def write_as_csv(filepath, header, lines):
    folder = os.path.split(filepath)[0]
    make_dir_if_not_exists(folder)

    with open(filepath, 'w') as f:
        f.write(f'{header}\n')
        for line in lines:
            f.write(f'{line}\n')


def files_with_suffix(directory, suffix, pure=False):
    # files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*{suffix}', recursive=True)]  # full paths
    files = [os.path.abspath(path) for path in glob.glob(os.path.join(directory, '**', f'*{suffix}'), recursive=True)]  # full paths
    if pure:
        files = [os.path.split(file)[-1] for file in files]
    return files


def prepend_to_paths(the_list, string):
    assert string.endswith('/'), 'String should end with /'
    return [f'{string}{filename}' for filename in the_list]


def assert_pure_names(the_list):
    existence_linux_sep = ['/' in filename for filename in the_list]
    existence_windows_sep = ['\\' in filename for filename in the_list]
    assert not any(existence_linux_sep) and not any(existence_windows_sep), f'In [assert_pure_name]: at least one of the filenames is not pure'


def pure_name(file_path):
    if file_path is None:
        return None
    return file_path.split(os.path.sep)[-1]


def pure_names(the_list, sep):
    # todo: could use os.path.split instead, then we do not need to get the sep argument explicitly
    assert type(the_list) is list and type(the_list[0]) is not list, 'Input should be 1d list'
    return [filename.split(sep)[-1] for filename in the_list]


def make_dir_if_not_exists(directory, verbose=True):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        if verbose:
            print(f'In [make_dir_if_not_exists]: created path "{directory}"')  # do not import logger, or you'll get import error


def get_paths(model_name):
    # params = read_params()
    params = globals.params
    return {
        'checkpoints_path': os.path.join(params['train']['checkpoints_path'], model_name)
    }


def parse_log(filepath):
    lines = read_file_to_list(filepath)
    steps, values = [], []
    for line in lines:
        if 'val_per_class_abs_err' in line:
            step = int(line.split(',')[1].split(':')[-1].strip())
            val = float(line.split(':')[-1].strip())
            steps.append(step)
            values.append(val)
    return steps, values


def csv_to_bins(csv_file, bin_column, sep=',', minlength=0):
    df = pd.read_csv(csv_file, sep=sep)
    all_labels = df[bin_column].tolist()
    return np.bincount(all_labels, minlength=minlength)  # minlength sets the min number of bins


def copy_files(source, dest, from_text_file=None, from_csv=None, sep=',', col='filename'):
    assert from_text_file or from_csv, 'Either a csv file or text file should be provided'
    if from_csv:
        # copy files with names specified in the 'filename' column of the csv file from source folder to dest folder
        file_names = pd.read_csv(from_csv, sep=sep)[col].tolist()
    else:
        file_names = helper.read_file_to_list(from_text_file)

    print(f'Found {len(file_names)} filenames in the csv files: {from_csv}')
    helper.make_dir_if_not_exists(dest)

    for i, filename in enumerate(file_names):
        shutil.copy(os.path.join(source, filename), os.path.join(dest, filename))
        if i == 0 or i % 49 == 0 or i == len(file_names) - 1:
            print(f'Done for {i + 1}/{len(file_names)}')
    print(f'Copied all files to: {dest}')
