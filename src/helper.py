import glob
import shutil
import yaml
import pandas as pd
import numpy as np
import os
import torch
import sys
import logging

import globals


# ----------- functions related to file I/O -----------
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


def pure_name(file_path):
    if file_path is None:
        return None
    return file_path.split(os.path.sep)[-1]


def pure_names_list(the_list, sep):
    # note: could use os.path.split instead, then we do not need to get the sep argument explicitly
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


def copy_files(source, dest, from_text_file=None, from_csv=None, sep=',', col='filename'):
    assert from_text_file or from_csv, 'Either a csv file or text file should be provided'
    if from_csv:
        # copy files with names specified in the 'filename' column of the csv file from source folder to dest folder
        file_names = pd.read_csv(from_csv, sep=sep)[col].tolist()
    else:
        file_names = read_file_to_list(from_text_file)

    print(f'Found {len(file_names)} filenames in the csv files: {from_csv}')
    make_dir_if_not_exists(dest)

    for i, filename in enumerate(file_names):
        shutil.copy(os.path.join(source, filename), os.path.join(dest, filename))
        if i == 0 or i % 49 == 0 or i == len(file_names) - 1:
            print(f'Done for {i + 1}/{len(file_names)}')
    print(f'Copied all files to: {dest}')


# ----------- generic helper functions -----------
def replace_all(lst, old_str, new_str):
    return [file.replace(old_str, new_str) for file in lst]


def get_logger(logger_name, include_datetime, pure_line=False):
    # creating logger
    root = logging.getLogger(logger_name)  # implements the singleton pattern in itself
    root.setLevel(logging.INFO)  # handles level INFO and above

    if include_datetime:
        message_format = "[%(levelname)s] [%(asctime)s] [%(filename)s line %(lineno)d] %(message)s"  # also get the function name
        datetime_format = "%Y-%m-%d %H:%M:%S"
    else:
        if pure_line:
            message_format = "%(message)s"  # only message without extra prints
            datetime_format = None
        else:
            message_format = "[%(filename)s line %(lineno)d] %(message)s"  # print filename, lin number, as well the message
            datetime_format = None

    # creating handler for outputting to stderr on console
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)  # handles INFO and above
    stream_handler.setFormatter(logging.Formatter(message_format, datetime_format))
    root.addHandler(stream_handler)
    return root


def waited_print(string):
    print(string)
    print('====== Waiting for input')
    input()


def show_as_csv(array):
    for row in array:
        str_list = [str(value) for value in row]
        print(','.join(str_list))


def as_str(the_list, sep):
    return sep.join([str(elem) for elem in the_list])


# ----------- functions related to imaging -----------
def unnormalize_image(image):
    image = image * 255
    image = np.rint(image).astype(np.uint8)
    return image


def remove_alpha_channel(image_or_image_batch):
    if len(image_or_image_batch.shape) == 4:  # with batch size
        return image_or_image_batch[:, 0:3, :, :]
    return image_or_image_batch[0:3, :, :]  # single image


def get_pil_image_size(pil_image):
    # Note: pil_image.size only returns the 2d size and does not show the channel size, so we should convert it to np array first
    return np.array(pil_image).shape


# ----------- utility functions related mostly related to torch models -----------
def show_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    globals.logger.info(f'Model total params: {total_params:,} - trainable params: {trainable_params:,}')


def save_checkpoint(path_to_save, step, model, optimizer, loss, lr):
    name = os.path.join(path_to_save, f'step={step}.pt')
    checkpoint = {'loss': loss,
                  'lr': lr,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    make_dir_if_not_exists(path_to_save, verbose=True)
    torch.save(checkpoint, name)
    globals.logger.info(f'Save state dict done at: "{name}"\n')


def load_checkpoint(path_to_load, step, model, optimizer=None, resume_train=True):
    name = os.path.join(path_to_load, f'step={step}.pt')
    checkpoint = torch.load(name, map_location=globals.get_current_device())
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = checkpoint['loss']
    lr = checkpoint['lr']
    globals.logger.info(f'In [load_checkpoint]: load state dict done from: "{name}"\n')

    # putting the model in the correct mode
    if resume_train:
        model.train()
    else:
        model.eval()
        for param in model.parameters():  # freezing the layers when using only for evaluation
            param.requires_grad = False
    return model.to(globals.get_current_device()), optimizer, loss, lr  # returned optimizer is None if not provided
