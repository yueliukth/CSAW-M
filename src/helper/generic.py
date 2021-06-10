import os
import numpy as np
import sys
import logging


def total_len(the_list):
    return np.sum([len(lst) for lst in the_list])


def list_lens(the_list):  # lens of a 2d list
    return [len(lst) for lst in the_list]


def flattened(the_list):
    flattened_list = []
    for one_dim_list in the_list:
        flattened_list.extend(one_dim_list)
    return flattened_list


def replace_all(lst, old_str, new_str):
    return [file.replace(old_str, new_str) for file in lst]


def get_logger(logger_name, log_file, error_file, include_datetime, pure_line=False):
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
            message_format = "[%(filename)s line %(lineno)d] %(message)s"
            datetime_format = None

    # creating handler for outputting to stderr on console
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)  # handles INFO and above
    stream_handler.setFormatter(logging.Formatter(message_format, datetime_format))
    root.addHandler(stream_handler)

    # handler for outputting to the log file (if provided)
    if log_file is not None:
        # if log_file contains a path in it (e.g. /path/to/log.txt) as opposed to log.txt
        log_dirname = os.path.dirname(log_file)  # could be parent directory or '' if the file does not contain any path in it
        if log_dirname != '' and not os.path.isdir(log_dirname):
            os.makedirs(log_dirname)  # create parent directory if not exists

        log_file_handler = logging.FileHandler(filename=log_file)
        log_file_handler.setLevel(logging.INFO)
        log_file_handler.setFormatter(logging.Formatter(message_format, datetime_format))
        root.addHandler(log_file_handler)

    # handler for outputting to the err file (if provided)
    if error_file is not None:
        err_dirname = os.path.dirname(error_file)
        if err_dirname != '' and not os.path.isdir(err_dirname):
            os.makedirs(err_dirname)

        err_file_handler = logging.FileHandler(filename=error_file)
        err_file_handler.setLevel(logging.WARNING)  # handles level WARNING and above
        err_file_handler.setFormatter(logging.Formatter(message_format, datetime_format))
        root.addHandler(err_file_handler)
    return root


def waited_print(string):
    print(string)
    print('====== Waiting for input')
    input()
