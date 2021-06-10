import torch
import helper

# global logger accessible to all modules
logger = None
params = None  # global params after reading the params yaml file, declared here so the changes to params based on args is visible to all modules


def setup_logger(logger_name='logger', log_file=None, error_file=None, include_datetime=False, pure_line=False):
    global logger
    logger = helper.get_logger(logger_name, log_file, error_file, include_datetime, pure_line=pure_line)
    handlers_as_str = '\n'.join(map(str, logger.handlers))
    logger.info(f"Set up logger done with name '{logger.name}' and handlers: \n{handlers_as_str}\n")


def set_current_device(device_id):
    try:
        torch.cuda.set_device(device_id)
        logger.info(f'Set device with id: {device_id}')
    except AttributeError:  # running on CPU
        logger.info('Using device: CPU')


def get_current_device():
    try:
        device_id = torch.cuda.current_device()
        current_device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    except AssertionError:  # in case it raises "AssertionError: Torch not compiled with CUDA enabled" -> use CPU
        current_device = torch.device('cpu')
    return current_device
