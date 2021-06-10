import os

import globals
import torch
from .file_io import make_dir_if_not_exists


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
