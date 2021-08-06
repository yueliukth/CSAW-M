import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn

import globals
import helper


def init_model(loss_type):
    fc_out_channels = 8 if loss_type == 'one_hot' else 7
    predictor = Predictor(fc_out_channels=fc_out_channels)
    globals.logger.info(f'\033[1mModel initialized\033[0m\n')
    helper.show_num_params(predictor)
    globals.logger.info(f'Model is now being moved to device: {globals.get_current_device()}\n')
    return predictor.to(globals.get_current_device())


def init_and_load_model_for_eval(model_name, loss_type, step, checkpoints_path=None):
    globals.logger.info(f'Initializing and loading model for evaluation...')
    model = init_model(loss_type)  # always init with no pretraining because then we will load a checkpoint
    if checkpoints_path is None:  # meaning that set_globals has been called in the beginning of the program
        path = helper.get_paths(model_name)['checkpoints_path']
    else:  # otherwise the folder containing checkpoints is provided
        path = checkpoints_path
    model = helper.load_checkpoint(path, step, model, resume_train=False)[0]  # load mode in eval mode
    return model


def load_resnet(model_name, pretrained, freeze_params, n_layers_to_remove, verbose=False):
    """
    Example from https://pytorch.org/docs/stable/torchvision/models.html
    and https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.

    :param n_layers_to_remove:
    :param model_name:
    :param pretrained:
    :param freeze_params: if True, parameters of the resnet will be frozen and do not contribute to gradient updates.
    :param verbose: if True, the function prints model summaries before and after removing the two last layers.
    :return: the pre-trained resnet34 model with the two last layers (pooling and fully connected) removed.

    Notes:
        - For 256x256 images:
            - Output shape of the forward pass is [512 x 8 x 8] excluding the batch size for resnet34.
            - Output shape of the forward pass is [2048 x 8 x 8] excluding the batch size for resnet50.
    """
    globals.logger.info(f'Loading {model_name}, pretrained={pretrained}')
    if model_name == 'resnet18':
        resnet_model = models.resnet18(pretrained)
    elif model_name == 'resnet34':
        resnet_model = models.resnet34(pretrained)
    else:
        # resnet_model = models.resnet50(pretrained)
        raise NotImplementedError('Need to fix spatial dimension')
    resnet_model.train()  # put the model in the correct mode

    if verbose:
        globals.logger.info('Loaded ResNet')
        globals.logger.info('ResNet summary before removing the last two layers')
        summary(resnet_model, input_size=(3, 316, 256))
        helper.waited_print('')

    # removing last layer(s)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-n_layers_to_remove])

    if verbose:
        globals.logger.info('ResNet summary after the last two layers')
        helper.show_num_params(resnet_model)
        summary(resnet_model, input_size=(3, 316, 256))  # feature maps of shape (512, 10, 8) for resnet18 and 34
        helper.waited_print('')

    # freeze the resnet
    if freeze_params:
        for param in resnet_model.parameters():  # freezing the parameters
            param.requires_grad = False
    return resnet_model


class Predictor(torch.nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True, if_freezed=False, fc_in_channels=512, fc_out_channels=8):
        super().__init__()
        self.backbone = load_resnet(backbone_name, pretrained=pretrained, freeze_params=if_freezed, n_layers_to_remove=1)  # resnet with average pooling
        self.fc_layer = nn.Linear(in_features=fc_in_channels, out_features=fc_out_channels)

    def forward(self, inp):
        backbone_out = self.backbone(inp).squeeze(3).squeeze(2)  # (N, C, 1, 1) -> (N, C)
        model_out = self.fc_layer(backbone_out)
        return model_out
