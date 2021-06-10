import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

import globals
import helper


def create_model_with_conf(model_name, mode, loss_type, pretraining, checkpoint_file=None):
    config = {
        'mode': mode,
        'backbone_name': 'resnet34',
        'pretraining': pretraining,
        'freezed': False,
        'fc_in_channels': 512,
        'fc_out_channels': 7 if loss_type == 'multi_hot' else 8,
        'checkpoint_file': checkpoint_file
    }
    globals.logger.info(f'Using config: \n{config}\n')
    return Predictor(**config)


def init_model(model_name, mode, loss_type, pretraining, checkpoint_file=None):
    """
    :param checkpoint_file:
    :param pretraining:
    :param model_name: name of model
    :param mode: 'usual' or 'sep_anns'
    :param loss_type: 'one_hot' or 'multi_hot'
    :return: model
    """
    predictor = create_model_with_conf(model_name, mode, loss_type, pretraining, checkpoint_file)
    # globals.logger.info(f'Model initialized - Model paths: {helper.get_paths(model_name)}\n')
    globals.logger.info(f'\033[1mModel initialized\033[0m\n')
    helper.show_num_params(predictor)
    globals.logger.info(f'Model is now being moved to device: {globals.get_current_device()}\n')
    return predictor.to(globals.get_current_device())


def init_and_load_model_for_eval(model_name, mode, loss_type, step, checkpoints_path=None):
    """
    :param model_name: see init_model documentation.
    :param mode: see init_model documentation.
    :param loss_type: see init_model documentation.
    :param step: -
    :param checkpoints_path: If provided, the checkpoint will be loaded from that path, otherwise the default checkpoints path
    that is in params.yml file will be used.
    :return: model
    """
    globals.logger.info(f'Initializing and loading model for evaluation...')
    model = init_model(model_name, mode, loss_type, pretraining='none')  # always init with no pretraining because then we will load a checkpoint
    if checkpoints_path is None:
        path = helper.get_paths(model_name)['checkpoints_path']
    else:
        path = checkpoints_path
    model = helper.load_checkpoint(path, step, model, resume_train=False)[0]  # load mode in eval mode
    return model


def check_backward_compat(model_name, mode, loss_type, step):
    model = init_and_load_model_for_eval(model_name, mode, loss_type, step)
    rand_tensor = torch.randn((10, 3, 256, 256), device=globals.get_current_device())
    model_out = model(rand_tensor)
    globals.logger.info('******** Model created and checkpoint loaded successfully ==> backward compatibility: OK **********\n\n')


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
    globals.logger.info(f'Loading ResNet, pretrained={pretrained}')
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
    def __init__(self, mode, backbone_name, pretraining, freezed, fc_in_channels, fc_out_channels, checkpoint_file=None):
        """
        :param mode: now could be 'usual' or 'sep_ann', the latter denoting learning different heads for different annotators.
        :param backbone_name:
        :param pretraining: could be 'image_net', 'moco', or 'none', which indicates ImageNet pretraining, MoCo pretraining, or no pretraining respectively.
        :param freezed:
        :param fc_in_channels:
        :param fc_out_channels:
        :param checkpoint_file: the checkpoint needed for loading MoCo encoder weights
        """
        super().__init__()
        if pretraining == 'moco':
            # self.backbone = load_moco_encoder(arch=backbone_name, checkpoint_file=checkpoint_file)
            raise NotImplementedError('This pre-training not implemented')
        else:
            pretrained = True if pretraining == 'image_net' else False
            self.backbone = load_resnet(backbone_name, pretrained=pretrained, freeze_params=freezed, n_layers_to_remove=1)  # resnet with average pooling
        self.mode = mode

        if mode == 'sep_anns':  # five fully connected layers for each annotator
            self.fc_layer1 = nn.Linear(fc_in_channels, fc_out_channels)
            self.fc_layer2 = nn.Linear(fc_in_channels, fc_out_channels)
            self.fc_layer3 = nn.Linear(fc_in_channels, fc_out_channels)
            self.fc_layer4 = nn.Linear(fc_in_channels, fc_out_channels)
            self.fc_layer5 = nn.Linear(fc_in_channels, fc_out_channels)
        else:
            self.fc_layer = nn.Linear(in_features=fc_in_channels, out_features=fc_out_channels)

    def forward(self, inp, ann_inds=None):
        backbone_out = self.backbone(inp).squeeze(3).squeeze(2)  # (N, C, 1, 1) -> (N, C)

        if self.mode == 'sep_anns':
            fc_out1 = self.fc_layer1(backbone_out)  # (N, 8) for softmax
            fc_out2 = self.fc_layer2(backbone_out)
            fc_out3 = self.fc_layer3(backbone_out)
            fc_out4 = self.fc_layer4(backbone_out)
            fc_out5 = self.fc_layer5(backbone_out)
            all_outs = torch.stack([fc_out1, fc_out2, fc_out3, fc_out4, fc_out5], dim=1)  # (N, 5, 8) or (N, 5, 7)

            if ann_inds is None:  # return all the logits from all doctors
                model_out = all_outs  # (N, 5, 8)
            else:
                select_outs = [all_outs[i, ann_inds[i], :] for i in range(inp.shape[0])]  # list of tensors of shape (8) or (7)
                model_out = torch.stack(select_outs, dim=0)  # list -> (N, 8)
        else:
            model_out = self.fc_layer(backbone_out)
        return model_out
