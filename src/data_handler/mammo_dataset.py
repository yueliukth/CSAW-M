from torch.utils import data

import os
from . import train_preprocessing
import helper
import globals


class MammoDataset(data.Dataset):
    def __init__(self, data_folder, img_size, imread_mode, line_parse_type, csv_sep_type, data_list=None, augments=None):
        self.data_folder = data_folder
        self.img_size = img_size
        self.data_list = data_list
        self.augments = augments
        self.line_parse_type = line_parse_type
        self.imread_mode = imread_mode
        self.csv_sep_char = ',' if csv_sep_type == 1 else ';'
        globals.logger.info(f'MammoDataset created with image size: {img_size}, imread_mode: {imread_mode}\n')

        # make data_list if not provided
        if self.data_list is None:
            self.data_list = helper.files_with_suffix(self.data_folder, suffix='.png', pure=True)
            globals.logger.info(f'Manually initialized data_list with all {len(self.data_list)} png images in data_folder')


    def parse_line(self, data_line):
        if self.line_parse_type == 1:
            image_name, label = data_line.split(self.csv_sep_char)[:2]
            label = int(label)
        elif self.line_parse_type == 0:  # data_line is the actual pure filename
            image_name, label = data_line, 'none'
        else:
            raise NotImplementedError('line_parse_type not implemented')
        return image_name, label

    def make_image_tensor(self, image_name):
        image_path = os.path.join(self.data_folder, image_name)  # make full path
        return train_preprocessing.load_and_preprocess(image_path, self.img_size, self.imread_mode, self.augments)[1]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_line = self.data_list[index]
        image_name, label = self.parse_line(data_line)
        image_tensor = self.make_image_tensor(image_name)

        if label != 'none':
            label = train_preprocessing.convert_label(label, direction='to_train')  # convert scalar labels from [1-8] to [1-7]
            multi_hot_label = train_preprocessing.make_multi_hot(label)
        else:
            multi_hot_label = 'none'

        return {
            'image': image_tensor,
            'label': label,
            'multi_hot_label': multi_hot_label,
            'image_name': image_name
        }


def init_data_loader(dataset_params, data_loader_params):
    if 'augments' in dataset_params.keys():  # val dataset params does not have 'augments'
        globals.logger.info(f'Augmentations are: {dataset_params["augments"]}')
    dataset = MammoDataset(**dataset_params)
    loader = data.DataLoader(dataset=dataset, **data_loader_params)
    return loader

