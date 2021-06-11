from torch.utils import data

import os
from . import train_preprocessing
import helper
import globals


class MammoDataset(data.Dataset):
    def __init__(self, data_folder, img_size, data_list=None, augments=None, line_parse_type=1, imread_mode=2, csv_sep_type=2):
        self.data_folder = data_folder
        self.img_size = img_size
        self.data_list = data_list
        self.augments = augments
        self.line_parse_type = line_parse_type
        self.imread_mode = imread_mode
        self.csv_sep_char = ',' if csv_sep_type == 1 else ';'
        globals.logger.info(f'MammoDataset created with image size: {img_size}, imread_mode: {imread_mode}\n')

    def parse_line(self, data_line):
        if self.line_parse_type == 1:
            image_name, label = data_line.split(self.csv_sep_char)[:2]
        else:
            raise NotImplementedError('line_parse_type not implemented')
        return image_name, int(label)

    def make_image_tensor(self, image_name):
        image_path = os.path.join(self.data_folder, image_name)  # make full path
        return train_preprocessing.load_and_preprocess(image_path, self.img_size, self.imread_mode, self.augments)[1]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_line = self.data_list[index]
        image_name, label = self.parse_line(data_line)
        image_tensor = self.make_image_tensor(image_name)
        label = train_preprocessing.convert_label(label, direction='to_train')  # convert scalar labels from [1-8] to [1-7]
        multi_hot_label = train_preprocessing.make_multi_hot(label)
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

