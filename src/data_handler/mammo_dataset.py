from torch.utils import data

import os
from . import train_preprocessing
import helper
import globals


class MammoDataset(data.Dataset):
    def __init__(self, mode, data_folder, img_size, augments=None, data_list=None,
                 line_parse_type=None, files_suffix='.png', imread_mode=1, exclude_list=None, csv_sep_type=1):
        self.mode = mode   # mode 'test' does not require all arguments
        self.image_names = helper.files_with_suffix(data_folder, files_suffix, pure=True)  # used in test mode only (otherwise names are inferred from data_list)
        self.exclude_list = exclude_list
        self.data_folder = data_folder
        self.data_list = data_list
        self.line_parse_type = line_parse_type
        self.imread_mode = imread_mode
        self.augments = augments
        self.img_size = img_size
        self.csv_sep_char = ',' if csv_sep_type == 1 else ';'
        globals.logger.info(f'MammoDataset created with image size: {img_size}, imread_mode: {imread_mode}\n')

        if self.exclude_list is not None:
            self.do_exclusion()

    def do_exclusion(self):
        globals.logger.info(f'image_names before exclusion has len: {len(self.image_names)}')
        self.image_names = [name for name in self.image_names if name not in self.exclude_list]
        globals.logger.info(f'image_names reduced to len: {len(self.image_names)}\n')

    def parse_line(self, data_line):
        if self.line_parse_type == 1:
            image_name, label = data_line.split(self.csv_sep_char)
            ann_ind, int_cancer_weight = 'none', 'none'  # indicates no index - we cannot directly use None, torch does not allow it
        elif self.line_parse_type == 2:
            image_name, label, _, ann_ind = data_line.split(self.csv_sep_char)  # ann_ind: annotator index, todo: convert to int
            int_cancer_weight = 'none'
        elif self.line_parse_type == 3:  # 3
            image_name, label, int_cancer_weight = data_line.split(self.csv_sep_char)  # todo: change name of int_cancer_weight, convert to int
            ann_ind = 'none'
        else:  # 4
            image_name, label, _, _ = data_line.split(self.csv_sep_char)
            ann_ind, int_cancer_weight = 'none', 'none'   # ann_ind could be obtained from data_line if needed
        return image_name, int(label), ann_ind, int_cancer_weight  # convert str to int

    def make_image_tensor(self, image_name):
        image_path = os.path.join(self.data_folder, image_name)  # make full path
        return train_preprocessing.load_and_preprocess(image_path, self.img_size, self.imread_mode, self.augments)[1]

    def __len__(self):
        if self.mode == 'test':
            return len(self.image_names)
        return len(self.data_list)

    def __getitem__(self, index):
        if self.mode == 'test':
            image_name = self.image_names[index]
            image_tensor = self.make_image_tensor(image_name)
            return {
                'image': image_tensor,
                'image_name': image_name
            }

        else:
            data_line = self.data_list[index]
            image_name, label, ann_ind, int_cancer_weight = self.parse_line(data_line)

            # image_path = os.path.join(self.data_folder, image_name)  # make full path
            # image_tensor = train_preprocessing.load_and_preprocess(image_path)
            image_tensor = self.make_image_tensor(image_name)
            one_hot_label = train_preprocessing.make_one_hot(label)
            multi_hot_label = train_preprocessing.make_multi_hot(label)

            return {
                'image': image_tensor,
                'label': label,
                'ann_ind': ann_ind,
                'int_cancer_weight': int_cancer_weight,
                'one_hot_label': one_hot_label,
                'multi_hot_label': multi_hot_label,
                'image_name': image_name
            }


def init_data_loader(dataset_mode, data_folder, data_list, line_parse_type, imread_mode,
                     n_workers, batch_size, img_size, shuffle, exclude_list=None, augments=None, csv_sep_type=1):
    """
    :param img_size:
    :param augments:
    :param dataset_mode: if 'test', no need to provide a data_list, for 'train' and 'val' we should have a data list from which we can extract true labels.
    :param data_folder: -
    :param data_list: a list containing data records, filename and label separated by a comma
    :param line_parse_type: how the data_list should be parsed, it is used for data_lists that have labels for different annotators
                            (as opposed to one tru label). See code for mode details.
    :param imread_mode: if we have usual 8-bit images, use imread_mode will be 1, meaning that the dataset will use PIL to read the image.
                        If we have 16-bit images that PIL cannot read, use imread_mode 2, in which case the image is first read by cv2 and then converted to a PIL Image.
    :param n_workers: -
    :param batch_size: -
    :param shuffle: -
    :param exclude_list: a list of filenames that should be excluded from the list of image names that the dataset uses to read data
                        (mainly used for excluding some images when doing inference).
    :return: the data loader
    """
    globals.logger.info(f'Augmentations are: {augments}')
    dataset = MammoDataset(mode=dataset_mode,
                           data_folder=data_folder,
                           img_size=img_size,
                           data_list=data_list,
                           line_parse_type=line_parse_type,
                           imread_mode=imread_mode,
                           exclude_list=exclude_list,
                           augments=augments,
                           csv_sep_type=csv_sep_type)
    loader = data.DataLoader(dataset=dataset, num_workers=n_workers, batch_size=batch_size, shuffle=shuffle)
    return loader

