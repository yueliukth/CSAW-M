import torch
from PIL import Image
from torchvision import transforms
import cv2
import globals


def make_one_hot(label, n_labels=8):
    one_hot = [0] * n_labels
    one_hot[label] = 1
    return one_hot


def make_multi_hot(label, n_labels=8):
    multi_hot = [0] * (n_labels - 1)
    if label > 0:
        for i in range(label):
            multi_hot[i] = 1
    return torch.tensor(multi_hot, dtype=torch.float32)


def get_transforms(image_size, augments):
    trans_list = []
    # first resize
    if image_size is not None:
        trans_list.append(transforms.Resize(image_size))  # for resize with tuple, torch expects a sequence like (h, w)

    # add other transformations at request
    if augments is not None:
        if 'h_flip' in augments:
            trans_list.append(transforms.RandomHorizontalFlip())
            # globals.logger.info(f'RandomHorizontalFlip added to transformations')

        if 'v_flip' in augments:
            trans_list.append(transforms.RandomVerticalFlip())
            # globals.logger.info(f'RandomVerticalFlip added to transformations')

        if 'rot_10' in augments:
            trans_list.append(transforms.RandomRotation(degrees=10))
            # globals.logger.info(f'RandomRotation with 10 degrees added to transformations')

        if 'rot_15' in augments:
            trans_list.append(transforms.RandomRotation(degrees=15))
            # globals.logger.info(f'RandomRotation with 15 degrees added to transformations')

        if 'color_jitter' in augments:
            jitters = {'brightness': 0.2, 'contrast': 0.2}
            trans_list.append(transforms.ColorJitter(**jitters))
            # globals.logger.info(f'ColorJitter with jitters: {jitters} added to transformations')

    # finally transform to tensor
    trans_list.append(transforms.ToTensor())
    # compose transformations
    trans = transforms.Compose(trans_list)
    return trans


def load_and_preprocess(image_path, img_size, read_mode, augments):
    trans = get_transforms(img_size, augments)
    if read_mode == 1:  # read with PIL
        image = Image.open(image_path)
    else:
        image_array = cv2.imread(image_path)  # read with cv2 as RGB with 3 channels - for images that PIL cannot understand
        image = Image.fromarray(image_array)  # convert to PIL image
    return image, trans(image)
