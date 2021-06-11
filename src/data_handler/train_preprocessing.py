import torch
from PIL import Image
from torchvision import transforms
import cv2


def convert_label(label, direction):
    """
    When preparing the label for training, we convert each label in range [1-8] to [0-7] since torch expect the labels
    to be in range [0-7] for 8 classes. We use the same converted label for producing multi-hot encoding.
    When evaluation, we convert back the prediction labels in [0-7] to be in [1-8].
    """
    if direction == 'to_train':
        return label - 1
    elif direction == 'from_train':  # 'from_train'
        return label + 1
    else:
        raise NotImplementedError('Direction for converting labels not implemented')


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

        if 'v_flip' in augments:
            trans_list.append(transforms.RandomVerticalFlip())

        if 'rot_10' in augments:
            trans_list.append(transforms.RandomRotation(degrees=10))

        if 'rot_15' in augments:
            trans_list.append(transforms.RandomRotation(degrees=15))

        if 'color_jitter' in augments:
            jitters = {'brightness': 0.2, 'contrast': 0.2}
            trans_list.append(transforms.ColorJitter(**jitters))

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
