from PIL import Image
import pydicom
import numpy as np
from matplotlib import cm


def read_dicom(filepath, left_oriented=True, for_vis=False, resize_factor=None):
    import helper
    helper.waited_print('NOTE: DO NOT USE THIS ANYMORE, WRITE FUNCTION THAT USES CV2 - ALSO WE DO NOT USE BONE COLOR SPACE ANYMORE')
    dataset = pydicom.dcmread(filepath)
    pixels = dataset.pixel_array

    if left_oriented:
        orientation = str(dataset.get('PatientOrientation', "(missing)"))
        if 'A' in orientation:  # anterior view, should be flipped
            pixels = np.flip(pixels, axis=1)

    if not for_vis:
        return pixels  # pixels represented by 12 bits

    else:
        assert resize_factor is not None
        pixels = pixels / np.max(pixels)  # normalize to 0-1
        image = Image.fromarray(np.uint8(cm.bone(pixels) * 255))
        if resize_factor > 1:
            image = image.resize((pixels.shape[1] // resize_factor, pixels.shape[0] // resize_factor))
        return image


def unnormalize_image(image):
    image = image * 255
    image = np.rint(image).astype(np.uint8)
    return image


def remove_alpha_channel(image_or_image_batch):
    if len(image_or_image_batch.shape) == 4:  # with batch size
        return image_or_image_batch[:, 0:3, :, :]
    return image_or_image_batch[0:3, :, :]  # single image


def get_pil_image_size(pil_image):
    # Note: pil_image.size only returns the 2d size and does not show the channel size, so we should convert it to np array first
    return np.array(pil_image).shape
