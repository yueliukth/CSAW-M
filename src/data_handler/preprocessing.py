import os
import cv2
import numpy as np
from skimage import exposure
import pandas as pd
from PIL import Image
import multiprocessing
import pydicom
import traceback

import helper


# function to remove texts in the image
# only breast remained
def segment_breast(img, low_int_threshold=0.05):
    # create img for thresholding and contours
    img_8u = (img.astype('float32') / img.max() * 255).astype('uint8')
    if low_int_threshold < 1:
        low_th = int(img_8u.max() * low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)
    breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1)
    
    # segment the breast
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
    x, y, w, h = cv2.boundingRect(contours[idx])
    img_breast_only = img_breast_only[y:y+h, x:x+w]
    return img_breast_only, (x, y, w, h)


# function to locate the center of mass
def new_cropping_single_dist(img):
    # org_img = img
    opening_it = 5
    kernel_size = (25, 25)
    kernel = np.ones(kernel_size, np.uint8)
    _, img = cv2.threshold(img, img.min(), img.max(), cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, kernel_size, 0)
    img = cv2.dilate(img, kernel, 5)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opening_it)
    opening = cv2.copyMakeBorder(opening, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=0)
    opening = opening.astype(np.uint8)
    
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, kernel_size, 0) 
    maxLoc = cv2.minMaxLoc(dist)[-1]
    return maxLoc


# function to add padding or perform cropping
# to make the center of mass be the center of the image
def pad_or_crop_single_maxloc(img, maxloc, full_height=632, full_width=512, if_movey=False):
    min_pixel_value = np.min(img)
    w = img.shape[1]
    h = img.shape[0]
    if maxloc[0] >= w-maxloc[0]:
        img_new = np.full((h,2*maxloc[0]), min_pixel_value)
        img_new[:, :w] = img
    elif maxloc[0] < w-maxloc[0]:
        img_new = np.full((h,2*(w-maxloc[0])), min_pixel_value)
        img_new[:, 2*(w-maxloc[0])-w:] = img
    img = img_new
    
    if if_movey:
        w = img_new.shape[1]
        h = img_new.shape[0]
        if maxloc[1] >= h-maxloc[1]:
            img_new = np.full((2*maxloc[1], w), min_pixel_value)
            img_new[:h, :] = img
        elif maxloc[1] < h-maxloc[1]:
            img_new = np.full((2*(h-maxloc[1]), w), min_pixel_value)
            img_new[2*(h-maxloc[1])-h:, :] = img
        img = img_new
    
    w = img.shape[1]
    h = img.shape[0]
    if h > full_height:
        img = img[int((h - full_height) / 2):int((h - full_height) / 2) + full_height, :]
    if w > full_width:
        img = img[:, int((w - full_width) / 2):int((w - full_width) / 2) + full_width]
    img_new = np.full((full_height, full_width), min_pixel_value)
    img_new[int((img_new.shape[0]-img.shape[0])/2):int((img_new.shape[0]-img.shape[0])/2)+img.shape[0], int((img_new.shape[1]-img.shape[1])/2):int((img_new.shape[1]-img.shape[1])/2)+img.shape[1]] = img
    return img_new


def raw_to_preprocessed(image_folder, labels_path, save_dir, if_movey=False):
    # read the csv file as df
    df = pd.read_csv(labels_path, delimiter=';', dtype={'sourcefile': str})

    n_images = len(helper.files_with_suffix(image_folder, suffix='.png'))
    print(f'Found {n_images} images in: {image_folder} and {len(df)} rows in: {labels_path}')

    # preprocess images on by one
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        filename = row['Filename']
        dicom_imagelaterality = row['Dicom_image_laterality']
        dicom_windowcenter = row['Dicom_window_center']
        dicom_windowwidth = row['Dicom_window_width']
        dicom_photometricinterpretation = row['Dicom_photometric_interpretation']
        
        img_path = os.path.join(image_folder, filename)
        img_array = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)  # read the 16-bit PNG as is

        # flip the image if necessary to make all breasts left-posed
        if dicom_imagelaterality == 'R':
            img_array = cv2.flip(img_array, 1)
    
        # intensity rescaling according to window center and window width
        img_array = exposure.rescale_intensity(img_array, in_range=(dicom_windowcenter - dicom_windowwidth / 2,
                                                                    dicom_windowcenter + dicom_windowwidth / 2))
        # invert the color if needed
        if dicom_photometricinterpretation == 'MONOCHROME1':
            img_array = cv2.bitwise_not(img_array)
        
        # segment the image so that only breast remained
        img_array, _ = segment_breast(img_array)

        # crop with distance transform and pad
        max_loc = new_cropping_single_dist(img_array)
        new_img = pad_or_crop_single_maxloc(img_array, max_loc, if_movey=if_movey)

        new_filepath = os.path.join(save_dir, filename)
        if not os.path.exists(os.path.dirname(new_filepath)):
            os.makedirs(os.path.dirname(new_filepath))
            print(f'Created folder: {os.path.dirname(new_filepath)}')
            
        cv2.imwrite(new_filepath, new_img.astype(np.uint16))
        print(f'Image saved to: \n{new_filepath}')
        print(f'Done for image {i + 1}/{df.shape[0]}\n')


def _save_as_png(dicom_folder, dicom_basenames, png_folder, image_size, reduce_bits=False):
    for i, dicom_basename in enumerate(dicom_basenames):
        dicom_filepath = os.path.join(dicom_folder, dicom_basename)
        png_basename = dicom_basename.replace('.dcm', '.png')
        new_path = os.path.join(png_folder, png_basename)

        try:
            img_dcm = pydicom.dcmread(dicom_filepath)
            img_array = img_dcm.pixel_array
        except:
            message = traceback.format_exc()  # get full traceback
            print(f'A problem occurred when reading: {dicom_filepath}: \n{message}')
            exit(1)

        # save images
        if reduce_bits:  # this reduces the quality of the PNG image - it has not been used in the project
            img_array = img_array / (2 ** 16 - 1)
            img_array = helper.unnormalize_image(img_array)  # reduce to 8 bits, using np.astype(np.uint8) destroys the image
            Image.fromarray(img_array).convert('RGB').resize(image_size).save(new_path)  # save as RGB not gray
        else:
            cv2.imwrite(new_path, cv2.resize(img_array.astype(np.uint16), image_size))

        print(f'Image saved to: \n{new_path}')
        print(f'Done for image {i + 1}/{len(dicom_basenames)}\n')


def dicoms_to_raw_pngs(dicom_folder, dicom_basenames, png_folder, image_size, reduce_bits=False, n_processes=1):
    # this is the main functions that is used to preprocess all images to png (supports multi-processing)
    helper.make_dir_if_not_exists(png_folder)
    print(f'Doing pre-processing for {len(dicom_basenames)} files, dicom_folder: {dicom_folder}')
    if n_processes > 1:
        chunks = np.array_split(dicom_basenames, n_processes)
        chunks = [ch.tolist() for ch in chunks]  # convert to list
        # manually prepare args to be send to each function
        all_args = [(dicom_folder, the_chunk, png_folder, image_size, reduce_bits) for the_chunk in chunks]
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(_save_as_png, all_args)
    else:
        _save_as_png(dicom_folder, dicom_basenames, png_folder, image_size, reduce_bits)
