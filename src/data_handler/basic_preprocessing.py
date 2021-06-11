import pandas as pd
import os
import numpy as np
import pydicom
import cv2
from skimage import exposure
import helper
from PIL import Image
import multiprocessing

# original resolution of images
FULL_HEIGHT = 4096
FULL_WIDTH = 3328


# function to locate the center of mass
def new_cropping_single_dist(img):
    org_img = img
    opening_it = 5
    kernel_size = (25, 25)
    kernel = np.ones(kernel_size, np.uint8)
    _, img = cv2.threshold(img, img.min(), img.max(), cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, kernel_size, 0)
    img = cv2.dilate(img, kernel, 5)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opening_it)
    opening = cv2.copyMakeBorder(opening, top=1, bottom=1, left=1, right=1, borderType= cv2.BORDER_CONSTANT, value=0 )
    opening = opening.astype(np.uint8)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, kernel_size, 0)
    maxLoc = cv2.minMaxLoc(dist)[-1]
    return maxLoc


# function to add padding or perform cropping
# to make the center of mass be the center of the image
def pad_or_crop_single_maxloc(img, maxloc):
    w = img.shape[1]
    h = img.shape[0]
    if maxloc[0] >= w-maxloc[0]:
        img_new = np.full((h,2*maxloc[0]), 0.0)
        img_new[:, :w] = img
    elif maxloc[0] < w-maxloc[0]:
        img_new = np.full((h,2*(w-maxloc[0])), 0.0)
        img_new[:, 2*(w-maxloc[0])-w:] = img
    img = img_new
    w = img_new.shape[1]
    h = img_new.shape[0]
    if maxloc[1] >= h-maxloc[1]:
        img_new = np.full((2*maxloc[1], w), 0.0)
        img_new[:h, :] = img
    elif maxloc[1] < h-maxloc[1]:
        img_new = np.full((2*(h-maxloc[1]), w), 0.0)
        img_new[2*(h-maxloc[1])-h:, :] = img
    img = img_new
    w = img.shape[1]
    h = img.shape[0]
    if h > FULL_HEIGHT:
        img = img[int((h - FULL_HEIGHT) / 2):int((h - FULL_HEIGHT) / 2) + FULL_HEIGHT, :]
    if w > FULL_WIDTH:
        img = img[:, int((w - FULL_WIDTH) / 2):int((w - FULL_WIDTH) / 2) + FULL_WIDTH]
    img_new = np.full((FULL_HEIGHT, FULL_WIDTH), 0.0)
    img_new[int((img_new.shape[0]-img.shape[0])/2):int((img_new.shape[0]-img.shape[0])/2)+img.shape[0], int((img_new.shape[1]-img.shape[1])/2):int((img_new.shape[1]-img.shape[1])/2)+img.shape[1]] = img
    return img_new


def save_as_png(dicom_folder, dicom_basenames, new_path_folder, data_csv_path, image_size=(FULL_WIDTH, FULL_HEIGHT), reduce_bits=False, do_preprocess=True):
    """
    In this function, images are all flipped (if needed) so the breast is positioned on the left-hand side of the image.
    """
    # print(f'Doing basic train_preprocessing for {len(dicom_basenames)} dicom files in: {dicom_folder}')
    helper.make_dir_if_not_exists(new_path_folder)

    # read the csv into dataframe and select the subset with the given dicom basenames
    new_df = pd.read_csv(data_csv_path, delimiter=';', dtype={'sourcefile': str})
    new_df = new_df[new_df['basename'].isin(dicom_basenames)]  # only consider available files in the dicom folder

    # loop over the rows of the dataframe and do pre-processing based on the attributes
    start_index = 0
    for i in range(start_index, new_df.shape[0]):
        row = new_df.iloc[i, :]
        dicom_basename = row['basename']
        dicom_filepath = os.path.join(dicom_folder, dicom_basename)
        png_basename = dicom_basename.replace('.dcm', '.png')
        new_path = os.path.join(new_path_folder, png_basename)

        dicom_windowcenter = str(row['dicom_windowcenter']).strip('][').replace('\'', '').split(', ')
        dicom_windowwidth = str(row['dicom_windowwidth']).strip('][').replace('\'', '').split(', ')
        dicom_windowcenter = [int(float(value)) for value in dicom_windowcenter]
        dicom_windowwidth = [int(float(value)) for value in dicom_windowwidth]

        # read dicom and keep a record of corrupted images
        try:
            img_dcm = pydicom.dcmread(dicom_filepath)
            img_array = img_dcm.pixel_array
        except:
            print('dicom_problem')
            new_df.loc[i, 'dicom_corrupted'] = 1
            continue

        if do_preprocess:
            # flip the image if necessary to make all breasts left-posed
            if row['dicom_imagelaterality'] == 'R':
                img_array = cv2.flip(img_array, 1)

            # intensity rescaling according to window center and window width
            img_array = exposure.rescale_intensity(img_array, in_range=(dicom_windowcenter[0] - dicom_windowwidth[0] / 2,
                                                                        dicom_windowcenter[0] + dicom_windowwidth[0] / 2))
            # invert the color if needed
            if row['dicom_photometricinterpretation'] == 'MONOCHROME1':
                img_array = cv2.bitwise_not(img_array)

            # crop with distance transform and pad
            max_loc = new_cropping_single_dist(img_array)
            new_img = pad_or_crop_single_maxloc(img_array, max_loc)
        else:
            new_img = img_array

        # save preprocessed images
        if reduce_bits:  # this reduces the quality of the PNG image
            new_img = new_img / (2 ** 16 - 1)
            new_img = helper.unnormalize_image(new_img)  # reduce to 8 bits, using np.astype(np.uint8) destroys the image
            Image.fromarray(new_img).convert('RGB').resize(image_size).save(new_path)  # save as RGB not gray
        else:
            cv2.imwrite(new_path, cv2.resize(new_img.astype(np.uint16), image_size))

        print(f'Image saved to: \n{new_path}')
        print(f'Done for image {i + 1}/{len(dicom_basenames)}\n')


def convert_dicoms_to_pngs(dicom_folder, png_folder, dicom_basenames, data_csv_path, image_size=None, reduce_bits=False, n_processes=1, do_preprocess=True):
    # this is the main functions that is used to preprocess all images to png (supports multi-processing)
    print(f'Doing pre-processing for {len(dicom_basenames)} files, dicom_folder: {dicom_folder}')
    if n_processes > 1:
        chunks = np.array_split(dicom_basenames, n_processes)
        chunks = [ch.tolist() for ch in chunks]  # convert to list
        # manually prepare args to be send to each function
        all_args = [(dicom_folder, the_chunk, png_folder, data_csv_path, image_size, reduce_bits, do_preprocess) for the_chunk in chunks]
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(save_as_png, all_args)
    else:
        save_as_png(dicom_folder, dicom_basenames, png_folder, data_csv_path, image_size, reduce_bits, do_preprocess=do_preprocess)
