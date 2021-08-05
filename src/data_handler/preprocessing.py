import copy
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


def get_breast_percent(img_array, if_pad_small_resolution=False, if_large_resolution=False):
    # pad small resolution images if necessary
    if if_pad_small_resolution:
        if if_large_resolution:
            x_max = 3328
            y_max = 4096
        else:
            x_max = 512
            y_max = 632
        margin_x = int((x_max - img_array.shape[1]) / 2)
        margin_y = int((y_max - img_array.shape[0]) / 2)
        if margin_x != 0 or margin_y != 0:
            img_array = cv2.copyMakeBorder(src=img_array, top=margin_y, bottom=margin_y, left=margin_x, right=margin_x,
                                           borderType=cv2.BORDER_CONSTANT, value=0)

    # segment the breast
    img_breast_only, breast_mask, _ = segment_breast(img_array)

    # calculate the breast area, whole image area and breast percent area
    mask_area = np.count_nonzero(breast_mask)
    img_area = breast_mask.shape[0] * breast_mask.shape[1]
    breast_percent = mask_area / img_area
    return breast_percent


# def threshold_img(img, low_int_threshold=0):
#     # create img for thresholding and contours
#     img_8u = (img.astype('float32') / img.max() * 255).astype('uint8')
#
#     if low_int_threshold < 1:
#         low_th = int(img_8u.max() * low_int_threshold)
#     else:
#         low_th = int(low_int_threshold)
#     _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
#     return img_bin


def get_centroid(img, threshold=0):
    _, _, biggest_contour = segment_breast(img, threshold)

    M = cv2.moments(biggest_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_text_contour(array, min_contour_area=10, max_contour_area=10000, kernel_size=(10, 10), save_as=None, colorful=False):
    """
    Input should be an array with dtype np.unit8.
    """
    _, array_bin = cv2.threshold(src=array, thresh=0, maxval=255, type=cv2.THRESH_BINARY)  # binarize image
    _, contours, _ = cv2.findContours(array_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get id of largest contour (breast)
    breast_cont_idx = np.argmax([cv2.contourArea(cont) for cont in contours])

    # fill inside breast with 0's to make sure breast is not considered when extracting contours
    array_breast_removed = cv2.drawContours(array.copy(), contours, breast_cont_idx, color=(0, 0, 0), thickness=cv2.FILLED)

    breast_cont = copy.deepcopy(contours[breast_cont_idx])  # used later

    # now that breast is filled with 0, we extract contours again
    # _, contours, _ = cv2.findContours(array_breast_removed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # We know letters are close together, so apply MORPH_CLOSE with a kernel to aggregate contours that are close together so we can form a unified cluster
    array_breast_removed = cv2.morphologyEx(src=array_breast_removed.copy(), op=cv2.MORPH_DILATE, kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=kernel_size))
    _, contours, _ = cv2.findContours(array_breast_removed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ignore contours that are very small (random dust in the air) or very large (vertical bar) - they cannot be a text cluster
    contours = [cont for cont in contours if min_contour_area < cv2.contourArea(cont) < max_contour_area]

    # skip the contours below the breast (those that were part of body, but not in the same contour as the breast)
    # contour shape (N, 1, 2) where N is the number of points in cv2 contour (x coordinate -> left-to-right, y coordinate -> top-to-down)
    if len(contours) > 0:
        breast_min_x = np.amin(breast_cont[:, 0, 0])
        breast_max_y_with_min_x = -np.inf
        for point in breast_cont:  # get the point with lowest x and highest y corresponding to breast
            if point[0, 0] == breast_min_x and point[0, 1] > breast_max_y_with_min_x:
                breast_max_y_with_min_x = point[0, 1]

        # for other contours get they min y coordinate
        contours_min_y = [min([copy.deepcopy(point).flatten()[1] for point in cont]) for cont in contours]
        # if the min y for each contour is larger than max y for breast computed above, that contour should be ignored (i.e. it is below the breast)
        i_cont_to_be_skipped = [i for i in range(len(contours)) if contours_min_y[i] > breast_max_y_with_min_x]
        for i in reversed(sorted(i_cont_to_be_skipped)):
            del contours[i]

    # if there are multiple contours remaining, select the one that is far on the right compared to others
    if len(contours) > 0:
        contours_max_x = [max([copy.deepcopy(point).flatten()[0] for point in cont]) for cont in contours]
        i_selected_contour = contours_max_x.index(max(contours_max_x))  # contour whose most right-hand side point is on the right of other contours
        contours = [contours[i_selected_contour]]

    assert len(contours) <= 1, f'There are more than one contour remaning for this image: {save_as}'  # there should remain one countour at most

    # possibly save the result
    if save_as:
        if len(contours) == 1:
            if colorful:
                color = (0, 255, 0)  # greeen
                img_8u = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)  # change to RGB so it shows the color
                thickness = cv2.FILLED
            else:
                color = (0, 0, 0)  # black
                img_8u = array
                thickness = cv2.FILLED
            img_8u = cv2.drawContours(img_8u.copy(), contours, contourIdx=-1, color=color, thickness=thickness)  # fill inside selected contour (contourIdx=-1 means all contours)

        elif len(contours) == 0:
            print(f'len contours = 0, saving as original image')
            img_8u = array  # same as input array
        else:
            raise NotImplementedError('Should only have 1 contour at most')
        cv2.imwrite(save_as, img_8u)
        print(f'Saved to: {save_as}\n\n')

    return contours  # list being either empty or containing exactly one contour


def segment_breast(img, threshold=0):
    # img_bin = threshold_img(img, low_int_threshold)
    _, img_bin = cv2.threshold(src=img, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    # find the largest contour
    _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)
    biggest_contour = contours[idx]

    breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, color=255, thickness=-1)  # -1 is the same as cv2.FILLED

    # segment the breast
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)

    # breast_mask is a binary mask, and img_breast_only is with real pixel values shown in the area of breast_mask
    return img_breast_only, breast_mask, biggest_contour


# function to locate the center of mass
# def new_cropping_single_dist(img):
#     opening_it = 5
#     kernel_size = (15, 15)
#     kernel = np.ones(shape=kernel_size, dtype=np.uint8)
#
#     # create binary image from source image
#     _, img = cv2.threshold(src=img, thresh=img.min(), maxval=img.max(), type=cv2.THRESH_BINARY)  # binarize img: make all pixels greater than thresh equal to maxval, and the rest 0
#
#     # gaussian bluring to remove sharp pixels
#     img = cv2.GaussianBlur(src=img, ksize=kernel_size, sigmaX=0, sigmaY=0)   # sigmaX=0, sigmaY=0 will result in OpenCV defining the std's by itself
#
#     # apply dilation to connect all breast tissue
#     img = cv2.dilate(src=img, kernel=kernel, iterations=5)
#
#     # apply opening to remove noise
#     opening = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel=kernel, iterations=opening_it)
#
#     # add boarders to the image (width 1, value 0) to prepare for later distance transform with which we find the distance from every nonzero pixels to its boarder
#     opening = cv2.copyMakeBorder(src=opening, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=0)
#     opening = opening.astype(np.uint8)
#
#     # distance transform
#     dist = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)  # calculates the distance to the closest zero pixel for each pixel of the source image (from OpenCV doc)
#
#     # apply gussain blurring on distance transform
#     dist = cv2.GaussianBlur(src=dist, ksize=kernel_size, sigmaX=0, sigmaY=0)
#
#     # finds the position with maximum maximum element value
#     max_loc = cv2.minMaxLoc(dist)[-1]
#     return max_loc


# function to add padding or perform cropping
# to make the centroid of the breast to be the center of the image
def move_breast_centroid(img, centroid, full_height, full_width, if_movey=False):
    min_pixel_value = np.min(img)
    w = img.shape[1]
    h = img.shape[0]
    if centroid[0] >= (w - centroid[0]):  # if centroid x in the right half of total width of image
        img_new = np.full((h, 2 * centroid[0]), min_pixel_value)  # make a big pad of zero whose width is two times that of centroid x (so centroid x would be in the center)
        img_new[:, :w] = img  # fill the left part of the big pad with the actual image (no change in h) - equivalent to right padding
    else:  # for smaller breasts
        img_new = np.full((h, 2 * (w - centroid[0])), min_pixel_value)  # do the above steps in the opposite way
        img_new[:, 2 * (w - centroid[0]) - w:] = img  # left padding
    img = img_new
    
    if if_movey:  # do above steps for along y axis this time
        w = img_new.shape[1]
        h = img_new.shape[0]
        if centroid[1] >= h - centroid[1]:
            img_new = np.full((2 * centroid[1], w), min_pixel_value)
            img_new[:h, :] = img
        elif centroid[1] < h - centroid[1]:
            img_new = np.full((2 * (h - centroid[1]), w), min_pixel_value)
            img_new[2 * (h - centroid[1]) - h:, :] = img
        img = img_new
    
    w = img.shape[1]
    h = img.shape[0]
    if h > full_height:
        img = img[int((h - full_height) / 2):int((h - full_height) / 2) + full_height, :]  # crop h in the center
    if w > full_width:
        img = img[:, int((w - full_width) / 2):int((w - full_width) / 2) + full_width]  # crop w in the center

    # pad the image in case its dims are less than full_height, full_width
    # img_new = np.full((full_height, full_width), min_pixel_value)
    # img_new[int((img_new.shape[0] - img.shape[0]) / 2):int((img_new.shape[0] - img.shape[0]) / 2) + img.shape[0],
    #         int((img_new.shape[1] - img.shape[1]) / 2):int((img_new.shape[1] - img.shape[1]) / 2) + img.shape[1]] = img
    return img


def raw_to_preprocessed(raw_folder, labels_path, save_dir, specials, desired_full_height=632, desired_full_width=512, if_movey=False):
    # read the csv file as df
    df = pd.read_csv(labels_path, delimiter=';', dtype={'sourcefile': str})

    n_images = len(helper.files_with_suffix(raw_folder, suffix='.png'))
    print(f'Found {n_images} images in: {raw_folder} and {len(df)} rows in: {labels_path}')

    # preprocess images on by one
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        filename = row['Filename']
        dicom_imagelaterality = row['Dicom_image_laterality']
        dicom_windowcenter = row['Dicom_window_center']
        dicom_windowwidth = row['Dicom_window_width']
        dicom_photometricinterpretation = row['Dicom_photometric_interpretation']

        img_path = os.path.join(raw_folder, filename)
        if not os.path.isfile(img_path):  # if the file does not exist, contine
            continue

        img_array = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)  # read the 16-bit PNG as is

        # flip the image if necessary to make all breasts left-posed
        if dicom_imagelaterality == 'R':
            img_array = cv2.flip(img_array, 1)

        # intensity rescaling according to window center and window width
        # note: CSAW-M images have one value for dicom_windowcenter and dicom_windowwidth (which may not be the case in other datasets)
        img_array = exposure.rescale_intensity(img_array,
                                               in_range=(dicom_windowcenter - dicom_windowwidth / 2,
                                                         dicom_windowcenter + dicom_windowwidth / 2),
                                               out_range=(0, 255)).astype(np.uint8)  # by default output dtype is float64, we make it uint8
        # take note of the min values in the rescaled array
        min_val = int(np.min(img_array))

        # invert the color if needed
        if dicom_photometricinterpretation == 'MONOCHROME1':
            img_array = cv2.bitwise_not(img_array)

        # get the centroid of the breast
        cx, cy = get_centroid(img_array)

        # move the centroid of the breast to the center of the image along x-axis only
        new_img = move_breast_centroid(img=img_array, centroid=(cx, cy), full_height=desired_full_height, full_width=desired_full_width, if_movey=if_movey)
        # make sure the output image dimensions are as desired
        assert new_img.shape[0] == desired_full_height and new_img.shape[1] == desired_full_width, \
            'Desired full_width and full_height after moving breast do not match of the desired full_width and full_height'

        if filename not in specials['skip']:
            config = {}  # default (empty config)
            for special_key in [key for key in specials.keys() if key.startswith('special')]:
                if filename in specials[special_key]['filenames']:
                    config = specials[special_key]['config']
                    print(f'For filename: {filename}, using config: {config}')

            contours_list = get_text_contour(new_img, **config)  # get the contour aroud text (list either empty or necessarily has one element)

            if len(contours_list) > 0:
                # contour = np.squeeze(contours_list[0])  # coordinates of the points in the selected contour, array (N, 1, 2) -> squeeze to (N, 2)
                # new_img = cv2.fillPoly(img=new_img.copy(), pts=[contour], color=min_val)  # fill the contour with min value of the array (air)
                # --- for debugging when drawing contours, use e.g.: color=128, thickness=3
                new_img = cv2.drawContours(image=new_img.copy(), contours=contours_list, contourIdx=-1, color=min_val, thickness=cv2.FILLED)  # fill the contour with min value of the array (air)

        # make the path
        new_filepath = os.path.join(save_dir, filename)
        if not os.path.exists(os.path.dirname(new_filepath)):
            os.makedirs(os.path.dirname(new_filepath))
            print(f'Created folder: {os.path.dirname(new_filepath)}')

        # save the image
        cv2.imwrite(new_filepath, new_img)
        print(f'Image saved to: \n{new_filepath}')
        print(f'Done for image {i + 1}/{df.shape[0]}\n')


def _save_as_raw_png(dicom_folder, dicom_basenames, png_folder, image_size):
    """
    Notes:
        - image_size should either be a (w, h) tuple like (512, 632) or
        a dict like {(3328, 4096): (512, 632), (2560, 3328): (394, 514)}, mapping different orig resolution to donwsampling resolution.
    """
    assert type(image_size) == tuple or type(image_size) == dict, 'image_size should either be a tuple or a dict'

    for i, dicom_basename in enumerate(dicom_basenames):
        dicom_filepath = os.path.join(dicom_folder, dicom_basename)
        png_basename = dicom_basename.replace('.dcm', '.png')
        new_path = os.path.join(png_folder, png_basename)

        try:
            img_dcm = pydicom.dcmread(dicom_filepath)
            img_array = img_dcm.pixel_array  # np array uint16
        except:
            message = traceback.format_exc()  # get full traceback
            print(f'A problem occurred when reading: {dicom_filepath}: \n{message}')
            exit(1)

        # determine down-sampled image size to keep the aspect ratio of the original resolution images
        if type(image_size) == tuple:
            determined_image_size = image_size
        elif type(image_size) == dict:
            orig_shape = img_array.shape  # (h, w)
            orig_shape_wh = (img_array.shape[1], img_array.shape[0])  # (w, h)
            assert orig_shape_wh in image_size, f'Down-sampling size for orig_shape={orig_shape_wh} not specified in image_size={image_size}'
            determined_image_size = image_size[orig_shape_wh]
        else:
            raise NotImplementedError('"determined_image_size" cannot be calculated for the current image_size')

        # save images
        # if reduce_bits:  # this reduces the quality of the PNG image - it has not been used in the project
        #     img_array = img_array / (2 ** 16 - 1)
        #     img_array = helper.unnormalize_image(img_array)  # reduce to 8 bits, using np.astype(np.uint8) destroys the image
        #     Image.fromarray(img_array).convert('RGB').resize(determined_image_size).save(new_path)  # save as RGB not gray
        # else:
        # save the raw PNG as 16-bit (16 bits allocated to dicom pixel_array)
        cv2.imwrite(new_path, cv2.resize(img_array.astype(np.uint16), determined_image_size))

        print(f'Image saved to: \n{new_path}')
        print(f'Done for image {i + 1}/{len(dicom_basenames)}\n')


def dicoms_to_raw_pngs(dicom_folder, dicom_basenames, png_folder, image_size, n_processes=1):
    # this is the main functions that is used to preprocess all images to png (supports multi-processing)
    helper.make_dir_if_not_exists(png_folder)
    print(f'Doing pre-processing for {len(dicom_basenames)} files, dicom_folder: {dicom_folder}')
    if n_processes > 1:
        chunks = np.array_split(dicom_basenames, n_processes)
        chunks = [ch.tolist() for ch in chunks]  # convert to list
        # manually prepare args to be send to each function
        all_args = [(dicom_folder, the_chunk, png_folder, image_size) for the_chunk in chunks]
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(_save_as_raw_png, all_args)
    else:
        _save_as_raw_png(dicom_folder, dicom_basenames, png_folder, image_size)


# given dicom_path that points to dicom images, this function outputs a csv file with all image basenames,
# full paths, and other dicom attributes that should be considered while preprocessing
def get_dicom_attr_from_dcm(dicom_path): 
    df = pd.DataFrame()
    i = 0
    for root, fodler, files in os.walk(dicom_path):
        for name in files:
            if name.endswith('.dcm'):        
                img_path = os.path.join(root,name)
                img_dcm = pydicom.dcmread(img_path)
                img_array = img_dcm.pixel_array
                dicom_windowcenter = img_dcm.WindowCenter
                dicom_windowwidth= img_dcm.WindowWidth
                dicom_imagelaterality = img_dcm.ImageLaterality
                dicom_photometricinterpretation = img_dcm.PhotometricInterpretation
                dicom_bitsallocated = img_dcm.BitsAllocated
                dicom_bitsstored = img_dcm.BitsStored
                dicom_presentationlutshape = img_dcm.PresentationLUTShape

                try:
                    dicom_voilutfunction = img_dcm.VOILUTFunction
                except Exception as e:
                    dicom_voilutfunction = ''
                try:
                    dicom_imagerpixelspacing = str(img_dcm.ImagerPixelSpacing)
                except Exception as e:
                    dicom_imagerpixelspacing = ''
                try:
                    dicom_pixelspacing = str(img_dcm.PixelSpacing)
                except Exception as e:
                    dicom_pixelspacing = ''

                df.loc[i, 'dicom_basename'] = name
                df.loc[i, 'dicom_path'] = img_path
                df.loc[i, 'dicom_resolution'] = str(np.shape(img_array))
                df.loc[i, 'dicom_bitsallocated'] = dicom_bitsallocated
                df.loc[i, 'dicom_bitsstored'] = dicom_bitsstored
                df.loc[i, 'dicom_imagerpixelspacing'] = dicom_imagerpixelspacing
                df.loc[i, 'dicom_pixelspacing'] = dicom_pixelspacing
                df.loc[i, 'dicom_windowcenter'] = dicom_windowcenter
                df.loc[i, 'dicom_windowwidth'] = dicom_windowwidth
                df.loc[i, 'dicom_voilutfunction'] = dicom_voilutfunction
                df.loc[i, 'dicom_imagelaterality'] = dicom_imagelaterality
                df.loc[i, 'dicom_photometricinterpretation'] = dicom_photometricinterpretation
                df.loc[i, 'dicom_presentationlutshape'] = dicom_presentationlutshape
                i += 1
    return df
