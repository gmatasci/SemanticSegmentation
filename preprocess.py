from __future__ import print_function, division, absolute_import
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from scipy.misc import imsave, imread, imresize
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter
from skimage.util import view_as_windows

sys.path.insert(0, "C:/Projects/Python/utils/")
from utils import*

import os
import time
import shutil

input_dir = r"C:\Projects\Datasets\ISPRS_BENCHMARK_DATASETS\Vaihingen\ISPRS_semantic_labeling_Vaihingen"
output_dir = r"C:\Projects\Trials\SemanticSeg\Data"

nr_classes = 6
patch_sz = 128
patch_sz_out = 36

# 1: Impervious surfaces (RGB: 255, 255, 255)  white
# 2: Building (RGB: 0, 0, 255)  blue
# 3: Low vegetation (RGB: 0, 255, 255)  cyan
# 4: Tree (RGB: 0, 255, 0)  green
# 5: Car (RGB: 255, 255, 0)  yellow
# 6: Clutter/background (RGB: 255, 0, 0)  red
color_map = np.array([[255, 255, 255],
                      [0, 0, 255],
                      [0, 255, 255],
                      [0, 255, 0],
                      [255, 255, 0],
                      [255, 0, 0]
                      ])

trn_ID = np.array([1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37])
val_ID = np.array([11, 15, 28, 30, 34])
# trn_ID = np.array([1, 3])
# val_ID = np.array([11, 15])


trn_dir = os.path.join(output_dir, "trn")
val_dir = os.path.join(output_dir, "val")
img_dir = os.path.join(input_dir, "top")
dsm_dir = os.path.join(input_dir, "dsm", "nDSM")  # contains nDSM tiles produced in M. Gerke, ''Use of the stair vision library within the ISPRS 2D semantic labeling benchmark (Vaihingen)'', ITC, Univ. Twente, Enschede, The Netherlands, Tech. Rep., 2015, doi: 10.13140/2.1.5015.9683.

gt_dir = os.path.join(input_dir, "gts_for_participants")

dirs = []
dirs.append(os.path.join(trn_dir, 'X', '0'))
dirs.append(os.path.join(trn_dir, 'Y', '0'))
dirs.append(os.path.join(val_dir, 'X', '0'))
dirs.append(os.path.join(val_dir, 'Y', '0'))
dirs.append(os.path.join(val_dir, 'Y_full'))

## START ---------------------------------------------------------------------

print(python_info())

print('Preprocess.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

start_time = tic()

for dir in dirs:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

# For each image in input_images directory
nb_images = len([name for name in os.listdir(gt_dir)])

# Pre-loop to read stats for layers to then apply normalizations (either min-max mapped to 0-255 for use with PNG and ImageDataGenerator or standardization)
dsm_min = np.inf
dsm_max = 0
for file_name in os.listdir(gt_dir):
    area = file_name.split('area')[1].split('.')[0]
    dsm = imread(os.path.join(dsm_dir, 'dsm_09cm_matching_area%s_normalized.jpg' % area), mode='F')
    if np.min(dsm) < dsm_min:
        dsm_min = np.min(dsm)
    if np.max(dsm) > dsm_max:
        dsm_max = np.max(dsm)

# Loop over areas based on GT image names (dsm and top folders also contain test areas for which GT was not available to contest participants)
for file_name in os.listdir(gt_dir):

    area = file_name.split('area')[1].split('.')[0]  # string with area ID

    # Read GT image (RGB with a color coding for each class, see color_map) and convert to label map
    gt = imread(os.path.join(gt_dir, file_name), mode='RGB')
    Y = np.zeros(shape=(gt.shape[0], gt.shape[1], 1))
    for cl in range(nr_classes):
        mask = (gt[:, :, 0] == color_map[cl, 0]) and (gt[:, :, 1] == color_map[cl, 1]) and (gt[:, :, 2] == color_map[cl, 2])
        Y[mask] = cl+1

    # Read high-res orthophotos and nDSM (not original raw DSM originally provided)
    img = imread(os.path.join(img_dir, 'top_mosaic_09cm_area%s.tif' % area), mode='RGB')
    dsm = imread(os.path.join(dsm_dir, 'dsm_09cm_matching_area%s_normalized.jpg' % area), mode='F')

    # Check if there any issues with nDSM (area 11 has some problems with the raw DSM)
    try:
        dsm_8bits = np.around((dsm - dsm_min) / (dsm_max - dsm_min) * 255)
        full_data = np.concatenate((img, np.expand_dims(dsm_8bits, 2), Y), 2)
    except:
        print("area%s: failed" % area)
        continue

    # Define borders to add by mirroring before and after the image
    height, width, depth = full_data.shape
    if int(area) in val_ID:
        stride = patch_sz_out  # on the validation set/map tiles the stride is equal to the output patch size (already cropped by the valid-padding UNet)
        border_bef = int((patch_sz - patch_sz_out) / 2)    # number of affected border pixels due to network architechture: to add before the image
    elif int(area) in trn_ID:
        stride = patch_sz   # on training set the stride corresponds to the input patch size because we don't want a lot of overlap
        border_bef = 0    # on training set we start extracting patches from the true corner of the image
    border_aft_h = (stride * (np.ceil(height / stride)) + border_bef - height).astype(np.int)  # border aft computed as the difference between a multiple of the stride (covering the entire imag, plus we add border before) and the actual width
    border_aft_w = (stride * (np.ceil(width / stride)) + border_bef - width).astype(np.int)
    borders = ((border_bef, border_aft_h), (border_bef, border_aft_w), (0, 0))

    # Pad image by mirroring based on borders defined above
    full_data_padded = np.pad(full_data, pad_width=borders, mode='reflect')

    # Extract patches with a given stride (output is a 5D tensor: nr_patches_in_height x nr_patches_in_width x patch_height x patch_width x channels)
    patches = np.squeeze(view_as_windows(full_data_padded, window_shape=(patch_sz, patch_sz, depth), step=stride))

    # Save full GT image for each validation area with class labels (to be reused by evaluate_UNet())
    if int(area) in val_ID:
        dir_to_save = val_dir
        imsave(os.path.join(dir_to_save, 'Y_full', "Y_area%s.png" % area), np.squeeze(Y.astype(np.uint8)))
    elif int(area) in trn_ID:
        dir_to_save = trn_dir  # directory in which we save the patches
    else:
        continue

    # Save patches by looping over the 5D patch-tensor
    patch_ID = 0
    for h in range(patches.shape[0]):
        for w in range(patches.shape[1]):

            # Define sequential (area<ID>_p<0:nr_patches_in_area>) patch name allowing to retrieve and stitch back all patches in each area
            patch_name = "area%s_p%d.png" % (area, patch_ID)

            # Save X patches (predictors)
            try:
                # TODO to be used in case of generator based on numpy arrays:
                # np.save(os.path.join(dir_to_save, 'X', "0", "area%s_p%d-%d" % (area, i, ii)), patches[i, ii, 0, :, :, :])  #np.concatenate((img[:, :, 0:2],  np.expand_dims(dsm, 2)), 2)
                imsave(os.path.join(dir_to_save, 'X', "0", "X_%s.png" % patch_name), np.moveaxis(patches[h, w, :, :, [0, 1, 3]], 0, -1).astype(np.uint8))  # keep NIR, R and nDSM (discarding green as with ImageDataGenerator one can only use 3 channels and save a PNG or JPEG)
            except:
                print("area%s_p%d-%d: failed" % (area, h, w))
                continue

            # Save Y patches (GT, response variable)
            # TODO to be used in case of generator based on numpy arrays:
            # np.save(os.path.join(dir_to_save, 'Y', "area%s" % area), Y )
            imsave(os.path.join(dir_to_save, 'Y', "0", "Y_%s.png" % patch_name), patches[h, w, :, :, 4].astype(np.uint8))
            patch_ID += 1

    print('area %s: success' % area)

print('Total ' + toc(start_time))
