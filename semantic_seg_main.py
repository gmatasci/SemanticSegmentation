"""
Project Name: CNN for semantic segmentation
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: SemanticSeg_UNet_main.py
Objective: Semantic segmentation (image classification) using UNet architechture on data from the ISPRS 2D Semantic Labeling Contest: http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html
"""

## TODO:
# - when running on a larger GPU, re-add layers in UNet (currently commented out) and increase patch size
# - use pre-trained network such as VGG16 or ResNet50 (after adapting input patch size)
# - sample training patches uniformley across classes (or at least augment cars)
# - create test generator to test on separate tiles (and use validation images just for Earlystopping)
# - build version with .npy files and a regular generator (with standardization) to keep also the green band
# - somehow add data augmentation
# - add NDVI as feature fed as input to the CNN
# - convert back to use flow from directory to avoid loading into memory

# DONE:
# - compute Kappa or AA as metrics on stitched image -- added Kappa and classification_report() to compute precision, recall and F-measure for each class
# - check output_layer_target if that s the one we have to modify to go from a map 2D tensor to a one-hot 3D tensor in Keras (e.g. with preprocessing_function=keras_one_hot) -- code is now modified to convert to one-hot the GT in numpy before feeding it to flow (should then match format output by UNet, i.e., 6 as last dimension)
# - if yes modify UNet code and use flow_from_directory() and use "Example of transforming images and masks together." from keras.io
# - add kappa as Keras metric used in training -- probably not differentiable, so not possible
# - check weird results (patches completely misclassified) when using clipping to 45 pix patches -- doesnt happen anymore with automatic cropping to patch_size_out
# - check re assembled image size wrt gt -- easier do to it like that with padding before and after (so we know where and how much to crop)
# - remove borders of patch and stitch together -- borders are now automatically removed as part of the network output (with padding valid)
# - add dropout -- dropout between conv layers doesnt work: removed and placed only once between the last convolution block
# - fix cmap for figs cars vs clutter -- new bounds vector with lower and upper limits 0.5 beyond class labels
# - build flow with X_stats and Y_stats -- done, training accuracy goes up, not validation (problem with generator?)
# - gridsearch and batch generator procedures as separate functions -- generators in main script still
# - add batch norm -- after each block of 2 3x3 convolutions

## IMPORT MODULES ---------------------------------------------------------------

import os
import sys
import glob
import shutil
import time
import numpy as np
import random
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib import colors
import json
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
import tensorflow as tf
from keras import backend as K
from keras import models, optimizers, callbacks

from keras.preprocessing.image import ImageDataGenerator

from image_utils import*
from UNet import*

sys.path.insert(0, "C:/Projects/Python/utils/")
from utils import*

## See devices to check whether GPU or CPU are being used
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

## PARAMETERS -------------------------------------------------------------------------------------------

PARAMS = {}

# PARAMS['exp_name'] = r'basic_UNet_model'  # Experiment name, to create a separate folder with results, figures, Tensorboard logs, etc.
PARAMS['exp_name'] = r'debugging'  # to avoid overwriting files in experiment folders

PARAMS['run_training'] = True   # if True trains model, if False loads saved model and goes to evalaution on test set
# PARAMS['run_training'] = False

# PARAMS['subsetting'] = False   # subset datasets for testing purposes
PARAMS['subsetting'] = True

PARAMS['plot_figures'] = True
# PARAMS['plot_figures'] = False

PARAMS['trn_ID'] = np.array([1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37])   # Indices of the areas for training and validation
PARAMS['val_ID'] = np.array([11, 15, 28, 30, 34])
# PARAMS['trn_ID'] = np.array([1, 3])
# PARAMS['val_ID'] = np.array([11, 15])

PARAMS['seed'] = 2018

PARAMS['nr_classes'] = 6

PARAMS['normaliz'] = True
# PARAMS['normaliz'] = False   # option to use when checking in the images produced by the generator make sense (if True, we have to rescale to plot 8-bit RGB image)

PARAMS['pct_patches_stats'] = 0.2   # fraction of training pacthes used to compute statistics to normalize data with generators

PARAMS['flips'] = True
# PARAMS['flips'] = False

PARAMS['patch_size'] = 128   # 256 goes out-of-memory
PARAMS['patch_size_out'] = 36  # 68 goes out-of-memory

PARAMS['batch_size_val'] = 8
PARAMS['nr_bands'] = 3  # for RG + DSM case

PARAMS['val_metric'] = 'val_loss'  # alternatively monitor='val_acc'
PARAMS['epochs'] = 2000
PARAMS['patience'] = 20  # number of epochs we tolerate the validation accuracy to be stagnant (not larger than current best accuracy)

# PARAMS['batch_size_trn'] = [4, 8, 16]
PARAMS['batch_size_trn'] = [2]

# PARAMS['dropout'] = [0, 0.2, 0.4]   # dropout 0 means we keep all the units
PARAMS['dropout'] = [0.2]

# PARAMS['learn_rate'] = [1e-4, 1e-5]     # 0.0001 usually gives the best results with Adam
PARAMS['learn_rate'] = [1e-4]

PARAMS['conf_mat_norm'] = True   # whether to normalize confusion by the true totals

## Definition of the directories
PARAMS['dirs'] = {}
PARAMS['dirs']['base'] = r'C:\Projects\Trials\SemanticSeg'
PARAMS['dirs']['log'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], "Tensorboard_logs")
PARAMS['dirs']['model'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], "Models")
PARAMS['dirs']['fig'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], "Figures")
PARAMS['dirs']['data'] = os.path.join(PARAMS['dirs']['base'], 'Data')
PARAMS['dirs']['res'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], "Results")
PARAMS['dirs']['gt_full_dir'] = os.path.join(PARAMS['dirs']['data'], 'val', 'Y_full')

PARAMS['nr_samples_trn'] = len(os.listdir(os.path.join(PARAMS['dirs']['data'], "trn", "X", '0')))
PARAMS['nr_samples_val'] = len(os.listdir(os.path.join(PARAMS['dirs']['data'], "val", "X", '0')))


## COLOR MAPS ---------------------------------------------------------------------

labels = ['impervious_surfaces',
          'building',
          'low_vegetation',
          'tree',
          'car',
          'clutter_background']

color_map = np.array([[255, 255, 255],
                      [0, 0, 255],
                      [0, 255, 255],
                      [0, 255, 0],
                      [255, 255, 0],
                      [255, 0, 0]
                      ])

# Make a color map of fixed colors for categorical labels
cmap = colors.ListedColormap(color_map / 255)
bounds = np.arange(0.5, color_map.shape[0] + 1.5, 1)  # vector with upper and lower limits for each class (class labels shifted by 0.5)
norm = colors.BoundaryNorm(bounds, cmap.N)

## DEFINE FUNCTIONS ------------------------------------------------------------------

def UNet_grid_search(train_gen, val_gen, val_metric, hparams):
    """
    Gridsearch over the hyper parameters
    :param train_gen: training generator
    :param val_gen: validation genrator
    :param hparams: dictionary with vectors for batch_size, learning rate and dropout
    :return best_val_metric: best validation metric value through the epochs for current set of hparams
    :return best_epoch: corresponding best epoch
    :return best_model_path: path to best model (saved by the Checkpoint callback)
    """

    train_gen.batch_size = hparams['bs']   # make batch size a varying parameter

    if val_metric == 'val_loss':
        val_mode = 'min'
    elif val_metric == 'val_acc':
        val_mode = 'max'

    # Create UNet model (model.summary() shows nr of trainable parameters)
    model = UNet().create_model(img_shape=(PARAMS['patch_size'], PARAMS['patch_size'], PARAMS['nr_bands']), num_class=PARAMS['nr_classes'], dropout=hparams['do'])

    # Define optimizer and compile
    adam = optimizers.Adam(lr=hparams['lr'])
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # Build string to give best models and folders different names for each hparams combination
    hparams_str = ''
    for k, v in hparams.items():
        hparams_str += '%s_%g_' % (k, v)
    hparams_str = hparams_str[0:-1]

    # Delete and recreate folder for Tensorboard logs
    log_dir_hparams = os.path.join(PARAMS['dirs']['log'], hparams_str)
    if os.path.exists(log_dir_hparams):
        shutil.rmtree(log_dir_hparams)
    os.makedirs(log_dir_hparams)

    # Earlystopping callback with a given patience
    earlystop_callback = callbacks.EarlyStopping(monitor=val_metric, mode=val_mode, patience=PARAMS['patience'])  # prefix 'val_' added automatically by Keras (based on name of Loss function)

    # Tensorboard callback to visualize network/evolution of metrics
    tb_callback = callbacks.TensorBoard(log_dir=log_dir_hparams, write_graph=True)

    # Checkpoint callback to save model each time the validation score (loss, acc, etc.) improves
    best_model_path = os.path.join(PARAMS['dirs']['model'], 'best_model_%s.hdf5' % hparams_str)
    checkpoint_callback = callbacks.ModelCheckpoint(best_model_path, monitor=val_metric, mode=val_mode, verbose=1, save_best_only=True)

    # Learning rate callback to reduce learning rate if val_loss does not improve after patience epochs (divide by 10 each time till a minimum of 0.000001)
    reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor=val_metric, mode=val_mode, factor=0.1, patience=PARAMS['patience']/2, min_lr=1e-6)

    # TODO Only to be used with flow_from_directory()
    # if PARAMS['nr_bands'] == 1:
    #     color_mode = 'grayscale'
    # else:
	#     color_mode = 'rgb'

    # Train model
    history = model.fit_generator(train_gen,
              epochs=PARAMS['epochs'],
              steps_per_epoch=np.ceil(PARAMS['nr_samples_trn'] / hparams['bs']),
              validation_data=val_gen,
              callbacks=[earlystop_callback, reduce_lr_callback, checkpoint_callback, tb_callback],
              validation_steps=np.ceil(PARAMS['nr_samples_val'] / PARAMS['batch_size_val']),
              verbose=2)

    ## Get best accuracy and corresponding epoch
    if val_metric == 'val_loss':
        best_val_metric = np.min(history.history[val_metric])
        best_epoch = np.argmin(history.history[val_metric])+1
    elif val_metric == 'val_acc':
        best_val_metric = np.max(history.history[val_metric])
        best_epoch = np.argmax(history.history[val_metric])+1

    return best_val_metric, best_epoch, best_model_path


def evaluate_UNet(model, test_gen, test_patches_names):
    """
    Evaluates the model on the Validation images
    :param model: Keras model object to apply
    :param test_gen: generator to load batches of test images
    :param test_patches_names: list of test patch names used to separate the patches of each area via the area ID
    :return RES: dictionary with the results on the test set: conf_mat (true labels as rows, predicted labels as columns), OA, Kappa, class_measures
    """

    test_gen.shuffle = False   # if shuffle was set to True reset it to False to make sure test patches are in the proper order to rebuild the image

    nb_patches = test_gen.n

    print("Testing on %d patches" % (nb_patches))

    # Compute number of patches in each area to split the 4D tensor resulting from the prediction
    nr_patches_per_image = np.zeros(len(PARAMS['val_ID'])).astype(np.int)
    for i, ID in enumerate(PARAMS['val_ID']):
        nr_patches_per_image[i] = sum(['_area%d' % ID in patch_name for patch_name in test_patches_names])

    # Predict on test patches and convert to labels
    Y_tst_pred_4D = model.predict_generator(test_gen, verbose=1)   # 4D: nr_patches x height_out x width_out x nr_classes
    Y_tst_pred_map_3D = np.argmax(Y_tst_pred_4D, axis=3)+1   # 3D: nr_patches x height_out x width_out

    # Initialize lists to contain labels across all areas
    Y_tst_class = []
    Y_tst_pred_class = []

    # Loop over the area IDs
    for i, ID in enumerate(PARAMS['val_ID']):

        # Read GT from which to determine height and width as well as to get GT labels list Y_tst_class
        gt = imread(os.path.join(PARAMS['dirs']['gt_full_dir'], "Y_area%s.png" % ID))
        Y_tst_class.extend(gt.flatten())  # grow full GT labels list
        height, width = gt.shape

        # Select portion of 4D tensor with maps of current area
        start = sum(nr_patches_per_image[0:i])
        stop = sum(nr_patches_per_image[0:i+1])
        Y_tst_pred_map_3D_area = Y_tst_pred_map_3D[start:stop, :, :]

        # Define padded height and width to reconstruct the map with stitched patches that will almost certainly go over original GT border
        stride = PARAMS['patch_size_out']  # stride is equal to the patch size after the valid-padding UNet automatically cropped them
        height_padded = (stride * np.ceil(height / stride)).astype(np.int)   # dimensions from which patches have been created (_padded)
        width_padded = (stride * np.ceil(width / stride)).astype(np.int)

        # Loop to fill the complete map with the patches
        Y_tst_pred_map_full = np.empty([height_padded, width_padded])
        p = 0
        for i_h, h in enumerate(range(0, height_padded, stride)):
            for i_w, w in enumerate(range(0, width_padded, stride)):
                Y_tst_pred_map_full[h:h + stride, w:w + stride] = Y_tst_pred_map_3D_area[p, :, :]
                p += 1

        # Crop to GT extent to ensure we can compare labels
        Y_tst_pred_map_full_cropped = Y_tst_pred_map_full[:height, :width]

        # Grow full predicted labels list
        Y_tst_pred_class.extend(Y_tst_pred_map_full_cropped.flatten())

        if PARAMS['plot_figures']:

            map_dir = os.path.join(PARAMS['dirs']['fig'], 'maps')
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)

            # Predicted map
            plt.figure()
            plt.imshow(Y_tst_pred_map_full_cropped, cmap=cmap, norm=norm)
            plt.axis('image')
            plt.savefig(os.path.join(map_dir, "Predicted_map_area%s.png" % ID), dpi=600, bbox_inches='tight')
            plt.savefig(os.path.join(map_dir, "Predicted_map_area%s.pdf" % ID), dpi=600, bbox_inches='tight')
            plt.close()

            # GT map
            plt.figure()
            plt.imshow(gt, cmap=cmap, norm=norm)
            plt.axis('image')
            plt.savefig(os.path.join(map_dir, "GT_map_area%s.png" % ID), dpi=600, bbox_inches='tight')
            plt.savefig(os.path.join(map_dir, "GT_map_area%s.pdf" % ID), dpi=600, bbox_inches='tight')
            plt.close()

    # Convert labels lists to array with labels starting at 1
    Y_tst_class = np.array(Y_tst_class, dtype=np.uint8)
    Y_tst_pred_class = np.array(Y_tst_pred_class, dtype=np.uint8)

    # Assess test predictions and save results
    RES = {}
    conf_mat = confusion_matrix(Y_tst_class, Y_tst_pred_class)
    if PARAMS['conf_mat_norm']:
        RES['conf_mat'] = np.round((conf_mat.astype(np.float) / conf_mat.sum(axis=1)[:, np.newaxis])*100, 1)  # normalized by true labels totals (true labels as rows, predicted labels as columns)
    else:
        RES['conf_mat'] = conf_mat
    RES['OA'] = np.round(accuracy_score(Y_tst_class, Y_tst_pred_class)*100, 2)
    RES['Kappa'] = cohen_kappa_score(Y_tst_class, Y_tst_pred_class)
    RES['class_measures'] = classification_report(Y_tst_class, Y_tst_pred_class, target_names=labels)
    # TODO add
    # RES['F1'] = f1_score(Y_tst_class, Y_tst_pred_class)

    print('Classification results:\n\n '
          'Confusion matrix:\n %s \n\n '
          'OA=%.2f, Kappa=%.3f \n\n '
          'Class-specific measures:\n %s'
          % (RES['conf_mat'], RES['OA'], RES['Kappa'], RES['class_measures']))

    return RES



## START ---------------------------------------------------------------------

print(python_info())

print('SemanticSeg_UNet_main.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

start_time = tic()

K.clear_session()  # release the memory on the GPU

# Create directories
for name, dir in PARAMS['dirs'].items():
    if not os.path.exists(dir):
        os.makedirs(dir)


## DEFINE GENERATORS --------------------------------------------------------------------------------------------------

# TRN generators

# X-specific arguments  TODO to add rotation_range=90. , shears and noise
data_gen_args_X_trn = dict(horizontal_flip=PARAMS['flips'],
                           vertical_flip=PARAMS['flips'],
                           featurewise_center=PARAMS['normaliz'],
                           featurewise_std_normalization=PARAMS['normaliz'])

# TODO TODEL Y-specific arguments, not needed as X_datagen_trn will consider only X as the images to augment
# Y_datagen_trn = ImageDataGenerator(**data_gen_args_Y_trn)
# data_gen_args_Y_trn = dict(horizontal_flip=PARAMS['flips'],
#                            vertical_flip=PARAMS['flips'])

X_datagen_trn = ImageDataGenerator(**data_gen_args_X_trn)

# List all training image patch names sorted by area then by patch number
patches_names_trn = os.listdir(os.path.join(PARAMS['dirs']['data'], "trn", "X", '0'))
patches_names_trn_sorted = sort_patch_names(patches_names_trn, PARAMS['trn_ID'])
nr_patches_trn = len(patches_names_trn_sorted)

# Load training image patches in a 4D matrix
X_trn = np.empty([nr_patches_trn, PARAMS['patch_size'], PARAMS['patch_size'], PARAMS['nr_bands']]).astype(np.uint8)
for i, patch_name in enumerate(patches_names_trn_sorted):
    X_trn[i, :, :, :] = imread(os.path.join(PARAMS['dirs']['data'], "trn", "X", '0', patch_name)).astype(np.uint8)

# Randomly sample a subset of the training patches to compute statistics to normalize data
# nr_patches_stats_trn = np.round(PARAMS['pct_patches_stats']*nr_patches_trn).astype(np.int)
# patches_rand_names_trn = [patches_names_trn_sorted[i] for i in sorted(random.sample(range(nr_patches_trn), nr_patches_stats_trn))]
X_stats_trn = X_trn  # TODO change back to a smaller X_stats_trn

# Fit training generator on training set subsample
X_datagen_trn.fit(X_stats_trn, seed=PARAMS['seed'])

# Load training GT in a 4D matrix and convert to one-hot format
Y_trn = np.empty([nr_patches_trn, PARAMS['patch_size'], PARAMS['patch_size'], PARAMS['nr_classes']])
for i, patch_name in enumerate(patches_names_trn_sorted):
    gt = imread(os.path.join(PARAMS['dirs']['data'], "trn", "Y", '0', patch_name.replace('X_', 'Y_')))
    Y_trn[i, :, :, :] = map_2_one_hot(gt, PARAMS['nr_classes'])

# Crop to central region only (border_bef is found based on analysis of the tensor shapes in Keras model definition)
size_diff = PARAMS['patch_size'] - PARAMS['patch_size_out']
border_bef = np.floor(size_diff / 2).astype(np.int)
Y_trn_cropped = Y_trn[:, border_bef:border_bef+PARAMS['patch_size_out'], border_bef:border_bef+PARAMS['patch_size_out'], :]

# Subset data for testing pruposes
if PARAMS['subsetting']:
    nr_patches_trn = 4
    X_trn = X_trn[0:nr_patches_trn, :, :, :]
    Y_trn_cropped = Y_trn_cropped[0:nr_patches_trn, :, :, :]

train_generator = X_datagen_trn.flow(x=X_trn, y=Y_trn_cropped, seed=PARAMS['seed'])  # add shuffle=False if we want the iterator to go through the samples in a sequantial fashion

# VAL generators

# X-specific arguments (no Y-specific arguments as we do not augment the validation set)
data_gen_args_X_val = dict(featurewise_center=PARAMS['normaliz'],
                           featurewise_std_normalization=PARAMS['normaliz'])

X_datagen_val = ImageDataGenerator(**data_gen_args_X_val)

# Fit validation generator on same training set subsample
X_datagen_val.fit(X_stats_trn, seed=PARAMS['seed'])

# List all validation image patch names sorted by area then by patch number (to allow repositioning them in the right spot when retiling)
patches_names_val = os.listdir(os.path.join(PARAMS['dirs']['data'], "val", "X", '0'))
patches_names_val_sorted = sort_patch_names(patches_names_val, PARAMS['val_ID'])
nr_patches_val = len(patches_names_val_sorted)

# Load validation image patches in a 4D matrix
X_val = np.empty([nr_patches_val, PARAMS['patch_size'], PARAMS['patch_size'], PARAMS['nr_bands']])
for i, patch_name in enumerate(patches_names_val_sorted):
    X_val[i, :, :, :] = imread(os.path.join(PARAMS['dirs']['data'], "val", "X", '0', patch_name))

# Load validation GT in a 4D matrix and convert to one-hot format
Y_val = np.empty([nr_patches_val, PARAMS['patch_size'], PARAMS['patch_size'], PARAMS['nr_classes']])
for i, patch_name in enumerate(patches_names_val_sorted):
    gt = imread(os.path.join(PARAMS['dirs']['data'], "val", "Y", '0', patch_name.replace('X_', 'Y_')))
    Y_val[i, :, :, :] = map_2_one_hot(gt, PARAMS['nr_classes'])

# Crop to central region only (border_bef is found based on analysis of the tensor shapes in Keras model definition)
Y_val_cropped = Y_val[:, border_bef:border_bef+PARAMS['patch_size_out'], border_bef:border_bef+PARAMS['patch_size_out'], :]

val_generator = X_datagen_val.flow(x=X_val, y=Y_val_cropped,
                                   batch_size=PARAMS['batch_size_val'],
                                   seed=PARAMS['seed'])   # add shuffle=False if we want the iterator to go through the samples in a sequantial fashion

## TRAINING --------------------------------------------------------------------------------------------------

if PARAMS['run_training']:

    # Gridsearch over the hyperparameters
    grid_search_list = []   # list to be converted to pd dataframe
    for bs in PARAMS['batch_size_trn']:
        for lr in PARAMS['learn_rate']:
            for do in PARAMS['dropout']:
                print('Batch size = %g, Learning rate = %g, dropout = %g' % (bs, lr, do))
                hp = {}
                hp['bs'] = bs        # at each iteration reset to specific value
                hp['lr'] = lr
                hp['do'] = do
                val_score, epoch, model_path = UNet_grid_search(train_gen=train_generator, val_gen=val_generator, val_metric='val_acc', hparams=hp)
                grid_search_list.append({'batch_size': bs, 'learn_rate': lr, 'dropout': do,
                                         'epoch': epoch, 'val_score': val_score, 'model_path': model_path})  # fill row entries with dictionary

    # Get best values
    grid_search_df = pd.DataFrame(grid_search_list)   # convert to pd dataframe
    if PARAMS['val_metric'] == 'val_loss':
        grid_search_df.sort_values(by='val_score', ascending=True, inplace=True)
    elif PARAMS['val_metric'] == 'val_acc':
        grid_search_df.sort_values(by='val_score', ascending=False, inplace=True)

    best_bs = grid_search_df['batch_size'].iloc[0]
    best_lr = grid_search_df['learn_rate'].iloc[0]
    best_do = grid_search_df['dropout'].iloc[0]
    best_epoch = grid_search_df['epoch'].iloc[0]
    best_model_path = grid_search_df['model_path'].iloc[0]

else:

    best_model_path = r'C:\Projects\Trials\SemanticSeg\wkg\debugging\Models\best_model_bs_2_lr_0.0001_do_0.2.hdf5'
    
## TESTING --------------------------------------------------------------------------------------------------

# Load best model from saved file, as model object after .fit() is a snapshot at the "best epoch + patience" point
best_model = models.load_model(best_model_path)  # custom_objects={"PSNRLoss": PSNRLoss}

# Apply model on test set (TODO here we use the val generator and val patch names, to change to test)
RES = evaluate_UNet(best_model, test_gen=val_generator, test_patches_names=patches_names_val_sorted)

# Convert to list any possible np array for json.dump() to work
for key, val in PARAMS.items():
    if isinstance(val, np.ndarray):
        PARAMS[key] = val.tolist()

# Save parameters of this run in results folder
params_filename = 'PARAMS_SemanticSegmentation_%s.json' % PARAMS['exp_name']
with open(os.path.join(PARAMS['dirs']['res'], params_filename), 'w') as fp:
    json.dump(PARAMS, fp)

# Convert to list any possible np array for json.dump() to work
for key, val in RES.items():
    if isinstance(val, np.ndarray):
        RES[key] = val.tolist()

# Save results of this run in results folder
res_filename = 'RES_SemanticSegmentation_%s.json' % (PARAMS['exp_name'])
with open(os.path.join(PARAMS['dirs']['res'], res_filename), 'w') as fp:
    json.dump(RES, fp)

print('Total ' + toc(start_time))



## TODO TODEL -----------------------

# Non-overlapping patches:
# height_padded = PARAMS['patch_size']*(np.ceil(height / PARAMS['patch_size'])).astype(np.int)
# width_padded = PARAMS['patch_size']*(np.ceil(width / PARAMS['patch_size'])).astype(np.int)
# Y_tst_pred_map_full = np.empty([height_padded, width_padded])  # to accomodate patches that would go over the border of the image
# p = 0
# for h in range(0, height_padded, PARAMS['patch_size']):
#     for w in range(0, width_padded, PARAMS['patch_size']):
#         Y_tst_pred_map_full[h:h+PARAMS['patch_size'], w:w+PARAMS['patch_size']] = Y_tst_pred_map_3D_area[p, :, :]
#         p += 1
# Y_tst_pred_map_full = Y_tst_pred_map_full[:height, :width]  # to reclip it back to its original size

# Overlapping patches:
# stride = PARAMS['patch_size_out']
# height, width = gt.shape
# height_padded = (stride * (np.ceil((height - PARAMS['patch_size']) / stride)) + PARAMS['patch_size_out']).astype(np.int)
# width_padded = (stride * (np.ceil((width - PARAMS['patch_size']) / stride)) + PARAMS['patch_size']).astype(np.int)

# height_padded = (stride * (np.ceil((height-PARAMS['patch_size'])/stride)) + stride).astype(np.int)
# width_padded = (stride * (np.ceil((width-PARAMS['patch_size'])/stride)) + stride).astype(np.int)

# ---------------------------

# padding_ht = height_padded - height
# padding_wt = width_padded - width
# if padding_ht < overlap or padding_wt < overlap:
#     print('Overlap = %g: padding_ht %g, padding_wt = %g', overlap, padding_ht, padding_wt)
#     break
# offset_ht = np.floor(padding_ht / 2)
# offset_wt = np.floor(padding_wt / 2)
#
# gt_padded = np.zeros((height_padded, width_padded))  # use ones to have a valid class for padding (Impervious surfaces)
# gt_padded[offset_ht:offset_ht+gt.shape[0], offset_wt:offset_wt+gt.shape[1]] = gt

# ---------------------------

# if PARAMS['nr_conv_3x3'] > 0:
#     height_clipped = height_padded - overlap
#     width_clipped = width_padded - overlap
#     Y_tst_pred_map_3D_area_clipped = Y_tst_pred_map_3D_area[:, PARAMS['nr_conv_3x3']:-PARAMS['nr_conv_3x3'], PARAMS['nr_conv_3x3']:-PARAMS['nr_conv_3x3']]
#     gt_clipped = gt_padded[PARAMS['nr_conv_3x3']:-PARAMS['nr_conv_3x3'], PARAMS['nr_conv_3x3']:-PARAMS['nr_conv_3x3']]
#
# elif PARAMS['nr_conv_3x3'] == 0:
#     height_clipped = height_padded
#     width_clipped = width_padded
#     Y_tst_pred_map_3D_area_clipped = Y_tst_pred_map_3D_area
#     gt_clipped = gt_padded


# ---------------------------------------------------------

# cannot use flow_from_directory() because then GT is a 2D tensor (as grey scale PNG image,
# only format accepted as we're using the ImageDataGenerator class) and we would need to have the one-hot transform in Keras
#  train_generator_X = X_datagen_trn.flow_from_directory(os.path.join(PARAMS['dirs']['data'], "trn", "X"),
#     target_size=(PARAMS['patch_size'], PARAMS['patch_size']),
#     color_mode=color_mode,
#     batch_size=hparams['bs'],
#     class_mode=None, seed=PARAMS['seed'])   # add shuffle=False if we want the iterator to go through the samples in a sequantial fashion
# train_generator_Y = Y_datagen_trn.flow_from_directory(os.path.join(PARAMS['dirs']['data'], "trn", "Y"),
#     target_size=(PARAMS['patch_size'], PARAMS['patch_size']),
#     color_mode='grayscale',
#     batch_size=hparams['bs'],
#     class_mode=None, seed=PARAMS['seed'])   # class_mode to be set to None as we want the image to be yield
# train_generator = zip(train_generator_X, train_generator_Y)
#
# val_generator_X = X_datagen_trn.flow_from_directory(os.path.join(PARAMS['dirs']['data'], "val", "X"),
#                                                   target_size=(PARAMS['patch_size'], PARAMS['patch_size']),
#                                                   color_mode=color_mode,
#                                                   batch_size=PARAMS['batch_size_val'],
#                                                   class_mode=None, seed=PARAMS['seed'])
# val_generator_Y = Y_datagen_trn.flow_from_directory(os.path.join(PARAMS['dirs']['data'], "val", "Y"),
#                                                   target_size=(PARAMS['patch_size'], PARAMS['patch_size']),
#                                                   color_mode='grayscale',
#                                                   batch_size=PARAMS['batch_size_val'],
#                                                   class_mode=None, seed=PARAMS['seed'])
# val_generator = zip(val_generator_X, val_generator_Y)

# # Check generator to see if images in the batches make sense
# nr_steps = 8
# for ds in ['trn', 'val']:
#     if ds == 'trn':
#         generator = train_generator
#     else:
#         generator = val_generator
#     for b in range(nr_steps):
#         X_batch, Y_batch = next(generator)
#         Y_batch_GT = np.argmax(Y_batch, axis=3) + 1
#         # if (b == 0) or ((b+1) % 1991 == 0):
#         nr_imgs = X_batch.shape[0]
#         for i in range(nr_imgs):
#             f, a = plt.subplots(1, 2)
#             a[0].imshow(X_batch[i])
#             a[0].set_title('X')
#             a[1].imshow(np.squeeze(Y_batch_GT[i]), cmap=cmap, norm=norm)
#             a[1].set_title('Y')
#             plt.savefig(os.path.join(PARAMS['dirs']['fig'], 'checks', 'generators', '%s_batch%d_img%d.png' % (ds, b, i)))
#             plt.close()