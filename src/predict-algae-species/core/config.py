"""
A simple Configuration file for training and/or predicting algae cells.
"""

import torch
import os

""" base path of the dataset """
ROOT = '/content/drive/Shared drives/algae-dataset'

""" define the path to the tiles and annotations dataset """
IMAGE_DATASET_PATH = os.path.join(ROOT, "tiles")
MASK_DATASET_PATH = os.path.join(ROOT, "annotations")

""" determine the algae species that we need """
ALGAE_SPECIES = ['Pp', 'Cr', 'Cv']

""" define the train/test split """
TEST_SPLIT = 0.2

""" determine the device to be used for training and evaluation """
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

""" determine if we will be pinning memory during data loading """
PIN_MEMORY = True if DEVICE == "cuda" else False

""" define the number of channels in the input, number of classes,
and number of levels in the U-Net model """
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

""" initialize learning rate, number of epochs to train for, and the
batch size """
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 16

""" define the input image dimensions """
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512

""" define threshold to filter weak predictions """
THRESHOLD = 0.5

""" Early stopping patience """
PATIENCE = 5

""" define the path to the base output directory """
BASE_OUTPUT = os.path.join(ROOT, "output")
if not os.path.exists(BASE_OUTPUT):
    os.mkdir(BASE_OUTPUT)

""" define the path to the output serialized model, model training
plot, and testing image paths """
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.ptm")
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
