import pickle
import os
from PIL import Image, ImageDraw
import time 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import scipy
from scipy import signal
from scipy.optimize import curve_fit
import tensorflow as tf
import sys

ROOT_DIR = "C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"

# Directory to save logs and trained model
MODEL_DIR = ROOT_DIR + "\\droplet_logs"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

DATASET_INPUT_DIR = ROOT_DIR + "\\datasets\\input"
DATASET_OUTPUT_DIR = ROOT_DIR + "\\datasets\\output"
list_DATASET_INPUT_DIR = os.listdir(DATASET_INPUT_DIR) 
list_DATASET_OUTPUT_DIR = os.listdir(DATASET_OUTPUT_DIR)

VERWENDETE_DATEN_LISTE = [item for item in list_DATASET_INPUT_DIR if not item in list_DATASET_OUTPUT_DIR]

for VERWENDETE_DATEN in VERWENDETE_DATEN_LISTE:

    # is the evaluation done on cluster? 1 = yes, 0 = no
    cluster = 0
    masks = 0
    # max. image size
    # The multiple of 64 is needed to ensure smooth scaling of feature
    # maps up and down the 6 levels of the FPN pyramid (2**6=64),
    # e.g. 64, 128, 256, 512, 1024, 2048, ...
    # Select the closest value corresponding to the largest side of the image.
    image_max = 1024
    # is the evaluation done on CPU or GPU? 1=GPU, 0=CPU
    device = 1
    dataset_path = VERWENDETE_DATEN   
    weights_path = r"Netze_Tropfen"
    weights_name = r"mask_rcnn0_droplet_1305"

    tf.to_float = lambda x: tf.cast(x, tf.float32)
    # Root directory of the project
    if cluster == 0:
        WEIGHTS_DIR = ROOT_DIR+"\\droplet_logs\\Netze_Tropfen\\mask_rcnn0_droplet_1305.h5"
        IMAGE_MAX = image_max
        MASKS = masks
    else:
        import argparse
        # Parse command line arguments
        parser = argparse.ArgumentParser(
                description='evaluation on cluster')
        parser.add_argument('--dataset_path', required=True,
                                help='Dataset path to find in Mask_R_CNN\datasets\input')
        parser.add_argument('--save_path', required=True,
                                help='Save path to find in Mask_R_CNN\datasets\output')
        parser.add_argument('--name_result_file', required=True,
                                help='Name of the excel result file to find in Mask_R_CNN\datasets\output')
        parser.add_argument('--weights_path', required=True,
                                help='Weights path to find in Mask_R_CNN\droplet_logs')
        parser.add_argument('--weights_name', required=True,
                                help='Choose Neuronal Network / Epoch to find in Mask_R_CNN\droplet_logs')
        parser.add_argument('--masks', required=False,
                            default=0,
                                help='Generate detection masks? 1 = yes, 0 = no')
        parser.add_argument('--device', required=False,
                            default=0,
                            help='is the evaluation done on CPU or GPU? 1=GPU, 0=CPU')
        parser.add_argument('--image_max', required=True,
                            default=1024,
                            help="max. image size")
        args = parser.parse_args()
        ROOT_DIR = os.path.join("/rwthfs/rz/cluster", os.path.abspath("../.."))
        WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", args.weights_path, args.weights_name + '.h5')
        DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", args.dataset_path)
        IMAGE_MAX = int(args.image_max)
        MASKS = int(args.masks)
            


    class DropletConfig(Config):
        """Configuration for training on the toy  dataset.
        Derives from the base Config class and overrides some values.
        """
        # Give the configuration a recognizable name
        NAME = "droplet"

        # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
        GPU_COUNT = 1

        # Generate detection masks
        #     False: Output only bounding boxes like in Faster-RCNN
        #     True: Generate masks as in Mask-RCNN
        if MASKS == 1:
            GENERATE_MASKS = True
        else: 
            GENERATE_MASKS = False

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # Background + droplet

        # Skip detections with < 90% confidence
        DETECTION_MIN_CONFIDENCE = 0.7

        # Input image resizing
        IMAGE_MAX_DIM = IMAGE_MAX

    ### Configurations
    config = DropletConfig()
    config.display()

    ### Notebook Preferences

    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    if device == 1:
        DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    else:
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    ### Load Model

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                    config=config)
    # Load weights
    print("Loading weights ", WEIGHTS_DIR)
    model.load_weights(WEIGHTS_DIR, by_name=True)

    
    images = []                                                                              
    filenames = []

    filenames_sorted = os.listdir(DATASET_INPUT_DIR+"\\"+VERWENDETE_DATEN)
    filenames_sorted.sort() 
    for filename in filenames_sorted: 
        if not filename.endswith('.json'):
            image = cv2.imread(os.path.join(DATASET_INPUT_DIR+"\\"+VERWENDETE_DATEN, filename))
            images.append(image)                  
            filenames.append(filename)
        
    os.mkdir(DATASET_OUTPUT_DIR+"\\"+VERWENDETE_DATEN)

    for image_num, image in enumerate(images):                                              
        #results = model.detect([image], filename=filenames[image_num], verbose=1)
        results = model.detect([image], verbose=1) 
        r = results[0]
        del r["masks"]
        pickle.dump(r, open(DATASET_OUTPUT_DIR+"\\"+VERWENDETE_DATEN+"\\"+filenames[image_num][6:-3], "wb"))
    
    
