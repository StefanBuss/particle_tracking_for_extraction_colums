###############################################################
########## Drop size anaylsis program for DisKoPump ###########
##########    Mathias Neufang and Dennis Linden     ###########
##########  Supervisor: Stephan Sibirtsev (AVT.FVT) ###########
###############################################################      

### Necessary Parameters and Data Names

# n-th result image saved   
save_xth_image = 1      
 # minimum aspect ratio: filters non-round drops            
min_aspect_ratio = 0.8   
# camera setup in micrometer. to read from Sopat log file enter pixelsize = 0
pixelsize = 1.0
# is the evaluation done on cluster? 1 = yes, 0 = no
cluster = 0

### please specify only for non-cluster evaluations 

# Dataset path to find in "Mask_R_CNN\datasets\input\"
dataset_path = r"test"              
# Save path to find in "Mask_R_CNN\datasets\output\
save_path = r"test"                 
# Name of the excel result file to find in "Mask_R_CNN\datasets\output\
name_result_file = "test"         
# Weights path to find in "Mask_R_CNN\droplet_logs\
weights_path = r"Netze_Tropfen"
# Choose Neuronal Network / Epoch to find in "Mask_R_CNN\droplet_logs\
weights_name = r"Quality_Sopat_1"

### Initialization

import os
import json
import sys
import random
import math
import re
import time
import numpy as np

import tensorflow as tf
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
import pandas as pd
from numpy import asarray
from random import random

tf.to_float = lambda x: tf.cast(x, tf.float32)

# Root directory of the project
if cluster == 0:
    ROOT_DIR = os.path.abspath("")
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", weights_path, weights_name + '.h5')
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", dataset_path)
    SAVE_DIR = os.path.join(ROOT_DIR, "datasets/output", save_path)
    DROPLET_DIR = os.path.join(ROOT_DIR, "datasets/droplet")
    EXCEL_DIR = os.path.join(SAVE_DIR, name_result_file + '.xlsx')
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
    args = parser.parse_args()
    ROOT_DIR = os.path.join("/rwthfs/rz/cluster", os.path.abspath("../.."))
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", args.weights_path, args.weights_name + '.h5')
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", args.dataset_path)
    SAVE_DIR = os.path.join(ROOT_DIR, "datasets/output", args.save_path)
    DROPLET_DIR = os.path.join(ROOT_DIR, "datasets/droplet")
    EXCEL_DIR = os.path.join(SAVE_DIR, args.name_result_file + '.xlsx')
    
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "droplet_logs")
#os.mkdir(SAVE_DIR)

if pixelsize == 0:
    sopat_find = [file for file in os.listdir(DATASET_DIR + '/val/') if file.endswith('.json')]
    sopat_name = (DATASET_DIR + '/val/' + sopat_find[0])
    sopat_name_new = (DATASET_DIR + '/val/Sopat_Log.json')

    with open(sopat_name, "r",encoding="utf-8") as sopat_content:
        content_lines = sopat_content.readlines()

    current_line = 1
    with open(sopat_name_new, "w",encoding="utf-8") as sopat_content_new:
        for line in content_lines:
            if current_line == 30:
                pass
            else:
                sopat_content_new.write(line)
            current_line += 1

    sopat_data = json.load(open(sopat_name_new, "r", encoding="utf-8"))

    pixelsize = sopat_data["sopatCamControlAcquisitionLog"]["conversionMicronsPerPx"]
      
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.droplet import train_droplet

### Configurations
config = train_droplet.DropletConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

### Notebook Preferences

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

### Load Validation Set

# Program does not validate but these lines are necessary for the program to work correctly

dataset = train_droplet.DropletDataset()
dataset.load_droplet(DROPLET_DIR, "val") # Dennis: dataset.load_droplet(DATASET_DIR, "train")

#Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

### Load Model

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases

# # Load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", WEIGHTS_DIR)
model.load_weights(WEIGHTS_DIR, by_name=True)

### Run Detection

detected_bboxes_total = []

IMG_DIR = os.path.join(DATASET_DIR, "val")

for filename in os.listdir(IMG_DIR):
    if not filename.endswith('.json'):
        image = cv2.imread(os.path.join(IMG_DIR, filename))          #img = mpimg.imread(os.path.join(folder, filename))
        imgage_width, image_height, channels = image.shape
        image_length_max = max([imgage_width,image_height])
        results = model.detect([image], filename=filename, verbose=1)                  # Dennis: results = model.detect(image[id], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]
        img_name = "result_{}.jpg".format(filename)             # img_name = "results{}.jpg".format(filename)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title=None, save_dir=SAVE_DIR, img_name=img_name, save_img=True, number_saved_images=save_xth_image)
        #plt.show()                                                                 # Dennis: commented
        detected_bboxes_total.extend(r['rois'])


### Calculate Vector For Droplet Size Distribution

bbox_sizes_total = []
edge_tol = image_length_max * 5/100
for i in range(len(detected_bboxes_total)):

    if (detected_bboxes_total[i][0] <= edge_tol or detected_bboxes_total[i][1] <= edge_tol or detected_bboxes_total[i][2] >= imgage_width-edge_tol or detected_bboxes_total[i][3] >= image_height-edge_tol):
    
        continue

    else:

        x_range = abs(detected_bboxes_total[i][1] - detected_bboxes_total[i][3])
        y_range = abs(detected_bboxes_total[i][0] - detected_bboxes_total[i][2])
        
        bbox_sizes_total.append((x_range,y_range))
    
print(bbox_sizes_total)

### Calculate Mean Diameter

mean_diameter_total = []

for i in range(len(bbox_sizes_total)):
    
    if (bbox_sizes_total[i][0] / bbox_sizes_total[i][1] >= 1/min_aspect_ratio or bbox_sizes_total[i][0] / bbox_sizes_total[i][1] <= min_aspect_ratio):
    
        continue                   # if function filters the droplets which are not round (mostly wrong detected droplets)

    else:

        mean_diameter_total.append(abs(bbox_sizes_total[i][0] + bbox_sizes_total[i][1]) / 2)

print(mean_diameter_total)

### Translate Mean Diameter In Actual Droplet Sizes (mm)

# recalculate mean diameter
mean_diameter_total_resize = [(i * pixelsize / 1000) for i in mean_diameter_total]

# save mean_diameter_total_true_resize
with open("Drops_30_10_500.txt", "wb") as fp:   #Pickling
    pickle.dump(mean_diameter_total_resize, fp)

### Convert Mean Diameter To Excel

df = pd.DataFrame(mean_diameter_total_resize).to_excel(EXCEL_DIR, header=False, index=False)