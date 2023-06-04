###############################################################
########## Drop size anaylsis program for DisKoPump ###########
##########    Mathias Neufang and Dennis Linden     ###########
##########  Supervisor: Stephan Sibirtsev (AVT.FVT) ###########
###############################################################

### Necessary Parameters and Data Names


### please specify only for non-cluster evaluations 

# Generate detection masks as in Mask-RCNN? If not, output will generate only bounding boxes like in Faster-RCNN.  1 = yes, 0 = no
masks = 0
# Dataset path to find in "Mask_R_CNN\datasets\input\"
dataset_path = r"test2"
# Save path to find in "Mask_R_CNN\datasets\output\
save_path = r"test2m" 
# Name of the excel result file to find in "Mask_R_CNN\datasets\output\
name_result_file = "test2m" 
# n-th result image saved   
save_xth_image = 1      
 # minimum aspect ratio: filters non-round drops            
min_aspect_ratio = 0.8   
# pixel size in [px/µm]. To read from Sopat log file enter pixelsize = 0
pixelsize = 1

### Initialization

import os
import sys
import itertools
import math
import logging
import json
import re
import random
import cv2
import pandas as pd
from collections import OrderedDict
import numpy as np
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from pathlib import Path


# Root directory of the project
ROOT_DIR = os.path.abspath("")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets\\input", dataset_path)
SAVE_DIR = os.path.join(ROOT_DIR, "datasets\\output", save_path)
EXCEL_DIR = os.path.join(SAVE_DIR, name_result_file + '.xlsx')
MASKS = masks

# read pixelsize from JSON-File (if input data is from a Sopat measurement)
if pixelsize == 0:
    sopat_name = (DATASET_DIR + '/' + 'Sopat_Log.json')
    sopat_name_new = (DATASET_DIR + '/Sopat_Log_New.json')

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
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

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

# Configurations
config = DropletConfig()

class DropletDataset(utils.Dataset):

    def load_droplet(self, dataset_dir, subset):
        """Load a subset of the Droplet dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("droplet", 1, "droplet")

        # define path of training/validation dataset
        # dataset_dir = os.path.join(dataset_dir, "all")

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "test.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes']
                            for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "droplet",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a droplet dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "droplet":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'polygon':
                rr, cc = skimage.draw.polygon(
                    p['all_points_y'], p['all_points_x'])

            elif p['name'] == 'ellipse':
                rr, cc = skimage.draw.ellipse(
                    p['cy'], p['cx'], p['ry'], p['rx'])

            else:

                rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])

            x = np.array((rr, cc)).T
            d = np.array(
                [i for i in x if (i[0] < info["height"] and i[0] > 0)])
            e = np.array([i for i in d if (i[1] < info["width"] and i[1] > 0)])

            rr = np.array([u[0] for u in e])
            cc = np.array([u[1] for u in e])

            if len(rr) == 0 or len(cc) == 0:
                continue
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "droplet":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

dataset = DropletDataset()
dataset.load_droplet(DATASET_DIR, None)
dataset.prepare()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Adjust the size attribute to control how big to render images"""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    plt.close()

# Load and display samples
detected_bboxes_total = []
for image_id in dataset.image_ids:
    image_width = image_name=dataset.image_info[image_id]['width']
    image_height = image_name=dataset.image_info[image_id]['height']
    image_name = dataset.image_info[image_id]['id']
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)
    ax = get_ax(1)
    image_length_max = max([image_width,image_height])
    if config.GENERATE_MASKS:
        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, ax=ax, captions=None,
                                    save_dir=SAVE_DIR, img_name=image_name, save_img=True, number_saved_images=save_xth_image)
    else:
        visualize.display_instances(image, bbox, None, class_ids, dataset.class_names, ax=ax, captions=None, show_mask=False,
                                    save_dir=SAVE_DIR, img_name=image_name, save_img=True, number_saved_images=save_xth_image)
    detected_bboxes_total.extend(bbox)

### Calculate Vector For Droplet Size Distribution
bbox_sizes_total = []
edge_tol = image_length_max * 5/100
for i in range(len(detected_bboxes_total)):
    if (detected_bboxes_total[i][0] <= edge_tol or detected_bboxes_total[i][1] <= edge_tol or detected_bboxes_total[i][2] >= image_width-edge_tol or detected_bboxes_total[i][3] >= image_height-edge_tol):
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

### Convert Mean Diameter To Excel
df = pd.DataFrame(mean_diameter_total_resize).to_excel(EXCEL_DIR, header=False, index=False)