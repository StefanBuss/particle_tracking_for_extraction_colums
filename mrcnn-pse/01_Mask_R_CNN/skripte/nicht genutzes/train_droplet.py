"""
Mask R-CNN
Train on the toy droplet dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

### Necessary Parameters and Data Names

# is the program execution done on cluster? 1 = yes, 0 = no
cluster = 1

### please specify only for non-cluster evaluations 

# Generate detection masks? 1 = yes, 0 = no
masks = 0
# is the program execution done on CPU or GPU? 1=GPU, 0=CPU
device = 1
# Number of images to train with on each GPU. 
# A 12GB GPU can typically handle 2 images of 1024x1024px.
# Adjust based on your GPU memory and image sizes. Use the highest number that your GPU can handle for best performance.
images_gpu = 1
# define base weights
base_weights = "coco"
# max. image size
# The multiple of 64 is needed to ensure smooth scaling of feature
# maps up and down the 6 levels of the FPN pyramid (2**6=64),
# e.g. 64, 128, 256, 512, 1024, 2048, ...
# Select the closest value corresponding to the largest side of the image.
image_max = 1024
# should early stopping be used? 1 = yes, 0 = no
early_stopping = 0
# epochs to train
epochs = 50
# quantity of train/validation dataset in [%]
dataset_quantity = 100
# number of folds for k-fold cross validation
k_fold = 6
# fold number to use for validation. Starting with 0
k_fold_val = 0
# dataset path to find in "Mask_R_CNN\datasets\input\"
dataset_path = r"test_train"                  
# weights path to find in "Mask_R_CNN\droplet_logs\
new_weights_path = r"test"
# Name of the excel result file to find in "Mask_R_CNN\droplet_logs\<WeightsFolderName>\"
name_result_file = "test"

### Initialization

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf
import random
import pandas as pd
from numpy import array

# Root directory of the project
if cluster == 0:
    ROOT_DIR = os.path.abspath("")
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", dataset_path)
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", new_weights_path)
    EXCEL_DIR = os.path.join(WEIGHTS_DIR, name_result_file + '.xlsx')
    BASE_WEIGHTS = base_weights
    IMAGE_MAX = image_max
    EARLY = early_stopping
    EPOCH_NUMBER = epochs
    DATASET_QUANTITY = dataset_quantity
    K_FOLD = k_fold
    K_FOLD_VAL = k_fold_val
    DEVICE = device
    IMAGES_GPU = images_gpu
    MASKS = masks

else:
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect droplets.')
    parser.add_argument('--dataset_path', required=True,
                        metavar="/path/to/droplet/dataset/",
                        help='Directory of the Droplet dataset')
    parser.add_argument('--name_result_file', required=True,
                        metavar="/path/to/droplet/dataset/",
                        help='')    
    parser.add_argument('--new_weights_path', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--base_weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image_max', required=True,
                        default=1024,
                        help="max. image size")
    parser.add_argument('--masks', required=True,
                        default=0,
                        help='Generate detection masks? 1 = yes, 0 = no')
    parser.add_argument('--device', required=True,
                        default=0,
                        help='is the evaluation done on CPU or GPU? 1=GPU, 0=CPU')
    parser.add_argument('--images_gpu', required=True,
                        default=1,
                        help='Number of images to train with on each GPU')
    parser.add_argument('--early_stopping', required=True,
                        default=0,
                        help='enables early stopping')
    parser.add_argument('--epochs', required=True,
                        default=15,
                        help='set number of training epochs, default = 15')
    parser.add_argument('--dataset_quantity', required=True,
                        default=100,
                        help='ratio of train/validation dataset in [%], default = 100')
    parser.add_argument('--k_fold', required=True,
                        default=5,
                        help='# number of folds for k-fold cross validation')       
    parser.add_argument('--k_fold_val', required=True,
                        default=0,
                        help='ratio of train dataset in [%], default = 80')                  
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
              
    ROOT_DIR = os.path.join("/rwthfs/rz/cluster", os.path.abspath("../.."))
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", args.dataset_path)
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", args.new_weights_path)
    EXCEL_DIR = os.path.join(WEIGHTS_DIR, args.name_result_file + '.xlsx')    
    BASE_WEIGHTS = args.base_weights
    IMAGE_MAX = int(args.image_max)
    EARLY = int(args.early_stopping)
    EPOCH_NUMBER = int(args.epochs)
    DATASET_QUANTITY = int(args.dataset_quantity)
    K_FOLD = int(args.k_fold)
    K_FOLD_VAL = int(args.k_fold_val)
    DEVICE = int(args.device)
    IMAGES_GPU = int(args.images_gpu)
    MASKS = int(args.masks)

COMMAND_MODE = "train"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
os.mkdir(WEIGHTS_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(r'D:\logs', "droplet_logs")

############################################################
#  Configurations
############################################################


class DropletConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "droplet"

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    if DEVICE == 1:
        GPU_COUNT = 2
    else:
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
    IMAGES_PER_GPU = IMAGES_GPU

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + droplet

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Input image resizing
    IMAGE_MAX_DIM = IMAGE_MAX


############################################################
#  Dataset
############################################################

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
        annotations_all = []
        annotations = []
        for dataset_folder in sorted(os.listdir(dataset_dir)):
            annotations_quality = json.load(open(os.path.join(dataset_dir, dataset_folder, "train.json")))
            annotations_quality = list(annotations_quality.values()) # don't need the dict keys
            # The VIA tool saves images in the JSON even if they don't have any
            # annotations. Skip unannotated images.
            annotations_quality = [a for a in annotations_quality if a['regions']]
            ### random choice of the training/validation dataset from the existing dataset
            # resetting the random seed to ensure comparability between runs
            np.random.seed(23)
            # define quantity of train/validation dataset
            train_val_set = int(round(DATASET_QUANTITY*len(annotations_quality)/100))
            # random choice of the training/validation dataset
            annotations_quality = np.random.choice(annotations_quality, train_val_set, replace=False)
            # split training/validation dataset in folds
            annotations_quality = np.array_split(annotations_quality,K_FOLD)
            # transponse list for further processing
            annotations_quality = np.transpose(annotations_quality)
            # merging the datasets of different qualities into one dataset
            annotations_all.extend(annotations_quality)
            # save the k-fold splitted training/validation dataset
            pd.DataFrame(annotations_all).to_excel(EXCEL_DIR, header=True, index=False)
            # go through columns of the k-fold splitted training/validation dataset
            for column in range(K_FOLD): 

                annotations = [row[column] for row in annotations_quality]   
                # check if partial dataset is a train or validation dataset
                if subset == "train" and column != K_FOLD_VAL:
                    annotations_use = annotations
                elif subset == "val" and column == K_FOLD_VAL:
                    annotations_use = annotations
                else:
                    continue
                # Add images
                for a in annotations_use:
                    # Get the x, y coordinaets of points of the polygons that make up
                    # the outline of each object instance. These are stores in the
                    # shape_attributes (see json format above)
                    # The if condition is needed to support VIA versions 1.x and 2.x.
                    if type(a['regions']) is dict:
                        polygons = [r['shape_attributes'] for r in a['regions'].values()]
                    else:
                        polygons = [r['shape_attributes'] for r in a['regions']] 

                    # load_mask() needs the image size to convert polygons to masks.
                    # Unfortunately, VIA doesn't include it in JSON, so we must read
                    # the image. This is only managable since the dataset is tiny.
                    image_path = os.path.join(dataset_dir, dataset_folder, a['filename'])
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
            if p['name']=='polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            
            elif p['name']=='ellipse':
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])

            else:
                
                rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
            
            x = np.array((rr, cc)).T
            d = np.array([i for i in x if (i[0] < info["height"] and i[0] > 0)])
            e = np.array([i for i in d if (i[1] < info["width"] and i[1] > 0)])

            rr = np.array([u[0] for u in e])
            cc = np.array([u[1] for u in e])

            if len(rr)==0 or len(cc)==0:
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


def train(model, custom_callbacks=None):
    """Train the model."""

    # Training dataset.
    dataset_train = DropletDataset()
    dataset_train.load_droplet(DATASET_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DropletDataset()
    dataset_val.load_droplet(DATASET_DIR, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCH_NUMBER,
                layers='heads', custom_callbacks=custom_callbacks)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    
    # Validate arguments
    if COMMAND_MODE == "train":
        assert DATASET_DIR, "Argument --dataset is required for training"
    elif COMMAND_MODE == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", BASE_WEIGHTS)
    print("Dataset: ", DATASET_DIR)
    print("Logs: ", WEIGHTS_DIR)

    # Configurations
    if COMMAND_MODE == "train":
        config = DropletConfig()
    else:
        class InferenceConfig(DropletConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if COMMAND_MODE == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=WEIGHTS_DIR, k_fold_val=K_FOLD_VAL)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=WEIGHTS_DIR, k_fold_val=K_FOLD_VAL)

    # Select weights file to load
    if BASE_WEIGHTS.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif BASE_WEIGHTS.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif BASE_WEIGHTS.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = BASE_WEIGHTS

    # Load weights
    print("Loading weights ", weights_path)
    if BASE_WEIGHTS.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    custom_callbacks = []
    
    if EARLY:
        custom_callbacks.append(tf.keras.callbacks.EarlyStopping(patience=EARLY))

    # Train or evaluate
    if COMMAND_MODE == "train":
        train(model, custom_callbacks=custom_callbacks)
    elif COMMAND_MODE == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(COMMAND_MODE))
