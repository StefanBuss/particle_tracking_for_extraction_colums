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

# is the evaluation done on cluster? 1 = yes, 0 = no
cluster = 0
# is the evaluation done on CPU or GPU? 1=GPU, 0=CPU
device = 1
# Number of images to train with on each GPU. 
# A 12GB GPU can typically handle 2 images of 1024x1024px.
# Adjust based on your GPU memory and image sizes. Use the highest number that your GPU can handle for best performance.
images_per_gpu = 1

### please specify only for non-cluster evaluations 

# train or splash?
command = "train"
# define base weights
base_weights = "coco"
# should early stopping be used? 1 = yes, 0 = no
early_stopping = 0
# epochs to train
epochs = 50
# quantity of train/validation dataset
train_val_set = 80
# ratio of train dataset in [%]
train_ratio = 80
# dataset path to find in "Mask_R_CNN\datasets\input\"
dataset_path = r"test"                  
# weights path to find in "Mask_R_CNN\droplet_logs\
new_weights_path = r"test"

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

# Root directory of the project
if cluster == 0:
    ROOT_DIR = os.path.abspath("")
    COMMAND_MODE = command
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", dataset_path)
    BASE_WEIGHTS = base_weights
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", new_weights_path)
    EARLY = early_stopping
    EPOCH_NUMBER = epochs
    TRAIN_VAL_SET = train_val_set
    TRAIN_RATIO = train_ratio

else:
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect droplets.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset_path', required=False,
                        metavar="/path/to/droplet/dataset/",
                        help='Directory of the Droplet dataset')
    parser.add_argument('--new_weights_path', required=False,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--base_weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--device', required=False,
                        default=0,
                        help='is the evaluation done on CPU or GPU? 1=GPU, 0=CPU')
    parser.add_argument('--images_per_gpu', required=False,
                        default=1,
                        help='Number of images to train with on each GPU')
    parser.add_argument('--early_stopping', required=False,
                        default=0,
                        help='enables early stopping')
    parser.add_argument('--epochs', required=False,
                        default=15,
                        help='set number of training epochs, default = 15')
    parser.add_argument('--train_val_set', required=True,
                        help='quantity of train/validation dataset')
    parser.add_argument('--train_ratio', required=False,
                        default=80,
                        help='ratio of train dataset in [%], default = 80')                    
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
              
    ROOT_DIR = os.path.join("/rwthfs/rz/cluster", os.path.abspath("../.."))
    COMMAND_MODE = args.command
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", args.dataset_path)
    BASE_WEIGHTS = args.base_weights
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", args.new_weights_path)
    EARLY = args.early_stopping
    EPOCH_NUMBER = args.epochs
    TRAIN_VAL_SET = args.train_val_set
    TRAIN_RATIO = args.train_ratio

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
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
    if device == 1:
        GPU_COUNT = 2
    else:
        GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + droplet

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class DropletDataset(utils.Dataset):

    def load_droplet(self, dataset_dir, subset, random_seed):
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
        annotations_all = json.load(open(os.path.join(dataset_dir, "train.json")))
        annotations_all = list(annotations_all.values()) # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations_all = [a for a in annotations_all if a['regions']]
        ### random choice of the training/validation dataset from the existing dataset
        # resetting the random seed to ensure comparability between runs
        np.random.seed(17)
        #print(len(annotations_all))
        # random choice of the training/validation dataset 
        annotations_all = np.random.choice(annotations_all, TRAIN_VAL_SET, replace=False)
        # define training dataset quantity
        train_set = int(TRAIN_RATIO*TRAIN_VAL_SET/100)
        # define indicies of the training/validation dataset
        annotations_indices_all = np.arange(TRAIN_VAL_SET)
        # distinction between training and validation
        if subset == "train":
            # resetting to the random seed defined at the beginning of the training run to ensure difference between training and validation dataset
            np.random.seed(random_seed)
            # random choice of the training dataset indices
            annotations_indices_train = np.random.choice(annotations_indices_all, train_set, replace=False)
            # define training dataset
            annotations_train = annotations_all[annotations_indices_train]
            annotations = annotations_train       
        elif subset == "val": 
            # resetting to the random seed defined at the beginning of the training run to ensure difference between training and validation dataset
            np.random.seed(random_seed)
            # random choice of the training dataset indices
            annotations_indices_train = np.random.choice(annotations_indices_all, train_set, replace=False)
            # define validation dataset indices
            annotations_indices_val = list(set(annotations_indices_all) - set(annotations_indices_train)) + list(set(annotations_indices_train) - set(annotations_indices_all))
            # define training dataset
            annotations_val = annotations_all[annotations_indices_val]
            annotations = annotations_val

        # Add images
        for a in annotations:
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
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "droplet",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
        
        #annotations_circle = [b for b in annotations_circle if b['']]

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

        #print(type(mask))
        
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
    # define random seed ensure difference between training and validation dataset
    random_seed = random.randint(0, 1000)
    # Training dataset.
    dataset_train = DropletDataset()
    dataset_train.load_droplet(DATASET_DIR, "train", random_seed)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DropletDataset()
    dataset_val.load_droplet(DATASET_DIR, "val", random_seed)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=int(EPOCH_NUMBER),
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
                                  model_dir=WEIGHTS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=WEIGHTS_DIR)

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
        custom_callbacks.append(tf.keras.callbacks.EarlyStopping(patience=int(EARLY)))

    # Train or evaluate
    if COMMAND_MODE == "train":
        train(model, custom_callbacks=custom_callbacks)
    elif COMMAND_MODE == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(COMMAND_MODE))
