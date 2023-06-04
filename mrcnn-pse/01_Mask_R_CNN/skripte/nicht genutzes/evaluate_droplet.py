###############################################################
########## Drop size anaylsis program for DisKoPump ###########
##########    Mathias Neufang and Dennis Linden     ###########
##########  Supervisor: Stephan Sibirtsev (AVT.FVT) ###########
###############################################################      

### Necessary Parameters and Data Names

# is the evaluation done on cluster? 1 = yes, 0 = no
cluster = 0
# n-th result image saved
save_xth_image = 1
 # minimum aspect ratio: filters non-round drops
min_aspect_ratio = 0.5
# pixel size in [µm/px]. To read from Sopat log file enter pixelsize = 0
pixelsize = 15.12162646  # Telezentrisch: 14.90757305 # Ailin:6.138581149     #neu: 15.12162646      
# 
splitratio=[0.9, 0.1]
split_ratio = "0.9 : 0.1" #jofa

### please specify only for non-cluster evaluations

# Generate detection masks as in Mask-RCNN? If not, output will generate only bounding boxes like in Faster-RCNN.  1 = yes, 0 = no
masks = 0
# max. image size
# The multiple of 64 is needed to ensure smooth scaling of feature
# maps up and down the 6 levels of the FPN pyramid (2**6=64),
# e.g. 64, 128, 256, 512, 1024, 2048, ...
# Select the closest value corresponding to the largest side of the image.
image_max = 1024
# is the evaluation done on CPU or GPU? 1=GPU, 0=CPU
device = 1
# Dataset path to find in "Mask_R_CNN\datasets\input\"
#dataset_path = r"test_aufstieg"
dataset_path = r"MV-CH050-10UM (02F72748253)"
# Save path to find in "Mask_R_CNN\datasets\output\
save_path = r"test_aufstieg_res"                 
# Name of the excel result file to find in "Mask_R_CNN\datasets\output\
name_result_file = "test_aufstieg_res"         
# Weights path to find in "Mask_R_CNN\droplet_logs\
weights_path = r"Netze_Tropfen"
# Choose Neuronal Network / Epoch to find in "Mask_R_CNN\droplet_logs\
weights_name = r"mask_rcnn0_droplet_1305"

### Info for sedimentation model
'''Properties for sedimentation model'''
d_um = 7.1 * 10 ** -3  # [m]
a15 = 1.52
a16 = 4.5
alpha_um = 8
alpha_def = 8
dh = 0.002
phi_st = 0.22
alpha_y = 0.15 # holdup
D = 0.05 # [m] # column diameter
n = 2 # Swarm Exponent
V_x = 0.38*(4/8) # [l/min] volume stream of continiuous phase 

'''Solvent properties'''
rho_c = 999 # [kg/m^3]
rho_d = 864.4 # [kg/m^3]
eta_d = 0.563*10**(-3)# [mPa*s]
eta_c = 0.939*10**(-3)  # [mPa*s]
sigma = 34* 10 ** -3  # [N/m]
g = 9.81            # [m/s^2] 

'''Camera settings'''
framerate = 300             # adapt to framerate of dataset
timediff = 1/framerate     # [s]
stroke = 0 # (0.4*1000)/pixelsize # (0.512*1000*2)/pixelsize
#Hub 10mm, 1.25 Hz, also pro Sek. 8mm Hub. Hebt nur Hälfte der Zeit, also in 12,5 Bildern. Pro Bild 0,64mm. Sicherheitsfaktor 2. framerate15: 1.33, framerate25: 0.512, framerate 50: 0.32

### Initialization

from cmath import sqrt
import os
#import fnmatch
import json
import sys
import random
import math
import re
import time
import statistics
#import winreg #anpa
#from winreg import EnumValue #jofa
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
from PIL import Image                      
from PIL.ExifTags import TAGS
import csv #anpa

import pickle

tf.to_float = lambda x: tf.cast(x, tf.float32)
# Root directory of the project
if cluster == 0:
    ROOT_DIR = os.path.abspath("")
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", weights_path, weights_name + '.h5')
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", dataset_path)
    SAVE_DIR = os.path.join(ROOT_DIR, "datasets/output", save_path)
    EXCEL_DIR = os.path.join(SAVE_DIR, name_result_file + '.xlsx')
    EXCEL_name_DIR = os.path.join(SAVE_DIR, name_result_file + '.xlsx') ##
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
    SAVE_DIR = os.path.join(ROOT_DIR, "datasets/output", args.save_path)
    EXCEL_DIR = os.path.join(SAVE_DIR, args.name_result_file + '.xlsx')
    EXCEL_name_DIR = os.path.join(SAVE_DIR, args.name_result_file + args.weights_name + '.xlsx') ##
    IMAGE_MAX = int(args.image_max)
    MASKS = int(args.masks)
    
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "droplet_logs")
os.mkdir(SAVE_DIR)

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

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

### Load Model

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
# Load weights
print("Loading weights ", WEIGHTS_DIR)
model.load_weights(WEIGHTS_DIR, by_name=True)

############################################
### Detection of Droplets and Velocities ###   # Anpa jf   
############################################

'''Used lists and parameters'''                 
images = []                                                                              
filenames = []
velocitiesNew_df_obj = pd.DataFrame(columns=['ImageNumber','ImageName', 'Diameter', 'VelocityVert', 'VelocityHori', 'VelocityTotal', 'RGB' ,'MaxAscent', 'DistAscent'])  #anpa: RGB, MaxAscent

'''Droplet classification'''
class Droplet:
    def __init__(self, roi, img, img_nmb, rgb):
        self.roi = roi
        self.img = img
        self.img_nmb = img_nmb
        self.rgb = rgb
        self.height = abs(roi[2]-roi[0])
        self.width = abs(roi[3]-roi[1])
        self.center_y = abs(roi[0]+(roi[2]-roi[0])/2)
        self.center_x = abs(roi[1]+(roi[3]-roi[1])/2)
        self.diameter = ((self.height+2*self.width)/3)*pixelsize*1/1000 #[mm] #aawe
        self.max_size_diff_obj = None
        self.max_ascent_obj = None
        self.dist_ascent_obj = None #anpa
        self.max_descent_obj = None # int(stroke/framerate) #anpa
        self.velocity_vert = None #anpa
        self.velocity_hori = None #anpa
        self.velocity_total = None #anpa
    
    def get_max_size_diff_obj(self, diameter_def, lower_threshold_def):
        if diameter_def <= lower_threshold_def:
            self.max_size_diff_obj = 0.1*diameter_def
        else: 
            self.max_size_diff_obj = 0.25*diameter_def

    def get_max_ascent_obj(self, diameter):
        diameter = diameter*1/1000 #Umwandlung des Durchmessers von [mm] zu [m]
        v_infty = get_max_ascent_velocity(diameter)
        if math.isnan(v_infty) == True: # If diameter too small, v_infty = NaN
            #max_velo_droplet = abs(2/9*(((diameter_droplet/2)**2)*g*(rho_d-rho_c))/eta_c) #[m/s], Stokes-Aufstieg
            self.max_ascent_obj = 0 # [px]
            self.max_descent_obj = 20 #anpa arbitrary value  
        else:
            self.max_ascent_obj = int((v_infty*1000000*timediff)/pixelsize + stroke) 
            self.max_descent_obj = int((v_infty*1000000*timediff)/pixelsize + stroke) # [px]
        

#################################################################
#anpa
# consistency check
    '''Determine Ascended Distance in Pixel'''
    def get_dist_ascent_obj(self, droplet_before):
        #self.dist_ascent_obj = abs((self.center_y-droplet_before.center_y)) #anpa
        #self.dist_ascent_obj = self.center_y-droplet_before.center_y
        self.dist_ascent_obj = droplet_before.center_y-self.center_y
#################################################################

    def get_velocity_obj(self, droplet_before):
        #self.velocity = abs(((self.center_y-droplet_before.center_y)*pixelsize)/timediff)*(1/1000) #anpa
        #self.velocity = (((self.center_y-droplet_before.center_y)*pixelsize)/timediff)*(1/1000)
        self.velocity_vert = (((droplet_before.center_y-self.center_y)*pixelsize)/timediff)*(1/1000)
        self.velocity_hori = (((droplet_before.center_x-self.center_x)*pixelsize)/timediff)*(1/1000)
        self.velocity_total = (self.velocity_hori**2 + self.velocity_vert**2)**(1/2)






    
'''Used functions'''
def get_max_ascent_velocity(diameter): #aawe
    Ar = (rho_c * abs(rho_c - rho_d) * g * (diameter ** 3)) / (eta_c ** 2)
    c_w = 432 / Ar + 20 / (Ar ** (1/3)) + (0.51 * (Ar ** (1/3))) / (140 + Ar ** (1/3)) # drag coefficient
    Re_ball = np.sqrt(4/3 * Ar / c_w) # Reynoldszahl eines runden Tropfens
    Re_bubble = Ar / (12 * ((0.065 * Ar + 1) ** (1/6))) # Reynoldszahl einer Blase
    f2 = 1 - (1 / (1 + (diameter / d_um) ** alpha_um))
    K_Hr = (3 * (eta_c + eta_d / f2)) / (2 * eta_c + 3 * eta_d / f2)
    f1 = 2 * (K_Hr - 1)
    v_rund = ((1 - f1) * Re_ball + f1 * Re_bubble) * eta_c / (diameter * rho_c) # Geschwindigkeit eines runden Tropfens
    v_os = np.sqrt(2 * a15 * sigma / (rho_d * diameter)) # Geschwindigkeit eines oszillierenden Tropfens
    v_schirm = np.sqrt(abs(rho_c - rho_d) * g * diameter / (2 * rho_c)) # Geschwindigkeit eines schirmförmigen Tropfens 
    v_def = (v_os ** alpha_def + v_schirm ** alpha_def) ** (1 / alpha_def) # Übergangsbereich von oszillierend zu schirmförmig 
    v_infty = (v_rund * v_def) / (v_rund ** a16 + v_def ** a16) ** (1 / a16) # finale Sedimentationsgeschwindigkeit (alle Geschwindigkeiten in m/s)
    return v_infty

def velocitiesNew(img_nmb, imgname, droplet_diameter, velocity_drop_vert, velocity_drop_hori, velocity_drop_total, rgb, max_ascent, dist_ascent, dataframe):
    now = {'ImageNumber':img_nmb,'ImageName': imgname, 'Diameter': droplet_diameter, 'VelocityVert': velocity_drop_vert, 'VelocityHori': velocity_drop_hori, 'VelocityTotal': velocity_drop_total, 'RGB':rgb, 'MaxAscent':max_ascent, 'DistAscent':dist_ascent}
    data = dataframe.append(now, ignore_index = True)
    return data

def split_two(lst, ratio):  #anpa:
    assert(np.sum(ratio) == 1.0)  
    train_ratio = ratio[0]
    indices_for_splittin = [int(len(lst) * train_ratio)]
    train, test = np.split(lst, indices_for_splittin)
    return train, test

def get_upper_lower(all_diameters_obj_def,ratio):
    sorted_diameters = sorted(all_diameters_obj_def, key = float, reverse=False)
    lower, upper = split_two(sorted_diameters,ratio)
    lower_threshold = max(lower)
    return lower_threshold

def size_distributionNew(list_diameters, list_velos, step_size):
    step_list = np.arange(0, max(list_diameters)+step_size, step_size)
    M_dis = np.zeros(shape=(len(step_list), 9))
    M_dis[:,0] = step_list
    velos_st_dev = []
    velos_st_dev_pos = []
    for i in range (0, len(step_list)):
        velos_st_dev.append([]) # keine andere Option leere List zu machen
        velos_st_dev_pos.append([]) 
    for help, x in enumerate(list_diameters): # Schleife über alle Tropfen(-Durchmesser)
        for i in range(0, len(step_list)): # Schleife über alle Bins
            if x < step_list[i]: # Wenn Durchmesser kleiner als
                M_dis[i,1] += 1 # Zähle Tropfen einer Klasse
                velos_st_dev[i].append(list_velos[help]) # Wozu? list... legt bin fest
                if list_velos[help] > 0:
                    M_dis[i,6] += 1
                    velos_st_dev_pos[i].append(list_velos[help])
                break
    for i in range(len(M_dis)):
        M_dis[i,2] = M_dis[i,1]/len(list_diameters) # Relative Anzahl der Tropfen bestimmen
        if M_dis[i,1] != 0: 
            M_dis[i,3] = statistics.mean(velos_st_dev[i])
        if M_dis[i,6] != 0:
            M_dis[i,7] = statistics.mean(velos_st_dev_pos[i])
        if len(velos_st_dev[i]) != 0: # Prüfe ob bin leer
            M_dis[i,4] = statistics.pstdev(velos_st_dev[i])
            M_dis[i,5] = statistics.median(velos_st_dev[i])
        if len(velos_st_dev_pos[i]) != 0: # Prüfe ob bin leer
            M_dis[i,8] = statistics.pstdev(velos_st_dev_pos[i])
    df_dis = pd.DataFrame(data=M_dis, columns=['DropletClasses','NumberOfDrops','relNumOfDrops', 'AverageVelocity', 'StDev', 'MedianVelo', 'DropletClassPos','AverageVelocityPos','StDevPos'])
    return df_dis



'''Droplet detection programm'''
filenames_sorted = os.listdir(DATASET_DIR)
filenames_sorted.sort() 
for filename in filenames_sorted: 
    if not filename.endswith('.json'):
        image = cv2.imread(os.path.join(DATASET_DIR, filename))
        images.append(image)                  
        filenames.append(filename)
# print(filenames) #anpa : test, if file order is correct

detected_bboxes_total = [] #anpa
detected_images_total = [] #anpa
rgb_list = [] #anpa
all_droplets_obj = []
all_diameters_obj = [] 

counter_1 = save_xth_image # anpa
for image_num, image in enumerate(images):                                              
        imgage_width, image_height, channels = image.shape
        image_length_max = max([imgage_width,image_height])
        results = model.detect([image], filename=filenames[image_num], verbose=1)    
        all_droplets_obj.append([])
        # Display results 
        ax = get_ax(1)
        r = results[0]
        image_name_per_drop = []  #anpa
        rgb = image.mean(axis=0).mean(axis=0)[1] #anpa
        rgb_list.append(rgb) #anpa
        img_name = str(image_num) + "_result_{}".format(filenames[image_num])   #anpa --> Benennung der result Bilder  
        # distinction between with or without mask detection     #((self.height+2*self.width)/3)*pixelsize*1/1000 #[mm]
        ###########################################
        detected_bboxes_total.extend(r['rois'])
        image_name_per_drop = [img_name]*len(r['rois']) #anpa
        detected_images_total.extend(image_name_per_drop) #anpa
        rois_current_picture = r["rois"]
        diameters_current_picture = []
        for l, roi in enumerate(rois_current_picture):
            droplet = Droplet(roi, img_name, image_num, rgb)
            all_droplets_obj[image_num].append(droplet)
            all_diameters_obj.append(droplet.diameter)
            height = abs(roi[2]-roi[0])
            width = abs(roi[3]-roi[1])
            diameter = ((height+2*width)/3)*pixelsize*1/1000
            diameters_current_picture.append(diameter)
        

        print(results)

        ##############################################
        if config.GENERATE_MASKS:
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    'droplet', r['scores'], ax=ax,
                                    title=None, save_dir=SAVE_DIR, img_name=img_name, save_img=True, number_saved_images=save_xth_image, counter_1=counter_1) #anpa
        else:
            visualize.display_instances(image, r['rois'], None, r['class_ids'], 
                                    'droplet', diameters_current_picture, ax=ax, show_mask=False,
                                    title=None, save_dir=SAVE_DIR, img_name=img_name, save_img=True, number_saved_images=save_xth_image, counter_1=counter_1)  #anpa
        plt.close('all')
        ##########################################
        if counter_1 == save_xth_image:
            counter_1 = 0
        counter_1 = counter_1 + 1
        ############################################

        pickle.dump(r, open(ROOT_DIR+"\\datasets\\output\\r_"+filenames[image_num][:-3], "wb"))







'''Determining ascended droplets and their velocities'''
lower_treshhold = get_upper_lower(all_diameters_obj,splitratio)
all_velos_obj = []
all_diameters_detected = []
all_max_ascend_detected = [] #anpa
all_dist_ascend_detected = [] #anpa
memory = []
drops_before_obj = []

#counter = 0 #anpa weg!!!
for image_num, droplets in enumerate(all_droplets_obj):
    droplet_bin = []
    if image_num == 0:
        for each_current in droplets:
            drops_before_obj.append(each_current) #drop_before_obj = Tropfen auf vorherigem Bild
    else:
        for each_current in droplets: # each_current = aktuell iterierter Tropfen
            each_current.get_max_size_diff_obj(each_current.diameter, lower_treshhold)
            each_current.get_max_ascent_obj(each_current.diameter) #aawe
            candidates_obj = []
            diffs_candidates = []
            memory.append(each_current)
            for each_before in (drops_before_obj): # iteriere alle Tropfen auf vorherigem Bild
            # Prüfungshierarchie: 
            # 1. Stufe: Prüfung ob überhaupt Tropfen detektiert wurden.
            # 2. Stufe: Prüfung des in Frage kommenden Bereichs -> Definition von Kandidaten (candidates_obj)
            # 3. Stufe: Wenn nur ein Kandidat -> Trivial. Wenn mehr: Größenvergleich
                if each_before not in droplet_bin: 
                    if (each_current.center_y) - (each_before.center_y) in range(int(-1*each_current.max_ascent_obj), int(each_current.max_descent_obj)) and ((each_current.center_x) - (each_before.center_x)) in range(int(-1*each_current.max_ascent_obj), int(each_current.max_descent_obj)): #anpa früher range(-100, 100)
                        if abs(each_current.diameter-each_before.diameter) <= each_current.max_size_diff_obj:
                            candidates_obj.append(each_before)
            if len(candidates_obj) == 1: 
                each_current.get_velocity_obj(candidates_obj[0])
                each_current.get_dist_ascent_obj(candidates_obj[0])
                if each_current.velocity_vert >= -50: #prüfe ob Geschwindigkeit hoch genug ist?
                    all_velos_obj.append(each_current.velocity_vert)
                    all_diameters_detected.append(each_current.diameter)
                    droplet_bin.append(candidates_obj[0])
                    velocitiesNew_df_obj = velocitiesNew(each_current.img_nmb, each_current.img, each_current.diameter, each_current.velocity_vert, each_current.velocity_hori, each_current.velocity_total, each_current.rgb, each_current.max_ascent_obj, each_current.dist_ascent_obj, velocitiesNew_df_obj) #anpa ascent
                #else:
                    #counter += 1
                    #print("bla")
            elif len(candidates_obj) >1: 
                for index, candidate in enumerate(candidates_obj):
                    diffs_candidates.append(abs(each_current.diameter - candidate.diameter))
                smallest_diff_obj = min(diffs_candidates)
                each_current.get_velocity_obj(candidates_obj[diffs_candidates.index(smallest_diff_obj)])
                each_current.get_dist_ascent_obj(candidates_obj[diffs_candidates.index(smallest_diff_obj)])
                if each_current.velocity_vert >= -50:
                    all_velos_obj.append(each_current.velocity_vert)
                    all_diameters_detected.append(each_current.diameter)
                    droplet_bin.append(candidates_obj[diffs_candidates.index(smallest_diff_obj)])
                    velocitiesNew_df_obj = velocitiesNew(each_current.img_nmb, each_current.img, each_current.diameter, each_current.velocity_vert, each_current.velocity_hori, each_current.velocity_total, each_current.rgb, each_current.max_ascent_obj, each_current.dist_ascent_obj, velocitiesNew_df_obj) #anpa ascent
                #else:
                    #counter += 1
                    #print("bla")
        drops_before_obj = memory
        memory = []






'''Create Excel with sedimentation model'''
diam = []
vel = [] 
u_x = V_x / ((0.25 * np.pi * D ** 2) *(1-alpha_y))  * (1/60000) # [m/s]

diameter = 0.0002
while diameter < 0.006:
    v_infty = get_max_ascent_velocity(diameter)
    pi_sigma = sigma * (rho_c ** 2 / (eta_c ** 4 * (rho_c - rho_d) * g)) ** (1 / 3)  # dimensionslose Oberflächenspannung
    k_s = 1.406 * phi_st ** 0.145 * pi_sigma ** (-0.028) * np.exp(-0.129 * (diameter / dh) ** 1.134 * (1 - phi_st) ** (-2.161))
    u_y = v_infty * (1 - alpha_y) ** (n-1) * k_s - u_x
    diam.append(diameter*1000)     # zu mm
    vel.append(u_y*1000)    # zu mm/s
    diameter += 0.0002
df_model = pd.DataFrame(diam, vel)

#################################################### anpa #####################################################
### Calculate Vector For Droplet Size Distribution
bbox_sizes_total = []
detected_images_prefiltered =[] #anpa
edge_tol = image_length_max * 5/100
for i in range(len(detected_bboxes_total)):

    if (detected_bboxes_total[i][0] <= edge_tol or detected_bboxes_total[i][1] <= edge_tol or detected_bboxes_total[i][2] >= imgage_width-edge_tol or detected_bboxes_total[i][3] >= image_height-edge_tol):
        continue
    else:
        x_range = abs(detected_bboxes_total[i][1] - detected_bboxes_total[i][3])
        y_range = abs(detected_bboxes_total[i][0] - detected_bboxes_total[i][2])
        bbox_sizes_total.append((x_range,y_range))
        detected_images_prefiltered.append(detected_images_total[i]) #anpa
#print(bbox_sizes_total) ###

### Calculate Mean Diameter
mean_diameter_total = []
detected_images_final=[]
for i in range(len(bbox_sizes_total)):  
    if (bbox_sizes_total[i][0] / bbox_sizes_total[i][1] >= 1/min_aspect_ratio or bbox_sizes_total[i][0] / bbox_sizes_total[i][1] <= min_aspect_ratio):
        continue                   # if function filters the droplets which are not round (mostly wrong detected droplets)
    else:
        detected_images_final.append(detected_images_prefiltered[i])
        mean_diameter_total.append(abs(bbox_sizes_total[i][0] + bbox_sizes_total[i][1]) / 2)
print(mean_diameter_total)
print(detected_images_final) ##
### Translate Mean Diameter In Actual Droplet Sizes (mm)
# recalculate mean diameter
mean_diameter_total_resize = [(i * pixelsize / 1000) for i in mean_diameter_total]
# get unique image names
def get_unique_numbers(detected_images_final):
    unique = []
    for image in detected_images_final:
        if image in unique:
            continue
        else:
            unique.append(image)
    return unique
uniqueList=get_unique_numbers(detected_images_final)

# sort information according to each image
myDF = pd.DataFrame(columns=['ImageNumber','ImageName','NumberOfDrops','Sauter','DropDiameters'])
counterStart=0
for numb, image2 in enumerate(uniqueList):
    dropsByImg=[]
    counter = counterStart
    for image1 in detected_images_final:
        if image2 is image1:
            dropsByImg.append(mean_diameter_total_resize[counter])
            counter = counter + 1
            counterStart = counter
    sauterByImg = sum(pow(value, 3) for value in dropsByImg)/sum(pow(value, 2) for value in dropsByImg)
    numberByImg = len(dropsByImg)
    q={'ImageNumber':numb, 'ImageName':image2,'NumberOfDrops':numberByImg,'Sauter':sauterByImg,'DropDiameters':[dropsByImg]}
    myDF = myDF.append(q, ignore_index=True)


# save mean_diameter_total_true_resize
#with open("Drops_30_10_500.txt", "wb") as fp:   #Pickling
#    pickle.dump(mean_diameter_total_resize, fp)

# save image name and detected drops in one sheet, save information per image in second sheet
df = pd.DataFrame(mean_diameter_total_resize,detected_images_final)
with pd.ExcelWriter(EXCEL_name_DIR) as writer:
    df.to_excel(writer,sheet_name='test_name')
    myDF.to_excel(writer,sheet_name='test_name2')







'''function for determining droplet size distribution (yada)'''
def size_distribution(list, step_size):
    # definition of droplet class limits
    step_list = np.arange(0, max(list)+step_size, step_size)
    M_dis = np.zeros(shape=(len(step_list), 3))
    M_dis[:,0] = step_list
    # 'find interval' function 
    for x in list:
        for i in range(0, len(step_list)):
            if x < step_list[i]:
                M_dis[i,1] = M_dis[i,1] + 1
                break
    # calculate relative droplet size distribution
    for i in range(len(M_dis)):
        M_dis[i,2] = M_dis[i,1]/len(list)
    df_dis = pd.DataFrame(data=M_dis, columns=['DropletClasses','NumberOfDrops','relNumOfDrops'])
    return df_dis
#########################################################################################################################





'''determine sauter diamater'''
sauter_alldropsdetected = sum(pow(value, 3) for value in all_diameters_obj)/sum(pow(value, 2) for value in all_diameters_obj)
sauter_onlyascended = sum(pow(value, 3) for value in all_diameters_detected)/sum(pow(value, 2) for value in all_diameters_detected)
sauter_data = {"sauter all drops": [sauter_alldropsdetected], "sauter only ascended": [sauter_onlyascended]}
sauter_df = pd.DataFrame(data=sauter_data)





'''Dataframes for Excel'''                                                       
settings_data = {'cluster':[cluster], 'save_xth_image':[save_xth_image], 'min_aspect_ratio':[min_aspect_ratio], 'pixelsize':[pixelsize], 'masks':[masks], 'image_max':[image_max], 'device':[device], 'dataset_path':[dataset_path], 'save_path':[save_path], 'name_result_file':[name_result_file], 'weights_name':[weights_name], 'framerate': [framerate], 'split-ratio': [split_ratio]}
settings_df = pd.DataFrame(data=settings_data)
distrib_velo_obj_df =  size_distributionNew(all_diameters_detected, all_velos_obj, 0.2) 
distribution_df = size_distribution(mean_diameter_total_resize, 0.2) #anpa
with pd.ExcelWriter(EXCEL_name_DIR) as writer:
    settings_df.to_excel(writer, sheet_name = 'settings')                                                            
    velocitiesNew_df_obj.to_excel(writer, sheet_name = 'velocitiesNewobj')
    distrib_velo_obj_df.to_excel(writer, sheet_name = 'velo_distribution_obj')
    df_model.to_excel(writer,sheet_name = 'Geschwindigkeiten_Modell')
    sauter_df.to_excel(writer, sheet_name = "Sauters")
    df.to_excel(writer, sheet_name = 'droplets')
    myDF.to_excel(writer, sheet_name = 'images')
    distribution_df.to_excel(writer, sheet_name = 'droplet_size_distribution')






"Dataframes for CSV" #anpa
frames = pd.DataFrame({'idx':['settings','velocitiesNewobj','velo_distribution_obj','Geschwindigkeiten_Modell',"Sauters",'droplets','images','droplet_size_distribution'],
 'dfs':[settings_df,velocitiesNew_df_obj,distrib_velo_obj_df,df_model,sauter_df,df,myDF,distribution_df]})
csv_DIR = os.path.join(SAVE_DIR,"csv_data") #create nested dataframe
os.mkdir(csv_DIR) #create csv folder
i = 0
while i < len(frames['idx']):
    csv_path = os.path.join(csv_DIR, f"{frames['idx'][i]}.csv") # iterate through all dataframes
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(frames['dfs'][i])
        # iterate over rows
        for entrie in  frames['dfs'][i].values:
            writer.writerow(entrie)
    i += 1
