
ROOT_DIR = "C:\\Users\\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"

import shutil
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import time 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import scipy
from scipy import signal
from scipy.optimize import curve_fit
import sys
import random
import pandas as pd
import moviepy.video.io.ImageSequenceClip

list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90", "20221124_Oben_1,2_1_100",
                "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90", "20221124_Unten_1,2_1_100",

                "20221125_Oben_3_1_40","20221125_Oben_3_1_80","20221125_Oben_3_1_90_ET_250","20221125_Oben_3_1_90_ET_500","20221125_Oben_3_1_100_ET_250","20221125_Oben_3_1_100_ET_500",
                "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", "20221125_Unten_3_1_100",

                "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90", "20221129_Oben_1_2_100",
                "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90", "20221129_Unten_1_2_100",
                             
                "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90", "20221130_Oben_1,2_1_100",
                "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90", "20221130_Unten_1,2_1_100"]


abstand_vom_rand = 10
#FIXE PARAMETER
alpha_gr = 8
d_um = 7.1 * 10 ** -3  # [m]
a15 = 1.52
a16 = 4.5
alpha_um = 8
alpha_def = 8
D = 0.05 # [m] # column diameter
A_col = 0.25 * np.pi * D ** 2  # [m^2]

phi_st = 0.22
dh = 0.002
rho_c = 999 # [kg/m^3]
rho_d = 864.4 # [kg/m^3]
eta_d = 0.563*10**(-3)# [Pa*s]
eta_c = 0.939*10**(-3)  # [Pa*s]
sigma = 34* 10 ** -3  # [N/m]
g = 9.81            # [m/s^2]

pixelsize = 15.12162646 #[µm/px]
start = 0

def load_data():
    list_elem = []
    for elem in os.listdir(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN):
        list_elem.append(int(elem))
    
    list_elem.sort()

    size = Image.open(ROOT_DIR+"\\datasets\\input\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[0])+".jpg").size
    bild_groeße_x, bild_groeße_y = int(size[0]), int(size[0])

    #HIER WERDEN NUR DIE BOUDNING BOXES SELEKTIERT, DIE NICHT ZU NAH AM RAND SIND
    list_r = []
    for elem in list_elem:
        r, r_selected = pickle.load(open(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN+"\\" + str(elem), "rb")), []
        
        for bounding_box in r["rois"]:
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(bounding_box)
            if (center_x - width/2 > abstand_vom_rand and center_x + width/2 < bild_groeße_x - abstand_vom_rand and
             center_y - height/2 > abstand_vom_rand and center_y + height/2 < bild_groeße_y - abstand_vom_rand):
                r_selected.append(list(bounding_box * pixelsize / 1000))
        list_r.append(np.array(r_selected))

    #HIER FINDET DIE UMDREHUNG DES KOORDINATENSYSTEMS STATT.
    r_selected_umgedrehtes_koordinantensystem = []
    for bild in list_r:
        bounding_boxen_mit_umgedrehter_y_achse = []
        for bounding_box in bild:
            bounding_boxen_mit_umgedrehter_y_achse.append([bild_groeße_y * pixelsize / 1000 - bounding_box[0], bounding_box[1], bild_groeße_y * pixelsize / 1000 - bounding_box[2], bounding_box[3]])
        
        r_selected_umgedrehtes_koordinantensystem.append(bounding_boxen_mit_umgedrehter_y_achse)
    #HIER FINDET DIE UMDREHUNG DES KOORDINATENSYSTEMS STATT.
        
    return r_selected_umgedrehtes_koordinantensystem, list_elem, len(list_elem)-1

def morphologie_eines_tropfens(r):
    height = abs(r[2]-r[0])
    width = abs(r[3]-r[1])
    center_y = abs(r[0] + height/2)
    center_x = abs(r[1] + width/2)
    durchmesser = (height + width)/2

    return [height, width, center_y, center_x, durchmesser]

def berechnen_sauterdurchmesser_pro_bild(bild_index):
    alle_tropfen_betrachtetes_bild = list_r[bild_index]

    if len(alle_tropfen_betrachtetes_bild) != 0:
        sum_d_hoch_drei, sum_d_hoch_zwei = 0, 0

        for tropfen_index, val in enumerate(alle_tropfen_betrachtetes_bild):
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index])
            sum_d_hoch_drei += durchmesser**3
            sum_d_hoch_zwei += durchmesser**2            

        return sum_d_hoch_drei/sum_d_hoch_zwei
    
    if len(alle_tropfen_betrachtetes_bild) == 0:
        return 0

def stationaerer_sauterdurchmesser():
    sum_all_sauterdurchmesser = 0
    list_avg_sauterdurchmesser = []
    
    for bild_index in range(start, ende):
        
        aktueller_sauterdurchmesser = berechnen_sauterdurchmesser_pro_bild(bild_index)
        sum_all_sauterdurchmesser += aktueller_sauterdurchmesser 
        avg_sauterdurchmesser = sum_all_sauterdurchmesser / (bild_index + 1)

        list_avg_sauterdurchmesser.append(avg_sauterdurchmesser)

    return list_avg_sauterdurchmesser
    
for VERWENDETE_DATEN in list_verwendete_daten:
    list_r, list_elem, ende = load_data()
    liste_sauterdurchmesser = stationaerer_sauterdurchmesser()
    #print(VERWENDETE_DATEN, liste_sauterdurchmesser[-1])

    pickle.dump(liste_sauterdurchmesser[-1], open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\sauterdurchmesser", "wb"))
