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

start = 0
step_size, untere_grenze, obere_grenze = 0.2, 0, 3
abstand_vom_rand = 10
pixelsize = 15.12162646 #[µm/px]

list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90", "20221124_Oben_1,2_1_100",
                "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90", "20221124_Unten_1,2_1_100",

                "20221125_Oben_3_1_40","20221125_Oben_3_1_80","20221125_Oben_3_1_90_ET_250","20221125_Oben_3_1_90_ET_500","20221125_Oben_3_1_100_ET_250","20221125_Oben_3_1_100_ET_500",
                "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", "20221125_Unten_3_1_100",

                "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90", "20221129_Oben_1_2_100",
                "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90", "20221129_Unten_1_2_100",
                             
                "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90", "20221130_Oben_1,2_1_100",
                "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90", "20221130_Unten_1,2_1_100"]


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




def größenverteilung_bild_zu_bild():
    list_größenverteilung_bild_zu_bild = []

    step_liste = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)
    for größe in step_liste:
        list_größenverteilung_bild_zu_bild.append(0)
    
    for bild_index in range(start, ende):
        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        for tropfen in alle_tropfen_betrachtetes_bild:
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(tropfen)

            for index, größe in enumerate(step_liste):
                if durchmesser < größe and durchmesser > größe - step_size:
                    list_größenverteilung_bild_zu_bild[index] = list_größenverteilung_bild_zu_bild[index] + 1

    liste_relative_anzahl_tropfen = []
    for i in range(len(list_größenverteilung_bild_zu_bild)):
        liste_relative_anzahl_tropfen.append(list_größenverteilung_bild_zu_bild[i]/sum(list_größenverteilung_bild_zu_bild))

    return liste_relative_anzahl_tropfen


def größenverteilung_id():
    #dic_id_groeßenverteilung_20221124_Oben_1,2_1_80
    dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_groeßenverteilung_" + VERWENDETE_DATEN, "rb"))

    summe = 0
    for größe in dic_id_groeßenverteilung:
        if größe != np.inf:
            summe += len(dic_id_groeßenverteilung[größe])

    liste_relative_anzahl_id = []
    for größe in dic_id_groeßenverteilung:
        if größe != np.inf:
            liste_relative_anzahl_id.append(len(dic_id_groeßenverteilung[größe])/summe)

    return liste_relative_anzahl_id


def plot_modell_wiederholder(y1,y2):
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(np.arange(len(x)) - bar_width/2, y1, width=bar_width, label = "Relative Anzahl IDs je Größenklasse")
    ax.bar(np.arange(len(x)) + bar_width/2, y2, width=bar_width, label = "Relative Anzahl Tropfen je Größenklasse")

    ax.set_xlabel('Durchmesser [mm]', fontsize = 24, fontname = "Arial")
    ax.set_ylabel('Relative Anzahl', fontsize = 24, fontname = "Arial")
    ax.set_title(VERWENDETE_DATEN, fontsize = 24, fontname = "Arial")
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.legend()

    plt.xticks(rotation=45, ha='right', fontsize = 24, fontname = "Arial")
    plt.yticks(fontsize = 24, fontname = "Arial")
    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\vergleich_relative_anzahl_"+VERWENDETE_DATEN+".png", dpi = 300, bbox_inches='tight')

tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
for i in range(len(tropfengröße)):
    if tropfengröße[i] != np.inf:
        string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
    else:
        string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
    x.append(string)

for VERWENDETE_DATEN in list_verwendete_daten:
    #list_r, list_elem, ende = load_data()
    liste_relative_anzahl_id = größenverteilung_id()
    #liste_relative_anzahl_tropfen = größenverteilung_bild_zu_bild()
    plot_modell_wiederholder(liste_relative_anzahl_id,liste_relative_anzahl_tropfen)
