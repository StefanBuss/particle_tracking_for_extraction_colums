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


list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90",
                    "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90",

                    "20221125_Oben_3_1_40", "20221125_Oben_3_1_80", "20221125_Oben_3_1_90_ET_250", "20221125_Oben_3_1_90_ET_500", 
                    "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", 

                    "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90",
                    "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90",
                             
                    "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90",
                    "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90"]

list_konti_dispers_hold_up = {"20221124_Oben_1,2_1_40": [0.1876 / 60000, 0.1574 / 60000, 7.5 / 100] , "20221124_Oben_1,2_1_80": [0.371 / 60000, 0.31 / 60000, 12.6 / 100],
                      "20221124_Oben_1,2_1_90": [0.414 / 60000, 0.339 / 60000, 16.9 / 100], "20221124_Unten_1,2_1_40": [0.1876 / 60000, 0.1574 / 60000, 7.5 / 100],
                      "20221124_Unten_1,2_1_80": [0.371 / 60000, 0.31 / 60000, 12.6 / 100], "20221124_Unten_1,2_1_90": [0.414 / 60000, 0.339 / 60000, 16.9 / 100],

                      "20221125_Oben_3_1_40": [0.21/60000, 0.075/60000, 8.75 / 100], "20221125_Oben_3_1_80": [0.444/60000, 0.145/60000, 14.3 / 100],
                      "20221125_Oben_3_1_90_ET_250": [0.472/60000, 0.157/60000, 17.5 / 100], "20221125_Oben_3_1_90_ET_500": [0.472/60000, 0.157/60000, 17.5 / 100],
                      "20221125_Unten_3_1_40": [0.21/60000, 0.075/60000, 8.75 / 100], "20221125_Unten_3_1_80": [0.444/60000, 0.145/60000, 14.3 / 100],
                      "20221125_Unten_3_1_90": [0.472/60000, 0.157/60000, 17.5 / 100],

                      "20221129_Oben_1_2_40": [0.09/60000, 0.185/60000, 4.5 / 100], "20221129_Oben_1_2_80": [0.18/60000, 0.359/60000, 7.5 / 100],
                      "20221129_Oben_1_2_90": [0.203/60000, 0.405/60000, 8.75 / 100], "20221129_Unten_1_2_40": [0.09/60000, 0.185/60000, 4.5 / 100],
                      "20221129_Unten_1_2_80": [0.18/60000, 0.359/60000, 7.5 / 100], "20221129_Unten_1_2_90": [0.203/60000, 0.405/60000, 8.75 / 100],
                                 
                      "20221130_Oben_1,2_1_40": [0.188/60000, 0.154/60000, 7.5 / 100], "20221130_Oben_1,2_1_80": [0.37/60000, 0.305/60000, 12.6 / 100],
                      "20221130_Oben_1,2_1_90": [0.407/60000, 0.335/60000, 16.9 / 100], "20221130_Unten_1,2_1_40": [0.188/60000, 0.154/60000, 7.5 / 100],
                      "20221130_Unten_1,2_1_80": [0.37/60000, 0.305/60000, 12.6 / 100], "20221130_Unten_1,2_1_90": [0.407/60000, 0.335/60000, 16.9 / 100]}


remove_data = True
konstante_framerate = False

abstand_vom_rand = 10
#parameter die welche den Suchradius definieren
delta_seite = 0.9
delta_oben = 0.9
delta_unten = 0.9
maximale_abweichung_des_durchmessers = 0.3
maximale_abweichung_center = 0.7

#FIXE PARAMETER
phi_st = 0.22
dh = 0.002
alpha_gr = 8
d_um = 7.1 * 10 ** -3  # [m]
a15 = 1.52
a16 = 4.5
alpha_um = 8
alpha_def = 8
D = 0.05 # [m] # column diameter
A_col = 0.25 * np.pi * D ** 2  # [m^2]

rho_c = 999 # [kg/m^3]
rho_d = 864.4 # [kg/m^3]
eta_d = 0.563*10**(-3)# [Pa*s]
eta_c = 0.939*10**(-3)  # [Pa*s]
sigma = 34* 10 ** -3  # [N/m]
g = 9.81            # [m/s^2]

pixelsize = 15.12162646 #[µm/px]
start = 0


def berechnezeit_differenz():
    if konstante_framerate == False:
        list_zeitdifferenz, sum_time, zeit = [], 0, [0]
        
        for i in range(1, len(list_elem)):
            timestamp_first = str(list_elem[i-1])
            timestamp_last = str(list_elem[i])
            
            zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
            list_zeitdifferenz.append(zeitdifferenz)
            zeit.append(np.round(zeit[i-1] + zeitdifferenz/1000, 4))

        return zeit
    else:
        timestamp_first = str(list_elem[0])
        timestamp_last = str(list_elem[-1])
        länge_datensatz = len(list_elem)
        zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
        
        return np.linspace(0, zeitdifferenz/1000, len(list_elem))
    
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

def avg_durchmesser_id(ID):
    sum_durchmesser = 0

    for i in range(len(dic_id[ID])):
        bild_index = dic_id[ID][i][0]
        tropfen_index = dic_id[ID][i][1]

        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index])
        sum_durchmesser += durchmesser
        #print("durchmesser: ", durchmesser)        Krasser unterschied zwischen durchmesser und avg_durchmesser

    avg_durchmesser = sum_durchmesser/(i+1)
    #print("avg_durchmesser: ", avg_durchmesser)        Krasser unterschied zwischen durchmesser und avg_durchmesser

    return avg_durchmesser

def morphologie_eines_tropfens(r):
    height = abs(r[2]-r[0])
    width = abs(r[3]-r[1])
    center_y = abs(r[0] + height/2)
    center_x = abs(r[1] + width/2)
    durchmesser = (height + width)/2

    return [height, width, center_y, center_x, durchmesser]

def geschwindigkeit_einer_id(id):
    erster_bild_index = dic_id[id][0][0]
    erster_tropfen_index = dic_id[id][0][1]
    alle_tropfen_betrachtetes_bild = list_r[erster_bild_index]
    height_current, width_current, center_y_current, center_x_current, durchmesser_current = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[erster_tropfen_index])

    letzter_bild_index = dic_id[id][-1][0]
    letzter_tropfen_index = dic_id[id][-1][1]
    alle_tropfen_letztes_bild = list_r[letzter_bild_index]
    height_last, width_last, center_y_last, center_x_last, durchmesser_last = morphologie_eines_tropfens(alle_tropfen_letztes_bild[letzter_tropfen_index])

    v_y = (center_y_last - center_y_current)/(zeit[letzter_bild_index] - zeit[erster_bild_index])
    v_x = (center_x_last - center_x_current)/(zeit[letzter_bild_index] - zeit[erster_bild_index])

    return v_y, v_x

def get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN):
    V_x = list_konti_dispers_hold_up[VERWENDETE_DATEN][1] #volume stream of continiuous phase [m^3/s]
    V_d = list_konti_dispers_hold_up[VERWENDETE_DATEN][0] #volume stream of dispersed phase [m^3/s]

    return V_x, V_d

def berechne_hold_up():
    Volumen_ges = 0
    sum_Volumen_größe_durch_geschwindigkeit_größe = 0
    for größe in dic_id_groeßenverteilung:
        if len(dic_id_groeßenverteilung[größe]) != 0:
            summe_geschwindigkeit = 0
            counter = 0
            Volumen_größe = 0
            for ID in dic_id_groeßenverteilung[größe]:
                v_y, v_x = geschwindigkeit_einer_id(ID)             #hier nicht gewschwindigkeit einer id sonder von bild zu bild rechnen
                summe_geschwindigkeit += v_y
                avg_durchmesser = avg_durchmesser_id(ID)
                Volumen = (avg_durchmesser ** 3 * np.pi) / 6
                Volumen_ges += Volumen
                Volumen_größe += Volumen
            avg_geschwindigkeit = abs(summe_geschwindigkeit/len(dic_id_groeßenverteilung[größe]))
            sum_Volumen_größe_durch_geschwindigkeit_größe += Volumen_größe/avg_geschwindigkeit

    return V_d * sum_Volumen_größe_durch_geschwindigkeit_größe / (Volumen_ges * A_col) * 10 ** 3

for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
    list_r, list_elem, ende = load_data()
    V_x, V_d = get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN)
    zeit = berechnezeit_differenz()
    dic_id = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
    dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "rb"))
    berechneter_hold_up = berechne_hold_up()
    u_x = V_x / (A_col * (1 - berechneter_hold_up))
    print(VERWENDETE_DATEN, berechneter_hold_up)
