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

ROOT_DIR = "C:\\Users\\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"

list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90", "20221124_Oben_1,2_1_100",
                    "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90", "20221124_Unten_1,2_1_100",

                    "20221125_Oben_3_1_40", "20221125_Oben_3_1_80", "20221125_Oben_3_1_90_Exposure_Time_250", "20221125_Oben_3_1_90_Exposure_Time_500", 
                    "20221125_Oben_3_1_100_Exposure_Time_250", "20221125_Oben_3_1_100_Exposure_Time_500",
                    "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", "20221125_Unten_3_1_100",

                    "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90", "20221129_Oben_1_2_100",
                    "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90", "20221129_Unten_1_2_100",
                             
                    "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90", "20221130_Oben_1,2_1_100",
                    "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90", "20221130_Unten_1,2_1_100"]

list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90",
                    "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90",

                    "20221125_Oben_3_1_40", "20221125_Oben_3_1_80", "20221125_Oben_3_1_90_Exposure_Time_250", "20221125_Oben_3_1_90_Exposure_Time_500", 
                    "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", 

                    "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90",
                    "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90",
                             
                    "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90",
                    "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90"]

list_verwendete_daten = ["test"]

list_konti_dispers_hold_up = {"20221124_Oben_1,2_1_40": [0.1876 / 60000, 0.1574 / 60000, 7.5 / 100] , "20221124_Oben_1,2_1_80": [0.371 / 60000, 0.31 / 60000, 12.6 / 100],
                      "20221124_Oben_1,2_1_90": [0.414 / 60000, 0.339 / 60000, 16.9 / 100], "20221124_Unten_1,2_1_40": [0.1876 / 60000, 0.1574 / 60000, 7.5 / 100],
                      "20221124_Unten_1,2_1_80": [0.371 / 60000, 0.31 / 60000, 12.6 / 100], "20221124_Unten_1,2_1_90": [0.414 / 60000, 0.339 / 60000, 16.9 / 100],

                      "20221125_Oben_3_1_40": [0.21/60000, 0.075/60000, 8.75 / 100], "20221125_Oben_3_1_80": [0.444/60000, 0.145/60000, 14.3 / 100],
                      "20221125_Oben_3_1_90_Exposure_Time_250": [0.472/60000, 0.157/60000, 17.5 / 100], "20221125_Oben_3_1_90_Exposure_Time_500": [0.472/60000, 0.157/60000, 17.5 / 100],
                      "20221125_Unten_3_1_40": [0.21/60000, 0.075/60000, 8.75 / 100], "20221125_Unten_3_1_80": [0.444/60000, 0.145/60000, 14.3 / 100],
                      "20221125_Unten_3_1_90": [0.472/60000, 0.157/60000, 17.5 / 100],

                      "20221129_Oben_1_2_40": [0.09/60000, 0.185/60000, 4.5 / 100], "20221129_Oben_1_2_80": [0.18/60000, 0.359/60000, 7.5 / 100],
                      "20221129_Oben_1_2_90": [0.203/60000, 0.405/60000, 8.75 / 100], "20221129_Unten_1_2_40": [0.09/60000, 0.185/60000, 4.5 / 100],
                      "20221129_Unten_1_2_80": [0.18/60000, 0.359/60000, 7.5 / 100], "20221129_Unten_1_2_90": [0.203/60000, 0.405/60000, 8.75 / 100],
                                 
                      "20221130_Oben_1,2_1_40": [0.188/60000, 0.154/60000, 7.5 / 100], "20221130_Oben_1,2_1_80": [0.37/60000, 0.305/60000, 12.6 / 100],
                      "20221130_Oben_1,2_1_90": [0.407/60000, 0.335/60000, 16.9 / 100], "20221130_Unten_1,2_1_40": [0.188/60000, 0.154/60000, 7.5 / 100],
                      "20221130_Unten_1,2_1_80": [0.37/60000, 0.305/60000, 12.6 / 100], "20221130_Unten_1,2_1_90": [0.407/60000, 0.335/60000, 16.9 / 100]}

start, ende = 0, 10
pixelsize = 15.12162646 #[µm/px]
abstand_vom_rand = 10

step_size, untere_grenze, obere_grenze = 0.2, 0, 3
step_size_gesch, untere_grenze_gesch, obere_grenze_gesch = 10, -160, 100
untere_grenze_parametrisierung = 0.2        #diese grenze muss größer gleich als untere_grenze sein 
obere_grenze_parametrisierung = 2.4         #diese grenze muss kleiner gleich als obere_grenze sein

delta_seite = 0.9
delta_oben = 0.9
delta_unten = 0.9
maximale_abweichung_des_durchmessers = 0.3
maximale_abweichung_center = 0.7

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

def get_v_infinity(diameter):       
    Ar = (rho_c * abs(rho_c - rho_d) * g * (diameter ** 3)) / (eta_c ** 2)                              #richtig
    c_w_unendlich = 432 / Ar + 20 / (Ar ** (1/3)) + (0.51 * (Ar ** (1/3))) / (140 + Ar ** (1/3))        #richtig
    Re_unendlich_blase = Ar / (12 * ((0.065 * Ar + 1) ** (1/6)))                                        #richtig
    Re_unendlich_kugel = np.sqrt(4/3 * Ar / c_w_unendlich)                                              #richtig
    f2 = 1 - (1 / (1 + (diameter / d_um) ** alpha_um))                                                  #richtig
    K_strich_Hr = (3 * (eta_c + eta_d / f2)) / (2 * eta_c + 3 * eta_d / f2)                             #richtig
    f1_strich = 2 * (K_strich_Hr - 1)                                                                   #richtig
    Re_unendlich_rund = (1 - f1_strich) * Re_unendlich_kugel + f1_strich * Re_unendlich_blase           #richtig
    
    v_gr = np.sqrt(abs(rho_c - rho_d) * g * diameter / (2 * rho_c))                                     #richtig
    v_os = np.sqrt(2 * a15 * sigma / (rho_c * diameter))                                                #hier war roh_d und roh_c vertauscht?? so richtig?
    v_os_gr = (v_os ** alpha_gr + v_gr ** alpha_gr) ** (1 / alpha_gr)                                   #richtig
    v_rund = Re_unendlich_rund * eta_c / (rho_c * diameter)                                             #richtig

    v_infty = (v_os_gr * v_rund) / (v_os_gr ** a16 + v_rund ** a16) ** (1 / a16)                        #richtig

    return v_infty

def get_k_v(diameter):
    pi_sigma = sigma * (rho_c ** 2 / (eta_c ** 4 * (rho_c - rho_d) * g)) ** (1 / 3)                                                     #richtig
    k_s = 1.406 * phi_st ** 0.145 * pi_sigma ** (-0.028) * np.exp(-0.129 * (diameter / dh) ** 1.134 * (1 - phi_st) ** (-2.161))         #richtig

    return k_s

def get_u_y(diameter, n):
    u_x = V_x / (A_col * (1 - hold_up))
    k_s = get_k_v(diameter)
    v_infty = get_v_infinity(diameter)
    v_y = k_s * v_infty * (1 - hold_up) ** n - u_x

    return v_y

def load_data():
    list_elem = []
    for elem in os.listdir(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN):
        list_elem.append(int(elem))
    
    list_elem.sort()

    size = Image.open(ROOT_DIR+"\\datasets\\input\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[0])+".jpg").size
    bild_groeße_x, bild_groeße_y = int(size[0]), int(size[0])

    list_r = []
    for elem in list_elem:
        r, r_selected = pickle.load(open(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN+"\\" + str(elem), "rb")), []
        
        for bounding_box in r["rois"]:
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(bounding_box)
            if (center_x - width/2 > abstand_vom_rand and center_x + width/2 < bild_groeße_x - abstand_vom_rand and
             center_y - height/2 > abstand_vom_rand and center_y + height/2 < bild_groeße_y - abstand_vom_rand):
                r_selected.append(list(bounding_box * pixelsize / 1000))
        list_r.append(np.array(r_selected))
        
    return list_r, list_elem

def morphologie_eines_tropfens(r):
    height = abs(r[2]-r[0])
    width = abs(r[3]-r[1])
    center_y = abs(r[0] + height/2)
    center_x = abs(r[1] + width/2)
    durchmesser = (height + width)/2

    return [height, width, center_y, center_x, durchmesser]

def show_image_draw_drop(bild_index, dic_id):
    input = ROOT_DIR+"\\datasets\\input\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[bild_index])+".jpg"

    im = Image.open(input).convert("RGBA")
    alle_tropfen_betrachtetes_bild = list_r[bild_index]
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font = "arial.ttf", size = 24)

    for i in range(len(dic_id)):
        for ii in range(len(dic_id[i])):
            if dic_id[i][ii][0] == bild_index:
                tropfen_index = dic_id[i][ii][1]
                height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index])

                text = str(i)
                font_width, font_height = font.getsize(text)
                draw.text((center_x * 1000 / pixelsize - font_width/2, center_y  * 1000 / pixelsize - font_height/2), text, fill = (255,0,0), font = font)
                draw.rectangle((alle_tropfen_betrachtetes_bild[tropfen_index][1] * 1000 / pixelsize,
                                alle_tropfen_betrachtetes_bild[tropfen_index][2] * 1000 / pixelsize,
                                alle_tropfen_betrachtetes_bild[tropfen_index][3] * 1000 / pixelsize,
                                alle_tropfen_betrachtetes_bild[tropfen_index][0] * 1000 / pixelsize), outline = (255,0,0))
    
    im.save(ROOT_DIR+"\\datasets\\output_images\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[bild_index]) + ".png")

def make_video(fps_make_video, image_folder, output_dir, save_dir):
    os.mkdir(output_dir)

    image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png")]
    image_files.sort()

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps_make_video)
    clip.write_videofile(save_dir)

def make_video_normalverteilung(fps_make_video, image_folder, output_dir):
    image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png") and img[-7:-4] in np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2).astype("str")]
    image_files.sort()

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps_make_video)
    clip.write_videofile(output_dir)

def verhaeltis_erkannte_bilder():
    list_all_erkannt = []
    list_all_tropfen = []
    verhaeltnis = []                #hier die funktion umschreiben. schneller laufen lassen
    for bild_index in range(start, ende):
        sum_tropfen_erkannt = 0
        for i in range(len(dic_id)):
            for ii in range(len(dic_id[i])):
                if dic_id[i][ii][0] == bild_index:
                    sum_tropfen_erkannt += 1
        
        list_all_tropfen.append(len(list_r[bild_index]))
        list_all_erkannt.append(sum_tropfen_erkannt)
        if sum_tropfen_erkannt != 0:
            verhaeltnis.append(sum_tropfen_erkannt/len(list_r[bild_index]))
        else:
            verhaeltnis.append(0)

    plot_funktion(zeit[:-1], [verhaeltnis], "anzahl konsekuitiv verfolgter tropfen/anzahl druch mrcnn detektierter tropfen", "zeit [s]", ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\verhaeltis_" + VERWENDETE_DATEN, 0, 1, VERWENDETE_DATEN)    

def Bedingung_vorgaenger_nachfolger(morph_aktuelles_bild, morph_naechstest_bild, alle_tropfen_betrachtetes_bild, alle_tropfen_naechstest_bild, tropfen_index_betrachtetes_bild, tropfen_index_naechstest_bild):

    height_current, width_current, center_y_current, center_x_current, durchmesser_current = morph_aktuelles_bild
    height_next, width_next, center_y_next, center_x_next, durchmesser_next = morph_naechstest_bild

    if durchmesser_current < 0.6:
        vergleich_druchmesser = 0.6
    else:
        vergleich_druchmesser = durchmesser_current
        
    return (abs((durchmesser_next-durchmesser_current)/durchmesser_current) < maximale_abweichung_des_durchmessers and

    alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild][0] - vergleich_druchmesser * delta_unten <
    alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild][0] and 
    alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild][1] - vergleich_druchmesser * delta_seite <
    alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild][1] and 
    alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild][2] + vergleich_druchmesser * delta_oben >
    alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild][2] and  
    alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild][3] + vergleich_druchmesser * delta_seite >
    alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild][3] and
                    
    center_y_current + vergleich_druchmesser * maximale_abweichung_center > center_y_next and
    center_y_current - vergleich_druchmesser * maximale_abweichung_center < center_y_next and
    center_x_current + vergleich_druchmesser * maximale_abweichung_center >  center_x_next and
    center_x_current - vergleich_druchmesser * maximale_abweichung_center < center_x_next)

def konsekutive_bilderverfolgung():
    dic_id, id_counter, dic_droplet, vorgaenger_nachfolger_beziehung = {}, 0, [], []
    os.mkdir(ROOT_DIR + "\\datasets\\output_images\\" + VERWENDETE_DATEN)
    for bild_index in range(start, ende):
        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        alle_tropfen_naechstest_bild = list_r[bild_index + 1]
        
        for tropfen_index_betrachtetes_bild in range(len(alle_tropfen_betrachtetes_bild)):
            dic_droplet.append([[bild_index, tropfen_index_betrachtetes_bild]])

        for tropfen_index_betrachtetes_bild in range(len(alle_tropfen_betrachtetes_bild)):
            morph_aktuelles_bild = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild])
            
            for tropfen_index_naechstest_bild in range(len(alle_tropfen_naechstest_bild)):
                morph_naechstest_bild = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild])
                      
                if Bedingung_vorgaenger_nachfolger(morph_aktuelles_bild, morph_naechstest_bild, alle_tropfen_betrachtetes_bild, alle_tropfen_naechstest_bild, tropfen_index_betrachtetes_bild, tropfen_index_naechstest_bild) == True:
                    for i, value in enumerate(dic_droplet):
                        if value[0][0] == bild_index and value[0][1] == tropfen_index_betrachtetes_bild:
                            dic_droplet[i].append([bild_index + 1, tropfen_index_naechstest_bild])
            
        List_second = [] 
        dic_droplet_neu = []
        for i, elem in enumerate(dic_droplet):               
            if len(elem) > 2:
                List_second.append(elem)
            else:
                dic_droplet_neu.append(elem)
                
        list_tasaelicher_nachfolger = []
        for i, elem in enumerate(List_second):
            vorgaenger = elem[0]
            list_abweichung = []
            morph_aktuelles_bild = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[vorgaenger[1]])
                      
            for ii in range(1,len(elem)):
                potentieller_nachfolger = elem[ii]
                morph_naechstest_bild = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[potentieller_nachfolger[1]])
                abweichung = ((morph_naechstest_bild[3] - morph_aktuelles_bild[3])**2 + (morph_naechstest_bild[2] - morph_aktuelles_bild[2])**2) ** 0,5
                list_abweichung.append(abweichung)
            index_min = np.argmin(list_abweichung)
            list_tasaelicher_nachfolger.append([vorgaenger, elem[index_min + 1]])
            
        dic_droplet = dic_droplet_neu + list_tasaelicher_nachfolger

        List = []
        for i, elem in enumerate(dic_droplet):
            if len(elem) == 2 and elem[1] not in List:
                List.append(elem[1])                
        
        list_tatsaechlicher_vorgaenger = []
        for i, elem in enumerate(List):
            List_potentieller_vorgaenger = []
            nachfolger = elem
            for ii, elemelem in enumerate(dic_droplet):
                if elem == elemelem[-1]:
                    List_potentieller_vorgaenger.append(elemelem[0])
                    
            abweichung_center_potentieller_vorgaenger = []
            for elem in List_potentieller_vorgaenger:
                morph_aktuelles_bild = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild])
                morph_naechstest_bild =  morphologie_eines_tropfens(alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild])

                abweichung = ((morph_naechstest_bild[3] - morph_aktuelles_bild[3])**2 + (morph_naechstest_bild[2] - morph_aktuelles_bild[2])**2) ** 0,5
                abweichung_center_potentieller_vorgaenger.append(abweichung)
                
            index_min = np.argmin(abweichung_center_potentieller_vorgaenger)
            vorgaenger = List_potentieller_vorgaenger[index_min]
            list_tatsaechlicher_vorgaenger.append([vorgaenger, nachfolger])

        dic_droplet = list_tatsaechlicher_vorgaenger
        
        if dic_id == {}:
            for i, val in enumerate(dic_droplet):
                dic_id[id_counter] = val
                id_counter += 1
        else:
            for i, val in enumerate(dic_droplet):
                found = False 
                for elem in dic_id: 
                    if dic_id[elem][-1] == val[0]:
                        dic_id[elem].append(val[1])
                        found = True
                if found == False:
                    dic_id[id_counter] = val
                    id_counter += 1        
        
        dic_droplet = []
        show_image_draw_drop(bild_index, dic_id)

    pickle.dump(dic_id, open(ROOT_DIR+"\\daten\\dic_id_"+VERWENDETE_DATEN, "wb"))
   
    return dic_id

def geschwindigkeit_bild_index_bis_bild_index_plus_eins():
    list_v_x, list_v_y = [], []
    vorgaenger_nachfolger_beziehung = pickle.load(open(ROOT_DIR+"\\daten\\vorgaenger_nachfolger_beziehung_"+VERWENDETE_DATEN, "rb"))

    for bild_index in range(start, ende):
        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        alle_tropfen_naechstest_bild = list_r[bild_index + 1]
        
        sum_v_x, sum_v_y = 0, 0
        for pair in vorgaenger_nachfolger_beziehung[bild_index]:
            
            height_current, width_current, center_y_current, center_x_current, durchmesser_current = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[pair[0][1]])
            height_next, width_next, center_y_next, center_x_next, durchmesser_next = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[pair[1][1]])

            v_x = (center_x_next - center_x_current)/(zeit[bild_index + 1] - zeit[bild_index])
            v_y = (center_y_next - center_y_current)/(zeit[bild_index + 1] - zeit[bild_index])          

            sum_v_x += v_x
            sum_v_y += v_y

        if len(vorgaenger_nachfolger_beziehung[bild_index]) != 0:
            avg_v_x = sum_v_x/len(vorgaenger_nachfolger_beziehung[bild_index])          #ist dies zeile Falsch. einmal überprüfen. gibt es immer nur zweier paare. und zb keine einer paare. stimmt die länge oder muss durch len +1 geteilt werden?
            avg_v_y = sum_v_y/len(vorgaenger_nachfolger_beziehung[bild_index])
        else:
            avg_v_x = 0
            avg_v_y = 0
            
        list_v_x.append(avg_v_x)
        list_v_y.append(avg_v_y)

    return list_v_x, list_v_y

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

    plot_funktion(zeit[:-1], [list_avg_sauterdurchmesser], "sauterdurchmesser [mm]", "zeit [s]", ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\sauterdurchmesser_"+VERWENDETE_DATEN, None, None, VERWENDETE_DATEN)

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

def nach_groeße_sortieren():
    dic_id_groeßenverteilung = {}

    for groeße in np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2):
        dic_id_groeßenverteilung[groeße] = []

    dic_id_groeßenverteilung[np.inf] = []

    for größe in np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2):
        for id in dic_id:
            avg_durchmesser = avg_durchmesser_id(id)
                
            if avg_durchmesser < größe and avg_durchmesser > größe - step_size:
                dic_id_groeßenverteilung[größe].append(id)

    for id in dic_id:
        avg_durchmesser = avg_durchmesser_id(id)
        if avg_durchmesser > obere_grenze:
            dic_id_groeßenverteilung[np.inf].append(id)

    pickle.dump(dic_id_groeßenverteilung, open(ROOT_DIR+"\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "wb"))
    return dic_id_groeßenverteilung

def AnzahlderTropfen_Geschwindigkeit_RGB():
    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\AnzahlderTropfen_Geschwindigkeit_RGB")

    dic_id_geschwindigkeitsverteilung_ges = {}
    
    dic_id_geschwindigkeitsverteilung_alle_größen = {}
    dic_id_geschwindigkeitsverteilung_alle_größen[-np.inf] = []
    for geschwindigkeit in np.round(np.arange(untere_grenze_gesch + step_size_gesch, obere_grenze_gesch + step_size_gesch, step_size_gesch), 2):
        dic_id_geschwindigkeitsverteilung_alle_größen[geschwindigkeit] = []
    dic_id_geschwindigkeitsverteilung_alle_größen[np.inf] = []

    max_anzahl = 0
    for größe in dic_id_groeßenverteilung:
        
        dic_id_geschwindigkeitsverteilung = {}
        dic_id_geschwindigkeitsverteilung[-np.inf] = []
        for geschwindigkeit in np.round(np.arange(untere_grenze_gesch + step_size_gesch, obere_grenze_gesch + step_size_gesch, step_size_gesch), 2):
            dic_id_geschwindigkeitsverteilung[geschwindigkeit] = []
        dic_id_geschwindigkeitsverteilung[np.inf] = []

        for geschwindigkeit in np.round(np.arange(untere_grenze_gesch + step_size_gesch, obere_grenze_gesch + step_size_gesch, step_size_gesch), 2):
            for ID in dic_id_groeßenverteilung[größe]:
                avg_gesch_y, avg_gesch_x  = geschwindigkeit_einer_id(ID)
                        
                if avg_gesch_y < geschwindigkeit and avg_gesch_y > geschwindigkeit - step_size_gesch:
                    dic_id_geschwindigkeitsverteilung[geschwindigkeit].append(ID)
                    dic_id_geschwindigkeitsverteilung_alle_größen[geschwindigkeit].append(ID)
                    
        for ID in dic_id_groeßenverteilung[größe]:
            avg_gesch_y, avg_gesch_x  = geschwindigkeit_einer_id(ID)
            if avg_gesch_y > obere_grenze_gesch:
                dic_id_geschwindigkeitsverteilung[np.inf].append(ID)
                dic_id_geschwindigkeitsverteilung_alle_größen[np.inf].append(ID)
            if avg_gesch_y < untere_grenze_gesch:
                dic_id_geschwindigkeitsverteilung[-np.inf].append(ID)
                dic_id_geschwindigkeitsverteilung_alle_größen[-np.inf].append(ID)
    
        dic_id_geschwindigkeitsverteilung_ges[größe] = dic_id_geschwindigkeitsverteilung

        zwischenspeicher = len(dic_id_geschwindigkeitsverteilung[max(dic_id_geschwindigkeitsverteilung, key=lambda k: len(dic_id_geschwindigkeitsverteilung[k]))])    
        if max_anzahl < zwischenspeicher:
            max_anzahl = zwischenspeicher

    for größe in dic_id_geschwindigkeitsverteilung_ges:
        plot_geschwindigkeitsverteilung(dic_id_geschwindigkeitsverteilung_ges[größe], größe, max_anzahl)
    plot_geschwindigkeitsverteilung(dic_id_geschwindigkeitsverteilung_alle_größen, None, None)

def get_rgb():
    images = []                                                                              
    filenames = []
    filenames_sorted = os.listdir(ROOT_DIR + "\\datasets\\input\\" + VERWENDETE_DATEN)[start:ende]
    filenames_sorted.sort() 
    for filename in filenames_sorted: 
        image = cv2.imread(os.path.join(ROOT_DIR + "\\datasets\\input\\" + VERWENDETE_DATEN, filename))
        images.append(image)                  
        filenames.append(filename)

    rgb_list = []
    for image_num, image in enumerate(images):                                              
        rgb = image.mean(axis=0).mean(axis=0)[1] 
        rgb_list.append(rgb) 

    return rgb_list

def size_distribution():
    step_list = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)
    M_dis = np.zeros(shape = (len(step_list) + 1, 6))
    M_dis[:-1,0] = step_list
    M_dis[len(step_list),0] = np.inf
    
    for i, größe in enumerate(dic_id_groeßenverteilung):
        M_dis[i,1] = len(dic_id_groeßenverteilung[größe])

        sum_v_y, sum_v_x = 0, 0
        sum_v_y_only_positive, sum_v_y_only_negative = 0, 0
        counter_only_positive, counter_only_negative = 0, 0
        
        for ID in dic_id_groeßenverteilung[größe]:
            v_y, v_x = geschwindigkeit_einer_id(ID)
            
            sum_v_y +=  v_y 
            sum_v_x +=  v_x
            if v_y > 0:
                sum_v_y_only_positive += v_y
                counter_only_positive += 1
            else:
                sum_v_y_only_negative += v_y
                counter_only_negative += 1

        if len(dic_id_groeßenverteilung[größe]) != 0:
            M_dis[i,2] = sum_v_y/len(dic_id_groeßenverteilung[größe])
            M_dis[i,3] = sum_v_x/len(dic_id_groeßenverteilung[größe])
        else:
            M_dis[i,2] = 0
            M_dis[i,3] = 0
            
        if counter_only_positive != 0:
            M_dis[i,4] = sum_v_y_only_positive/counter_only_positive
        else:
            M_dis[i,4] = 0

        if counter_only_negative != 0:
            M_dis[i,5] = sum_v_y_only_negative/counter_only_negative
        else:
            M_dis[i,5] = 0

    pickle.dump(M_dis, open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "wb"))

def plot_funktion(x_achse, y_achse, y_label, x_label, SAVE_DIR, unteres_y_lim, oberes_y_lim, title):
    fig, axs = plt.subplots(figsize=(16,9))
    for elem in y_achse:
        axs.plot(x_achse, elem)
    axs.set_ylabel(y_label, fontsize = 12)
    axs.set_xlabel(x_label, fontsize=12)
    axs.set_title(title)
    axs.tick_params(direction="in")
    
    if unteres_y_lim != None and oberes_y_lim != None:
        axs.set_ylim([unteres_y_lim, oberes_y_lim])

    plt.savefig(SAVE_DIR, dpi = 300)
    plt.close()

def plot_bar(loc_label,label, pos_string,strings, x_achse, y_achse, y_modell, y_label, x_label, title, SAVE_DIR):
    fig, ax = plt.subplots(figsize=(16,9))
    if label != None:
        ax.bar(x_achse, y_achse, width = 0.8, label = label)
    else:
        ax.bar(x_achse, y_achse, width = 0.8)
        
    if y_modell != None:
        ax.plot(x_achse, y_modell[0], label = y_modell[2])        
        ax.plot(x_achse, y_modell[1], label = y_modell[3])
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.tick_params(direction="in")
    
    for i, string in enumerate(strings):
        ax.text(pos_string[0], pos_string[1] - i * 0.05, string, ha='left', va='top', transform=plt.gca().transAxes)

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels , loc = loc_label)  #'upper left'
    
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.16)
    fig.savefig(SAVE_DIR, dpi = 300)
    plt.close()
    
def plot_size_distribution(): 
    M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
    step_list = list(np.round(np.arange(untere_grenze_parametrisierung + step_size, obere_grenze_parametrisierung + step_size, step_size), 2))
    tropfengröße = list(M_dis[:,0])
    erste_index = tropfengröße.index(step_list[0])
    letzter_index = tropfengröße.index(step_list[-1])

    anzahl_tropfen, v_y, x = list(M_dis[:,1])[erste_index:letzter_index+1], list(M_dis[:,2])[erste_index:letzter_index+1], []
    for i, größe in enumerate(step_list):
        x.append(str(np.round(größe - step_size, 2)) + " bis " + str(größe))

    print(x)
    plot_bar(loc_label = None, label = None,pos_string = [0.8,0.9], strings = ["anzahl Tropfen: " + str(sum(anzahl_tropfen))], x_achse = x, y_achse = anzahl_tropfen, y_modell =  None, y_label = 'Anzahl der Tropfen', x_label = 'Tropfengröße [mm]', title = 'Anzahl der Tropfen  '+VERWENDETE_DATEN,SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\anzahl_tropfen_"+VERWENDETE_DATEN)
    
    y_modell_n, y_modell_n_ges, y_inf = [], [], []
    for diameter in step_list:
        y_modell_n.append(-get_u_y(diameter*10**-3, n)*10**3)
        y_modell_n_ges.append(-get_u_y(diameter*10**-3, n_ges)*10**3)
        y_inf.append(-get_v_infinity(diameter*10**-3)*10**3)
        
    plot_bar(loc_label = "lower left", label = "gemessenes v_y je Tropfengröße", pos_string = [0.05,0.2], strings = ["anzahl Tropfen: " + str(sum(anzahl_tropfen))],x_achse = x, y_achse = v_y, y_modell = [y_modell_n, y_modell_n_ges, "u_y mittels Modell (Parametriserung dieses Datensatzes) n = " + str(np.round(n,2)), "u_y mittels Modell (Gesamt-Parametrisierung) n = " + str(np.round(n_ges,2))], y_label = 'v_y [mm/s]', x_label = 'Tropfengröße [mm]',title = 'v_y  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y_parametrisiert_"+VERWENDETE_DATEN)
    plot_bar(loc_label = "lower left", label = "gemessenes v_y je Tropfengröße", pos_string = [0.05,0.2], strings = ["anzahl Tropfen: " + str(sum(anzahl_tropfen))],x_achse = x, y_achse = v_y, y_modell = None, y_label = 'v_y [mm/s]', x_label = 'Tropfengröße [mm]',title = 'v_y  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y"+VERWENDETE_DATEN)
    plot_bar(loc_label = "lower left", label = "gemessenes v_y je Tropfengröße", pos_string = [0.05,0.2], strings = ["anzahl Tropfen: " + str(sum(anzahl_tropfen))],x_achse = x, y_achse = v_y, y_modell = [y_modell_n, y_inf,  "u_y mittels Modell (Parametriserung dieses Datensatzes) n = " + str(np.round(n,2)), "v_inf"], y_label = 'v_y [mm/s]', x_label = 'Tropfengröße [mm]',title = 'v_y  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y_parametrisiert_v_inf_"+VERWENDETE_DATEN)

def plot_geschwindigkeitsverteilung(dict, größe, max_anzahl):
    x, keys = [], list(dict)
    for i in range(len(dict)):
        if keys[i] == np.inf:
            string = str(keys[i-1])+ " bis " + str(keys[i])
        elif keys[i] == -np.inf:
            string =  str(keys[i]) + " bis " + str(np.round(keys[i+1] - step_size_gesch, 2))
        else:
            string = str(np.round(keys[i] - step_size_gesch, 2)) + " bis " + str(keys[i])
        x.append(string)

    y, gesamt_anzahl_tropfen = [], 0
    for geschwindigkeit in dict:
        anzahl = len(dict[geschwindigkeit])
        y.append(anzahl)
        gesamt_anzahl_tropfen += anzahl

    if max_anzahl == None:
        max_anzahl = 0
        for geschwindigkeit in dict:
            anzahl = len(dict[geschwindigkeit])
            if anzahl > max_anzahl:
                max_anzahl = anzahl
          
    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(x, y, width = 0.8)
    ax.set_xlabel('Geschwindigkeit [mm/s]')
    ax.set_ylabel('Anzahl der Tropfen')
    ax.set_title('AnzahlderTropfen_Geschwindigkeit_RGB ' + VERWENDETE_DATEN + " " + str(größe))
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.16)
    ax.set_ylim([0, max_anzahl])
    ax.tick_params(direction="in")
    
    if größe == np.inf:
        plt.text(0.9, max_anzahl * 0.9, "größe: " + str(obere_grenze) + " bis " + str(größe) + " mm")
    elif größe == None:
        plt.text(0.9, max_anzahl * 0.9, "größe: alle größen")
    else:
        plt.text(0.9, max_anzahl * 0.9, "größe: " + str(np.round(größe - step_size, 2)) + " bis " + str(größe) + " mm")

    plt.text(0.9, max_anzahl * 0.85, "anzahl tropfen: " + str(gesamt_anzahl_tropfen))
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\AnzahlderTropfen_Geschwindigkeit_RGB" + "\\AnzahlderTropfen_Geschwindigkeit_RGB_" + str(größe)+".png", dpi = 300)
    plt.close()

#Achtung vermutlich falsche funktion
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

def fit_funktion(data, zeit):
    freq = 1.25
    
    guess_amplitude = 3*np.std(data)/(2**0.5)
    guess_phase = 0
    guess_offset = np.mean(data)

    p0 = [guess_amplitude, guess_phase, guess_offset]

    def my_sin(x, amplitude, phase, offset):
        return np.sin(x * 2 * np.pi * freq + phase) * amplitude + offset

    fit = curve_fit(my_sin, zeit, data, p0=p0)
    data_fit = my_sin(np.linspace(0, zeit, len(zeit)), *fit[0])
    
    return data_fit[-1], fit[0][1]

def plot_rgb_v_y_gekuertzt(list_v_y, zeit):
    bereich = 4
    for i, val in enumerate(zeit):
        if val > bereich:
            break
        
    fittet_v_y, phase_v_y = fit_funktion(list_v_y[:i], zeit[:i])
    fittet_rgb, phase_rgb = fit_funktion(rgb_list[:i], zeit[:i])
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    ax1.plot(zeit[:i], rgb_list[:i], 'b',linestyle='none', marker = "o", label = "RGB")
    ax1.plot(zeit[:i], fittet_rgb, 'b-', label = "fittet rgb")
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('RGB', color='b')
    ax1.tick_params('y', colors='b')
    ax1.tick_params(direction="in")
    
    ax2 = ax1.twinx()
    ax2.plot(zeit[:i], list_v_y[:i], 'k',linestyle='none', marker = "o", label = "v_y")
    ax2.plot(zeit[:i], fittet_v_y, 'k-', label = "fitted v_y")
    
    ax2.set_ylabel('v_y [mm/s]', color='k')
    ax2.set_ylim([-80, 40])
    ax2.tick_params('y', colors='k')
    ax2.tick_params(direction="in")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()

    plt.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\rgb_v_y_gekuertzt_"+VERWENDETE_DATEN, dpi = 300)
    plt.close()

def berechne_phasenverschiebung(datensatz_eins, datensatz_zwei):
    max_eins = max(datensatz_eins)
    min_eins = min(datensatz_eins)
    max_zwei = max(datensatz_zwei)
    min_zwei = min(datensatz_zwei)

    list_null_durchgang_eins = []
    for i in range(len(datensatz_eins)-1):
        if datensatz_eins[i] < (max_eins+min_eins)/2 and datensatz_eins[i + 1] > (max_eins+min_eins)/2:
            list_null_durchgang_eins.append(i)

    list_null_durchgang_zwei = []
    for i in range(len(datensatz_zwei)-1):
        if datensatz_zwei[i] < (max_zwei+min_zwei)/2 and datensatz_zwei[i + 1] > (max_zwei+min_zwei)/2:
            list_null_durchgang_zwei.append(i)
    phasenverschiebung = []
    for i in range( min(len(list_null_durchgang_eins), len(list_null_durchgang_zwei)) - 1):
        phase = zeit[list_null_durchgang_zwei[i]] - zeit[list_null_durchgang_eins[i]]
        if zeit[list_null_durchgang_zwei[i]] < zeit[list_null_durchgang_eins[i]]:
            phasenverschiebung.append(zeit[list_null_durchgang_zwei[i+1]] - zeit[list_null_durchgang_eins[i]])
        else:
            phasenverschiebung.append(zeit[list_null_durchgang_zwei[i]] - zeit[list_null_durchgang_eins[i]])
    return phasenverschiebung

def plot_phasenverschiebung():
    list_phasenverschiebung_in_degree = []
    for VERWENDETE_DATEN in list_verwendete_daten:  
        phasenverschiebung_in_sekunden = np.mean(pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\phasenverschiebung", "rb")))
        phasenverschiebung_in_rad = 2 * np.pi * phasenverschiebung_in_sekunden / 0.8
        phasenverschiebung_in_degree = phasenverschiebung_in_rad * 360 / (2 * np.pi)
        list_phasenverschiebung_in_degree.append(phasenverschiebung_in_degree)

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(list_verwendete_daten, list_phasenverschiebung_in_degree, width = 0.8)
    ax.set_ylabel("phasenverschiebung [°]")
    ax.set_title("Phasenverschiebung zwischen v_y und rgb in Grad")
    ax.tick_params(direction="in")        
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR+"\\datasets\\output_analyseddata\\phasenverschiebung.png", dpi = 300)
    
def plot_rgb_v_y():
    fittet_v_y, phase_v_y = fit_funktion(list_v_y, zeit[:-1])
    fittet_rgb, phase_rgb = fit_funktion(rgb_list, zeit[:-1])
    fig, ax1 = plt.subplots(figsize=(16,9))
 
    ax1.plot(zeit[:-1], rgb_list, 'b', linestyle='none', marker = "o", label = "RGB")
    ax1.plot(zeit[:-1], fittet_rgb, 'b-', label = "fittet rgb")
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('RGB', color='b')
    ax1.tick_params('y', colors='b')
    ax1.tick_params(direction="in")

    ax2 = ax1.twinx()
    ax2.plot(zeit[:-1], list_v_y, 'k',linestyle='none', marker = "o", label = "v_y")
    ax2.plot(zeit[:-1], fittet_v_y, 'k-', label = "fitted v_y")
    
    ax2.set_ylabel('v_y [mm/s]', color='k')
    ax2.set_ylim([-80, 40])
    ax2.tick_params('y', colors='k')
    ax2.tick_params(direction="in")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()

    plt.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\rgb_v_y_"+VERWENDETE_DATEN, dpi = 300)
    plt.close()

    phasenverschiebung = berechne_phasenverschiebung(fittet_rgb,fittet_v_y)
    pickle.dump(phasenverschiebung, open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\phasenverschiebung", "wb"))
    plot_rgb_v_y_gekuertzt(list_v_y, zeit)

def parametrisiere_modell():
    summe_oberezeile = 0
    summe_unterezeile = 0
    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen")

    for i, größe in enumerate(dic_id_groeßenverteilung):
        if größe != np.inf and größe in np.round(np.arange(untere_grenze_parametrisierung + step_size, obere_grenze_parametrisierung + step_size, step_size), 2):
            durchmesser = größe*10**-3
            v_infinity = get_v_infinity(durchmesser)
            k_v = get_k_v(durchmesser)
            v_messwert = -M_dis[i,2] * 10**-3

            summe_oberezeile += k_v * v_infinity * (u_x + v_messwert)
            summe_unterezeile += (k_v * v_infinity) ** 2
            
    n = np.log(summe_oberezeile/summe_unterezeile)/np.log(1-hold_up)
    pickle.dump(n, open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\v_y_positive_negative_all_anzahl_tropfen\\exponent", "wb"))
    return n

def get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN):
    hold_up_ohne_umrechnung = list_konti_dispers_hold_up[VERWENDETE_DATEN][2] #hold_up = epsilon = alpha_y
    hold_up_neu = umrechnung_holp_up(hold_up_ohne_umrechnung) 
    V_x = list_konti_dispers_hold_up[VERWENDETE_DATEN][1] #volume stream of continiuous phase [m^3/s]
    V_d = list_konti_dispers_hold_up[VERWENDETE_DATEN][0] #volume stream of dispersed phase [m^3/s]
    u_x = V_x / (A_col * (1 - hold_up_neu))

    return hold_up_neu, V_x, V_d, u_x

def gesamt_parametrisierung():
    summe_oberezeile, summe_unterezeile = 0, 0
    for VERWENDETE_DATEN in list_verwendete_daten:
        hold_up, V_x, V_d, u_x = get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN)
        dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "rb"))
        M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))

        for i, größe in enumerate(dic_id_groeßenverteilung):
            if größe != np.inf and größe in np.round(np.arange(untere_grenze_parametrisierung + step_size, obere_grenze_parametrisierung + step_size, step_size), 2):
                durchmesser = größe*10**-3
                v_infinity = get_v_infinity(durchmesser)
                k_v = get_k_v(durchmesser)
                v_messwert = -M_dis[i,2] * 10**-3

                summe_oberezeile += k_v * v_infinity * (u_x + v_messwert)
                summe_unterezeile += (k_v * v_infinity) ** 2

    n_ges = np.log(summe_oberezeile/summe_unterezeile)/np.log(1-hold_up)
    return n_ges

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

    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    plot_bar(loc_label = "upper left",label = "anzahl_tropfen_von_bild_zu_bild", pos_string = None, strings = [], x_achse = x, y_achse = list_größenverteilung_bild_zu_bild, y_modell = None, y_label = "Anzahl Tropfen", x_label = "Tropfengröße", title = 'Anzahl Tropfen von Bild zu Bild  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\anzahl_tropfen_ohne_ids_"+VERWENDETE_DATEN)

    return list_größenverteilung_bild_zu_bild
        
def länge_eine_folge():
    länge_dic, länge_list = {},[]
    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\langeids")

    dic_anzahl_laenge_ids_alle_groeßen = {}
    for laenge in range(2,100):
        dic_anzahl_laenge_ids_alle_groeßen[laenge] = 0
    for größe in dic_id_groeßenverteilung:
        länge_dic[größe], summe = [], 0
            
        dic_anzahl_laenge_ids = {}
        for laenge in range(2,100):
            dic_anzahl_laenge_ids[laenge] = 0
            
        for ID in dic_id_groeßenverteilung[größe]:
            laenge_id = dic_id[ID][-1][0] - dic_id[ID][0][0] + 1
            summe += laenge_id
            for laenge in range(2,100):
                if laenge == laenge_id:
                    dic_anzahl_laenge_ids[laenge] = dic_anzahl_laenge_ids[laenge] + 1
                    dic_anzahl_laenge_ids_alle_groeßen[laenge] = dic_anzahl_laenge_ids_alle_groeßen[laenge] + 1 
       
        list_laenge_ids = list(dic_anzahl_laenge_ids)
        anzahl_ids = list(dic_anzahl_laenge_ids.values())
        plot_bar(loc_label = "upper left", label = None, pos_string = None, strings = [], x_achse = list_laenge_ids, y_achse = anzahl_ids, y_modell = None, y_label = "Anzahl IDS", x_label = "Lange IDS", title = 'Anzahl an Bilder auf dem Tropfen verfolgt wird  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\langeids\\langeids_" + str(größe)+"_" + VERWENDETE_DATEN + ".png")

        #test_sum = 0
        #for i in range(len(list_laenge_ids)):
        #    test_sum += list_laenge_ids[i]*anzahl_ids[i]
        #print(test_sum/sum(anzahl_ids))
        #print(summe/len(dic_id_groeßenverteilung[größe]))

        if len(dic_id_groeßenverteilung[größe]) != 0:
            länge_dic[größe].append(summe/len(dic_id_groeßenverteilung[größe]))
            länge_list.append(summe/len(dic_id_groeßenverteilung[größe]))
        else:
            länge_dic[größe].append(0)
            länge_list.append(0)

    list_laenge_ids = list(dic_anzahl_laenge_ids_alle_groeßen)
    anzahl_ids = list(dic_anzahl_laenge_ids_alle_groeßen.values())
    plot_bar(loc_label = "upper left", label = None, pos_string = None, strings = [], x_achse = list_laenge_ids, y_achse = anzahl_ids, y_modell = None, y_label = "Anzahl IDS", x_label = "Lange IDS", title = 'Anzahl an Bilder auf dem Tropfen verfolgt wird  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\langeids\\langeids_alle_groeßen_" + VERWENDETE_DATEN + ".png") 
    
    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)

    plot_bar(loc_label = "upper left",label = "Durchschnittliche länge der IDS", pos_string = None, strings = [], x_achse = x, y_achse = länge_list[0:-1], y_modell = None, y_label = "Durchschnittliche länge der IDS", x_label = "Tropfengröße [mm]", title = 'Länge IDS  '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\länge_ids_"+VERWENDETE_DATEN)

def umrechnung_holp_up(hold_up):
    Vakt = 0.005            #m^3
    hres_eins = 0.15        #m
    hres_zwei = 0.55        #m
    Vres = (hres_eins + hres_zwei) * np.pi * 0.25 * 0.05**2
    hold_up_neu = (hold_up * Vakt) / (Vakt + Vres)
    
    return hold_up_neu

def Volumen_berechnung_kugel(durchmesser):
    return np.pi / 6 * durchmesser**3

def Volumenanteil_pro_groeßenklasse():
    v_ges = 0
    for ID in dic_id:
        avg_durchmesser = avg_durchmesser_id(ID)
        v_ges += Volumen_berechnung_kugel(avg_durchmesser)

    v_rel = []
    for größe in dic_id_groeßenverteilung:
        v_größe = 0
        for ID in dic_id_groeßenverteilung[größe]:
            avg_durchmesser = avg_durchmesser_id(ID)
            v_größe += Volumen_berechnung_kugel(avg_durchmesser)

        v_rel.append(v_größe/v_ges)

    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    plot_bar(loc_label = "upper left",label = "relativer_volumenanteil", pos_string = None, strings = [], x_achse = x, y_achse = v_rel[:-1], y_modell = None, y_label = "relativer_volumenanteil", x_label = "Tropfengröße [mm]", title = 'relativer volumenanteil '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\relativer_volumenanteil_"+VERWENDETE_DATEN)

def relative_verfolgungsrate():
    list_anzahl_verfolgung_in_tropfenklasse = []
    for größe in dic_id_groeßenverteilung:
        anzahl_verfolgung_in_tropfenklasse = 0
        for ID in dic_id_groeßenverteilung[größe]:
            anzahl_verfolgung_in_tropfenklasse += len(dic_id[ID])
        list_anzahl_verfolgung_in_tropfenklasse.append(anzahl_verfolgung_in_tropfenklasse)

    list_relative_verfolgungsrate = np.array(list_anzahl_verfolgung_in_tropfenklasse[:-1])/np.array(list_größenverteilung_bild_zu_bild)
    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    plot_bar(loc_label = "upper left",label = "relative verfolgungsrate", pos_string = None, strings = [], x_achse = x, y_achse = list_relative_verfolgungsrate, y_modell = None, y_label = "relative verfolgungsrate", x_label = "Tropfengröße [mm]", title = 'relative_verfolgungsrate '+ VERWENDETE_DATEN, SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\relative_verfolgungsrate_"+VERWENDETE_DATEN)

def laenge_datensatz():
    list_zeitdifferenz = []
    for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
        list_elem = []
        for elem in os.listdir(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN):
            list_elem.append(int(elem))
            
        list_elem.sort()

        timestamp_first = str(list_elem[0])
        timestamp_last = str(list_elem[-1])
                
        zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
        list_zeitdifferenz.append(zeitdifferenz)

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(list_verwendete_daten, list_zeitdifferenz, width = 0.8)
    #ax.set_xlabel(x_label)
    ax.set_ylabel("Zeit [ms]")
    ax.set_title("Länge der Datensätze")
    ax.tick_params(direction="in")

    #lines, labels = ax.get_legend_handles_labels()
    #ax.legend(lines, labels , loc = loc_label)  #'upper left'
        
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR+"\\datasets\\output_analyseddata\\länge_datensätze.png", dpi = 300)
    plt.close()

def exponent_parametrisierung():
    list_n = []
    for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
        n = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\v_y_positive_negative_all_anzahl_tropfen\\exponent", "rb"))
        print(n, VERWENDETE_DATEN)
        list_n.append(n)

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(list_verwendete_daten, list_n, width = 0.8)
    #ax.set_xlabel(x_label)
    ax.set_ylabel("exponent n")
    ax.set_title("exponent n")
    ax.tick_params(direction="in")

    #lines, labels = ax.get_legend_handles_labels()
    #ax.legend(lines, labels , loc = loc_label)  #'upper left'
        
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR+"\\datasets\\output_analyseddata\\exponent_n.png", dpi = 300)
    plt.close()

def berechnezeit_differenz():
    list_zeitdifferenz, sum_time, zeit = [], 0, [0]
    
    for i in range(1, len(list_elem)):
        timestamp_first = str(list_elem[i-1])
        timestamp_last = str(list_elem[i])
        
        zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
        list_zeitdifferenz.append(zeitdifferenz)
        zeit.append(np.round(zeit[i-1] + zeitdifferenz/1000, 4))

    plot_funktion(np.linspace(0, len(zeit[:-1]), len(zeit[:-1])), [list_zeitdifferenz], "Zeitdifferenz zum vorherigen Bild [ms]", "Bild", ROOT_DIR + "\\datasets\\timestamps\\timestamps_" + VERWENDETE_DATEN + ".png", None, None, VERWENDETE_DATEN)
    plot_funktion(np.linspace(0, len(zeit), len(zeit)), [zeit, np.linspace(0, 10, len(zeit)) ], "Zeit in [s]", "Bild", ROOT_DIR + "\\datasets\\timestamps\\zeit_" + VERWENDETE_DATEN + ".png", None, None, VERWENDETE_DATEN)
    
    return zeit

modus = 0
if __name__ == "__main__":
    if modus == 0:
        for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
            
            list_r, list_elem = load_data()
            zeit = berechnezeit_differenz()
            list_r, list_elem, zeit = list_r[start:ende+1], list_elem[start:ende+1], zeit[start:ende+1]
            dic_id = konsekutive_bilderverfolgung()

    if modus == 1:
        for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
            
            hold_up, V_x, V_d, u_x = get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN)
            
            list_r, list_elem = load_data()
            zeit = berechnezeit_differenz()
            list_r, list_elem, zeit = list_r[start:ende+1], list_elem[start:ende+1], zeit[start:ende+1]
    
            dic_id = pickle.load(open(ROOT_DIR+"\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
            dic_id_groeßenverteilung = nach_groeße_sortieren()

            os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN)
            Volumenanteil_pro_groeßenklasse()
            länge_eine_folge()
            size_distribution()
            rgb_list = get_rgb()
            list_v_x, list_v_y = geschwindigkeit_bild_index_bis_bild_index_plus_eins()
            plot_rgb_v_y()
            
            stationaerer_sauterdurchmesser()
            verhaeltis_erkannte_bilder()
            list_größenverteilung_bild_zu_bild = größenverteilung_bild_zu_bild()
            relative_verfolgungsrate()
            
        n_ges = gesamt_parametrisierung()
        
        for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
        
            hold_up, V_x, V_d, u_x = get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN)
            
            list_r, list_elem = load_data()
            zeit = berechnezeit_differenz()
            list_r, list_elem, zeit = list_r[start:ende+1], list_elem[start:ende+1], zeit[start:ende+1]

            dic_id = pickle.load(open(ROOT_DIR+"\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
            dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "rb"))
            M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
            
            n = parametrisiere_modell()
            plot_size_distribution()
            AnzahlderTropfen_Geschwindigkeit_RGB()

            make_video_normalverteilung(1,ROOT_DIR+"\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN+"\\AnzahlderTropfen_Geschwindigkeit_RGB",ROOT_DIR+"\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN+"\\AnzahlderTropfen_Geschwindigkeit_RGB\\video.mp4")
        laenge_datensatz()
        exponent_parametrisierung()
        plot_phasenverschiebung()
        
"""

python C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN\\skripte\\konsekuitive_bilderkennung.py

df_M_dis = pd.DataFrame(data = M_dis, columns=["Tropfen Größe", "Anzahl Tropfen", "y Geschwindigkeit gesamt", "x Geschwindigkeit", "y Geschwindigkeit nur positive ids", "y Geschwindigkeit nur negative ids"])
with pd.ExcelWriter(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\MDIS.xlsx") as writer:                                                           
    df_M_dis.to_excel(writer, sheet_name = 'df_M_dis')


df_M_dis = pd.read_excel(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\MDIS.xlsx")
        #print(list(df_M_dis)[1:])
        M_dis = np.zeros(shape = (len(list(df_M_dis["Tropfen Größe"])), 6))
        M_dis[:,0] = list(df_M_dis["Tropfen Größe"])
        M_dis[:,1] = list(df_M_dis["Anzahl Tropfen"])
        M_dis[:,2] = list(df_M_dis["y Geschwindigkeit gesamt"])
        M_dis[:,3] = list(df_M_dis["x Geschwindigkeit"])
        M_dis[:,4] = list(df_M_dis["y Geschwindigkeit nur positive ids"])
        M_dis[:,5] = list(df_M_dis["y Geschwindigkeit nur negative ids"])
        #print(M_dis)

"""


#warum so viele kleine tropfen unten sind? es könnte sein dass sie an siebböden entsehen. und dann langsam abstgeigen und somit sich na der kammera unteen akkumulieren. nicht ganz stationärer zustand
#eine große datenanalse ergib kein sinn. es sollte erst die hardware verbessert werden und reproduiziuerbare ergebnisse bekommen bei zweimal den selben versuch
#der grund warum kleine tropfen schlechter verfolgt werden liegt vermutlich darin ..... 
        
#sinus bei ux ?
#1,25 hz = 1,25 1/s


#wie bekomm ich es auf der gpu zum laufen?


"""
1. Tracking von Tropfen über mehrere Bilder hinweg              #-> siehe konsekutive_bilderverfolgung
    a. Speicherstruktur muss überlegt werden                    #-> siehe konsekutive_bilderverfolgung
2. Strukturierte Datenablage überlegen
    a. Info je Bild
        i. D32,                                                 #-> siehe berechnen_sauterdurchmesser_pro_bild
        ii. Geschwindigkeit,                                    #-> kinetik
        iii. Pulsinfo                                           #was ist damit gemeint? Phasenverschiebung
    b. Stationaerer d32                                          #was ist stationaerer d32?
3. Datenanalyse
    a. Geschwindigkeit je Tropfenklasse                         #-> size_distribution
    b. Geschwindigkeit waehrend einer Periode                    #-> kinetik ??
    c. Negative/positive Geschwindigkeiten                      #-> kinetik ??
    d. Einfluss des Lastfalls
    e. Einfluss des Phasenverhaeltnisses
    f. Unterschied oben/unten
4. Modellvergleich
    a. Parametrisierung je Datenpunkte
    b. Gesamt-Parametrisierung
"""

