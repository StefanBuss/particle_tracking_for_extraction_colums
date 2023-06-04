
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




def laenge_datensatz():

    list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90", "20221124_Oben_1,2_1_100",
                    "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90", "20221124_Unten_1,2_1_100",

                    "20221125_Oben_3_1_40", "20221125_Oben_3_1_80", "20221125_Oben_3_1_90_Exposure_Time_250", "20221125_Oben_3_1_90_Exposure_Time_500", 
                    "20221125_Oben_3_1_100_Exposure_Time_250", "20221125_Oben_3_1_100_Exposure_Time_500",
                    "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", "20221125_Unten_3_1_100",

                    "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90", "20221129_Oben_1_2_100",
                    "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90", "20221129_Unten_1_2_100",
                             
                    "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90", "20221130_Oben_1,2_1_100",
                    "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90", "20221130_Unten_1,2_1_100"]

    
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

    #lines, labels = ax.get_legend_handles_labels()
    #ax.legend(lines, labels , loc = loc_label)  #'upper left'
        
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.3)
    plt.show()
    fig.savefig("C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN\\datasets\\output_analyseddata\\länge_datensätze.png", dpi = 300)
    plt.close()


def exponent_parametrisierung():
    list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90",
                    "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90",

                    "20221125_Oben_3_1_40", "20221125_Oben_3_1_80", "20221125_Oben_3_1_90_Exposure_Time_250", "20221125_Oben_3_1_90_Exposure_Time_500", 
                    "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", 

                    "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90",
                    "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90",
                             
                    "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90",
                    "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90"]

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

    #lines, labels = ax.get_legend_handles_labels()
    #ax.legend(lines, labels , loc = loc_label)  #'upper left'
        
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom = 0.3)
    plt.show()
    fig.savefig("C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN\\datasets\\output_analyseddata\\exponent_n.png", dpi = 300)
    plt.close()


def länge_ids():
    list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90",
                    "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90",

                    "20221125_Oben_3_1_40", "20221125_Oben_3_1_80", "20221125_Oben_3_1_90_Exposure_Time_250", "20221125_Oben_3_1_90_Exposure_Time_500", 
                    "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", 

                    "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90",
                    "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90",
                             
                    "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90",
                    "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90"]
    
    for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
        länge_dic, länge_list = {},[]
        dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "rb"))
        dic_id = pickle.load(open(ROOT_DIR+"\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
    
        for größe in dic_id_groeßenverteilung:
            länge_dic[größe], summe = [], 0
            
            dic_anzahl_laenge_ids = {}
            for laenge in range(2,13):
                dic_anzahl_laenge_ids[laenge] = 0
            
            for ID in dic_id_groeßenverteilung[größe]:
                laenge_id = dic_id[ID][-1][0] - dic_id[ID][0][0] + 1
                summe += laenge_id

                for laenge in range(2,13):
                    if laenge == laenge_id:
                        dic_anzahl_laenge_ids[laenge] = dic_anzahl_laenge_ids[laenge] + 1
            print(dic_anzahl_laenge_ids)
            indexe = list(dic_anzahl_laenge_ids)
            lanege_ids = list(dic_anzahl_laenge_ids.values())
            print(indexe)
            print(lanege_ids)

            fig, ax = plt.subplots(figsize=(16,9))
            
            ax.bar(indexe, lanege_ids, width = 0.8)                
            #ax.set_xlabel(x_label)
            #ax.set_ylabel(y_label)
            #ax.set_title(title)
            
            #lines, labels = ax.get_legend_handles_labels()
            #ax.legend(lines, labels , loc = loc_label)  #'upper left'
            
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom = 0.16)
            plt.show()
            
            if len(dic_id_groeßenverteilung[größe]) != 0:
                länge_dic[größe].append(summe/len(dic_id_groeßenverteilung[größe]))
                länge_list.append(summe/len(dic_id_groeßenverteilung[größe]))
            else:
                länge_dic[größe].append(0)
                länge_list.append(0)


länge_ids()
