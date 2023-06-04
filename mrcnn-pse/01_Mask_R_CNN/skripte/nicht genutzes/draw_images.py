import pickle
import os
from PIL import Image, ImageDraw
import time 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import scipy
from scipy import signal
from scipy.optimize import curve_fit
import sys

VERWENDETE_DATEN = "20221129_Oben_1_2_90"
ROOT_DIR = "C:\\Users\\StefanvenderBuss\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"

list_elem = []
for elem in os.listdir(ROOT_DIR+"\\datasets\\output\\"+VERWENDETE_DATEN):
    list_elem.append(int(elem))

list_elem.sort()

list_r = []
for elem in list_elem:
    r = pickle.load(open(ROOT_DIR+"\\datasets\\output\\"+VERWENDETE_DATEN+"\\"+str(elem), "rb"))
    list_r.append(r["rois"])


def draw(bild_index,tropfen_index):
    input = ROOT_DIR+"\\datasets\\input\\"+VERWENDETE_DATEN+"\\Image_" + str(list_elem[bild_index])+".jpg"

    im = Image.open(input).convert("RGBA")
    alle_tropfen_betrachtetes_bild = list_r[bild_index]
    draw = ImageDraw.Draw(im)
    
    draw.rectangle((alle_tropfen_betrachtetes_bild[tropfen_index][1],
            alle_tropfen_betrachtetes_bild[tropfen_index][2],
            alle_tropfen_betrachtetes_bild[tropfen_index][3],
            alle_tropfen_betrachtetes_bild[tropfen_index][0]), outline = (255,0,0))

    #draw.rectangle((0,0,30,20), outline = (255,0,0))
    #[x0, y0, x1, y1]
    
    im.show()


bild_index = 16
tropfen_index = 19

draw(bild_index,tropfen_index)







