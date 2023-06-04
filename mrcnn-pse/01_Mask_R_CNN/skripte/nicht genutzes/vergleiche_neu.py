from PIL import Image, ImageDraw, ImageFont
import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = "C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"

dic_Tag_phasenverhältnis = {"20221124": "1,2_1", "20221125": "3_1", "20221129": "1_2", "20221130": "1,2_1"}
list_lastfälle = ["40", "80", "90"]
list_positionen = ["Oben", "Unten"]
list_phasenverhältnis = ['1_2', '1,2_1', '3_1']

list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90", "20221124_Oben_1,2_1_100",
                "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90", "20221124_Unten_1,2_1_100",

                "20221125_Oben_3_1_40","20221125_Oben_3_1_80","20221125_Oben_3_1_90_ET_250","20221125_Oben_3_1_90_ET_500","20221125_Oben_3_1_100_ET_250","20221125_Oben_3_1_100_ET_500",
                "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", "20221125_Unten_3_1_100",

                "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90", "20221129_Oben_1_2_100",
                "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90", "20221129_Unten_1_2_100",
                             
                "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90", "20221130_Oben_1,2_1_100",
                "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90", "20221130_Unten_1,2_1_100"]

dic_Tag_phasenverhältnis = {}
list_lastfälle = []
list_positionen = []
list_phasenverhältnis = []
sort_list = []

for VERWENDETE_DATEN in list_verwendete_daten:
    if VERWENDETE_DATEN.split("_")[1] not in list_positionen:
        list_positionen.append(VERWENDETE_DATEN.split("_")[1])
    if VERWENDETE_DATEN.split("_")[4] not in list_lastfälle and VERWENDETE_DATEN.split("_")[4] != "100":
        list_lastfälle.append(VERWENDETE_DATEN.split("_")[4])        
    if VERWENDETE_DATEN.split("_")[2] + "_" + VERWENDETE_DATEN.split("_")[3] not in list_phasenverhältnis:
        list_phasenverhältnis.append(VERWENDETE_DATEN.split("_")[2] + "_" + VERWENDETE_DATEN.split("_")[3])
        sort_list.append(float(VERWENDETE_DATEN.split("_")[2].replace(",","."))/float(VERWENDETE_DATEN.split("_")[3].replace(",",".")))
    if VERWENDETE_DATEN.split("_")[0] not in dic_Tag_phasenverhältnis:
        dic_Tag_phasenverhältnis[VERWENDETE_DATEN.split("_")[0]] = VERWENDETE_DATEN.split("_")[2] + "_" + VERWENDETE_DATEN.split("_")[3]
kombinierte_liste = list(zip(sort_list, list_phasenverhältnis))
kombinierte_liste.sort()
list_phasenverhältnis = [element[1] for element in kombinierte_liste]



def vergleich_oben_unten():
    print("start vergleich oben unten")
    list_vergleich_oben_unten = []
    for lastfall in list_lastfälle:
        for Tag in dic_Tag_phasenverhältnis:
            if Tag == "20221125" and lastfall == "90":
                #list_vergleich_oben_unten.append([Tag + "_Oben_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall + "_ET_250", Tag + "_Unten_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall])
                list_vergleich_oben_unten.append([Tag + "_Oben_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall + "_ET_500", Tag + "_Unten_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall])
            else:
                list_vergleich_oben_unten.append([Tag + "_Oben_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall, Tag + "_Unten_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall])
    
def vergleich_lastfall():
    print("start vergleich lastfall")

    list_alle_lastfall_vergleiche = []
    for poistion in list_positionen:
        for Tag in dic_Tag_phasenverhältnis:
            list_vergleich = []
            for lastfall in list_lastfälle:
                if Tag == "20221125" and lastfall == "90" and poistion == "Oben":
                    list_vergleich.append(Tag + "_" + poistion + "_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall + "_ET_500")
                else:
                    list_vergleich.append(Tag + "_" + poistion + "_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall)

            list_alle_lastfall_vergleiche.append(list_vergleich)

    fig, ax = plt.subplots(figsize=(16,9)) 
    for elem in list_alle_lastfall_vergleiche:
        array = []
        for VERWENDETE_DATEN in elem:
            n = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\v_y_positive_negative_all_anzahl_tropfen\\exponent", "rb"))
            array.append(float(n))
        split_data = VERWENDETE_DATEN.split("_")
    
        if split_data[2] + "/" + split_data[3] == "1,2/1":
            color_plot = (0,0,1)
        if split_data[2] + "/" + split_data[3] == "3/1":
            color_plot = (0,1,0)
        if split_data[2] + "/" + split_data[3] == "1/2":
            color_plot = (1,0,0)

        if split_data[1] == "Unten":
            ax.plot([40,80,90], array, label = "kameraposition: " + split_data[1] + ", Phasenverhältnis: " + split_data[2] + "/" + split_data[3], color = color_plot)[0].set_dashes([6, 3])
        else:
            ax.plot([40,80,90], array, label = "kameraposition: " + split_data[1] + ", Phasenverhältnis: " + split_data[2] + "/" + split_data[3], color = color_plot)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial')  

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial')
        
    ax.set_xlabel("Flutpunktbelastung [%]", fontsize = 24, fontname = "Arial")
    ax.set_ylabel("Schwarmexponent n", fontsize = 24, fontname = "Arial")
    ax.legend(loc = "upper left")
    ax.tick_params(direction="in")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\schwarmexponent_lastfall", dpi = 300, bbox_inches='tight')



    fig, ax = plt.subplots(figsize=(16,9)) 
    for elem in list_alle_lastfall_vergleiche:
        array = []
        for VERWENDETE_DATEN in elem:
            phasenverschiebung_in_sekunden = np.mean(pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\phasenverschiebung", "rb")))
            phasenverschiebung_in_rad = 2 * np.pi * phasenverschiebung_in_sekunden / 0.8
            phasenverschiebung_in_degree = phasenverschiebung_in_rad * 360 / (2 * np.pi)
            array.append(phasenverschiebung_in_degree)
        split_data = VERWENDETE_DATEN.split("_")

        if split_data[2] + "/" + split_data[3] == "1,2/1":
            color_plot = (0,0,1)
        if split_data[2] + "/" + split_data[3] == "3/1":
            color_plot = (0,1,0)
        if split_data[2] + "/" + split_data[3] == "1/2":
            color_plot = (1,0,0)

        if split_data[1] == "Unten":
            ax.plot([40,80,90], array, label = "kameraposition: " + split_data[1] + ", Phasenverhältnis: " + split_data[2] + "/" + split_data[3], color = color_plot)[0].set_dashes([6, 3])
        else:
            ax.plot([40,80,90], array, label = "kameraposition: " + split_data[1] + ", Phasenverhältnis: " + split_data[2] + "/" + split_data[3], color = color_plot)
            
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial')  

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial') 
    
    ax.set_xlabel("Flutpunktbelastung [%]", fontsize = 24, fontname = "Arial")
    ax.set_ylabel("Phasenverschiebung [°]", fontsize = 24, fontname = "Arial")
    ax.legend(loc = "lower right")
    ax.tick_params(direction="in")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\Phasenverschiebung_lastfall", dpi = 300, bbox_inches='tight')




    fig, ax = plt.subplots(figsize=(16,9)) 
    for elem in list_alle_lastfall_vergleiche:
        array = []
        for VERWENDETE_DATEN in elem:
            array.append(float(pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\sauterdurchmesser", "rb"))))
        split_data = VERWENDETE_DATEN.split("_")

        if split_data[2] + "/" + split_data[3] == "1,2/1":
            color_plot = (0,0,1)
        if split_data[2] + "/" + split_data[3] == "3/1":
            color_plot = (0,1,0)
        if split_data[2] + "/" + split_data[3] == "1/2":
            color_plot = (1,0,0)

        if split_data[1] == "Unten":
            ax.plot([40,80,90], array, label = "kameraposition: " + split_data[1] + ", Phasenverhältnis: " + split_data[2] + "/" + split_data[3], color = color_plot)[0].set_dashes([6, 3])
        else:
            ax.plot([40,80,90], array, label = "kameraposition: " + split_data[1] + ", Phasenverhältnis: " + split_data[2] + "/" + split_data[3], color = color_plot)
            
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial')  

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial') 
    
    ax.set_xlabel("Flutpunktbelastung [%]", fontsize = 24, fontname = "Arial")
    ax.set_ylabel("Sauterdurchmesser [mm]", fontsize = 24, fontname = "Arial")
    ax.legend(loc = "upper right")
    ax.tick_params(direction="in")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\Sauterdurchmesser_lastfall", dpi = 300, bbox_inches='tight')
         
def vergleich_phasenverhältnis():
    dic_phasenverhältnis_Tag = {}

    print("start vergleich phasenverhältnis")
    """
    for Tag in dic_Tag_phasenverhältnis: 
        if dic_Tag_phasenverhältnis[Tag] not in list_phasenverhältnis:
            list_phasenverhältnis.append(dic_Tag_phasenverhältnis[Tag])
    print(list_phasenverhältnis)
    """
    
    for lastfall in list_phasenverhältnis:
        dic_phasenverhältnis_Tag[lastfall] = []

    for Tag in dic_Tag_phasenverhältnis: 
        for phasenverhältnis in dic_phasenverhältnis_Tag:
            if dic_Tag_phasenverhältnis[Tag] == phasenverhältnis:
                dic_phasenverhältnis_Tag[phasenverhältnis].append(Tag)
                
    elements = []
    for phasenverhältnis in dic_phasenverhältnis_Tag:
        elements.append(dic_phasenverhältnis_Tag[phasenverhältnis])

    combinations = []

    for triplet in itertools.combinations(elements, len(dic_phasenverhältnis_Tag)):
        for combination in itertools.product(*triplet):
            combinations.append(list(combination))

    list_alle_phasenvergleiche = []
    for position in list_positionen:
        for lastfall in list_lastfälle:
            for combination in combinations:
                list_vergleich = []
                for Tag in combination:
                    if Tag == "20221125" and lastfall == "90" and position == "Oben":
                        list_vergleich.append(Tag + "_" + position + "_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall + "_ET_500")
                    else:
                        list_vergleich.append(Tag + "_" + position + "_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall)
                list_alle_phasenvergleiche.append(list_vergleich)
    
    fig, ax = plt.subplots(figsize=(16,9))
    for elem in list_alle_phasenvergleiche:
        x_array, array = [], []
        for VERWENDETE_DATEN in elem:
            split_array = VERWENDETE_DATEN.split("_")
            x_array.append(split_array[2]+"/"+split_array[3])
            n = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\v_y_positive_negative_all_anzahl_tropfen\\exponent", "rb"))
            array.append(float(n))
        x = range(len(x_array))

        if split_array[4] == "40":
            color_plot = (0,0,1)
        if split_array[4] == "80":
            color_plot = (0,1,0)
        if split_array[4] == "90":
            color_plot = (1,0,0)

        if split_array[1] == "Unten":
            ax.plot(x, array, label = "Kameraposition: "+split_array[1] + ", Flutpunktbelastung: "+split_array[4]+ " [%]", color = color_plot)[0].set_dashes([6, 10])
        else:
            ax.plot(x, array, label = "Kameraposition: "+split_array[1] + ", Flutpunktbelastung: "+split_array[4]+ " [%]", color = color_plot)

    ax.set_xticks(x)
    ax.set_xticklabels(x_array, fontsize = 18, fontname = "Arial")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(18)
        tick_label.set_fontname('Arial')
    
    ax.set_xlabel("Phasenverhätnis", fontsize = 24, fontname = "Arial")
    ax.set_ylabel("Schwarmexponent n", fontsize = 24, fontname = "Arial")
    ax.legend(loc = "upper left")
    ax.tick_params(direction="in")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\schwarmexponent_phasenverhätnis", dpi = 300, bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(16,9))
    for elem in list_alle_phasenvergleiche:
        x_array, array = [], []
        for VERWENDETE_DATEN in elem:
            split_array = VERWENDETE_DATEN.split("_")
            x_array.append(split_array[2]+"/"+split_array[3])
            
            phasenverschiebung_in_sekunden = np.mean(pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\phasenverschiebung", "rb")))
            phasenverschiebung_in_rad = 2 * np.pi * phasenverschiebung_in_sekunden / 0.8
            phasenverschiebung_in_degree = phasenverschiebung_in_rad * 360 / (2 * np.pi)
            array.append(phasenverschiebung_in_degree)
            
        x = range(len(x_array))

        if split_array[4] == "40":
            color_plot = (0,0,1)
        if split_array[4] == "80":
            color_plot = (0,1,0)
        if split_array[4] == "90":
            color_plot = (1,0,0)
        
        if split_array[1] == "Unten":
            ax.plot(x, array, label = "Kameraposition: "+split_array[1] + ", Flutpunktbelastung: "+split_array[4]+ " [%]", color = color_plot)[0].set_dashes([6, 3])
        else:
            ax.plot(x, array, label = "Kameraposition: "+split_array[1] + ", Flutpunktbelastung: "+split_array[4]+ " [%]", color = color_plot)
    ax.set_xticks(x)

    ax.set_xticklabels(x_array, fontsize = 18, fontname = "Arial")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(18)
        tick_label.set_fontname('Arial')

    ax.set_xlabel("Phasenverhätnis", fontsize = 18, fontname = "Arial")
    ax.set_ylabel("Phasenverschiebung [°]", fontsize = 18, fontname = "Arial")
    ax.tick_params(direction="in")
    ax.legend(loc = "lower left")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\Phasenverschiebung_phasenverhätnis", dpi = 300 , bbox_inches='tight')
    



    fig, ax = plt.subplots(figsize=(16,9))
    for elem in list_alle_phasenvergleiche:
        x_array, array = [], []
        for VERWENDETE_DATEN in elem:
            split_array = VERWENDETE_DATEN.split("_")
            x_array.append(split_array[2]+"/"+split_array[3])
            array.append(float(pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\sauterdurchmesser", "rb"))))
        x = range(len(x_array))

        if split_array[4] == "40":
            color_plot = (0,0,1)
        if split_array[4] == "80":
            color_plot = (0,1,0)
        if split_array[4] == "90":
            color_plot = (1,0,0)
        
        if split_array[1] == "Unten":
            ax.plot(x, array, label = "Kameraposition: "+split_array[1] + ", Flutpunktbelastung: "+split_array[4]+ " [%]", color = color_plot)[0].set_dashes([6, 3])
        else:
            ax.plot(x, array, label = "Kameraposition: "+split_array[1] + ", Flutpunktbelastung: "+split_array[4]+ " [%]", color = color_plot)
    ax.set_xticks(x)

    ax.set_xticklabels(x_array, fontsize = 24, fontname = "Arial")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial')

    ax.set_ylim([1.65, 2.25])
    ax.set_xlabel("Phasenverhätnis", fontsize = 24, fontname = "Arial")
    ax.set_ylabel("Sauterdurchmesser [mm]", fontsize = 24, fontname = "Arial")
    ax.tick_params(direction="in")
    ax.legend(loc = "upper right")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\Sauterdurchmesser_phasenverhätnis", dpi = 300 , bbox_inches='tight')



#vergleich_oben_unten()
vergleich_lastfall()
vergleich_phasenverhältnis()


#40 = Blaue
#80 = Grun
#90 = Rot
