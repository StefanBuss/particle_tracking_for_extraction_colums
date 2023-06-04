from PIL import Image, ImageDraw, ImageFont
import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

ROOT_DIR = "C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"

font_size_plot = 30
legend_fontsize = 24
legend_fontsize_small = 14
step_size_gesch, untere_grenze_gesch, obere_grenze_gesch = 10, -100, 160
step_size, untere_grenze, obere_grenze = 0.2, 0, 3

list_verwendete_daten = ["20221124_Oben_1,2_1_40", "20221124_Oben_1,2_1_80", "20221124_Oben_1,2_1_90", "20221124_Oben_1,2_1_100",
                "20221124_Unten_1,2_1_40", "20221124_Unten_1,2_1_80", "20221124_Unten_1,2_1_90", "20221124_Unten_1,2_1_100",

                "20221125_Oben_3_1_40","20221125_Oben_3_1_80","20221125_Oben_3_1_90_ET_250","20221125_Oben_3_1_90_ET_500","20221125_Oben_3_1_100_ET_250","20221125_Oben_3_1_100_ET_500",
                "20221125_Unten_3_1_40", "20221125_Unten_3_1_80", "20221125_Unten_3_1_90", "20221125_Unten_3_1_100",

                "20221129_Oben_1_2_40", "20221129_Oben_1_2_80", "20221129_Oben_1_2_90", "20221129_Oben_1_2_100",
                "20221129_Unten_1_2_40", "20221129_Unten_1_2_80", "20221129_Unten_1_2_90", "20221129_Unten_1_2_100",
                             
                "20221130_Oben_1,2_1_40", "20221130_Oben_1,2_1_80", "20221130_Oben_1,2_1_90", "20221130_Oben_1,2_1_100",
                "20221130_Unten_1,2_1_40", "20221130_Unten_1,2_1_80", "20221130_Unten_1,2_1_90", "20221130_Unten_1,2_1_100"]

tropfengröße = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)

def erzeuge_vergleiche():
    dic_Tag_phasenverhältnis, list_lastfälle, list_positionen, list_phasenverhältnis, sort_list = {}, [], [], [], []

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

    list_vergleich_oben_unten = []
    for lastfall in list_lastfälle:
        for Tag in dic_Tag_phasenverhältnis:
            if Tag == "20221125" and lastfall == "90":
                #list_vergleich_oben_unten.append([Tag + "_Oben_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall + "_ET_250", Tag + "_Unten_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall])
                list_vergleich_oben_unten.append([Tag + "_Oben_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall + "_ET_500", Tag + "_Unten_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall])
            else:
                list_vergleich_oben_unten.append([Tag + "_Oben_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall, Tag + "_Unten_" + dic_Tag_phasenverhältnis[Tag] + "_" + lastfall])

    dic_phasenverhältnis_Tag = {}
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

    return list_alle_lastfall_vergleiche, list_vergleich_oben_unten, list_alle_phasenvergleiche

def vergleich_oben_unten():
    step_size, untere_grenze, obere_grenze = 0.2, 0.2, 2.4
    step_list, x = list(np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)), []
    for i, größe in enumerate(step_list):
        x.append(str(np.round(größe - step_size, 2)) + " bis " + str(größe))
    
    for elem in list_vergleich_oben_unten:
        v_y_vergleich_oben_unten = []
        for VERWENDETE_DATEN in elem:
            M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
            tropfengröße = list(M_dis[:,0])
            erste_index = tropfengröße.index(step_list[0])
            letzter_index = tropfengröße.index(step_list[-1])

            v_y = list(M_dis[:,2])[erste_index:letzter_index+1]
            v_y_vergleich_oben_unten.append(v_y)

        bar_width = 0.4
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width/2, v_y_vergleich_oben_unten[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)) + bar_width/2, v_y_vergleich_oben_unten[1], width=bar_width, label=elem[1])
        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Geschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Geschwindigkeit je Größenklasse - Kameraposition', fontsize = legend_fontsize)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)
        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\vergleich_kameraposition_v_y_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\vergleich_kameraposition_v_y_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()


def vergleich_lastfall():
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
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')  

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')

    ax.set_xlabel("Flutpunktbelastung [%]", fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel("Schwarmexponent", fontsize = font_size_plot, fontname = "Arial")
    ax.legend(loc = "center left", fontsize = legend_fontsize_small, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.7)
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
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')  

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial') 
    
    ax.set_xlabel("Flutpunktbelastung [%]", fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel("Phasenverschiebung [°]", fontsize = font_size_plot, fontname = "Arial")
    ax.legend(loc = "center left", fontsize = legend_fontsize_small, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.7)
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
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')  

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial') 
    
    ax.set_xlabel("Flutpunktbelastung [%]", fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel("Sauterdurchmesser [mm]", fontsize = font_size_plot, fontname = "Arial")
    ax.legend(loc = "center left", fontsize = legend_fontsize_small, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.7)
    ax.tick_params(direction="in")
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\Sauterdurchmesser_lastfall", dpi = 300, bbox_inches='tight')
         
def vergleich_phasenverhältnis():
   
    
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
    ax.set_xticklabels(x_array, fontsize = font_size_plot, fontname = "Arial")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')
    
    ax.set_xlabel("Phasenverhätnis", fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel("Schwarmexponent", fontsize = font_size_plot, fontname = "Arial")
    ax.legend(loc = "center left", fontsize = legend_fontsize_small, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.7)
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

    ax.set_xticklabels(x_array, fontsize = font_size_plot, fontname = "Arial")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')

    ax.set_xlabel("Phasenverhätnis", fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel("Phasenverschiebung [°]", fontsize = font_size_plot, fontname = "Arial")
    ax.tick_params(direction="in")
    ax.legend(loc = "center left", fontsize = legend_fontsize_small, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.7)
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

    ax.set_xticklabels(x_array, fontsize = font_size_plot, fontname = "Arial")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(font_size_plot)
        tick_label.set_fontname('Arial')

    ax.set_ylim([1.65, 2.25])
    ax.set_xlabel("Phasenverhätnis", fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel("Sauterdurchmesser [mm]", fontsize = font_size_plot, fontname = "Arial")
    ax.tick_params(direction="in")
    ax.legend(loc = "center left", fontsize = legend_fontsize_small, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.7)
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\Sauterdurchmesser_phasenverhätnis", dpi = 300 , bbox_inches='tight')


def berechne_relative_anzahl(dic):
    keys = list(dic.keys())
    summe = 0
    for key in keys:
        summe += len(dic[key])

    relative_anzahl = []
    for key in keys:
        relative_anzahl.append(len(dic[key])/summe)
    return relative_anzahl


def plot_lastfall_geschwindigkeitsverteilung():
    for elem in list_alle_lastfall_vergleiche:
        dic_relative_anzahl = {}
        for größe in tropfengröße:
            dic_relative_anzahl[größe] = []
        relative_anzahl_alle_größen = []
        
        for VERWENDETE_DATEN in elem:
            dic_id_geschwindigkeitsverteilung_ges = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_ges_"+VERWENDETE_DATEN, "rb"))
            dic_id_geschwindigkeitsverteilung_alle_größen = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_alle_größen_"+VERWENDETE_DATEN, "rb"))
            
            for größe in dic_id_geschwindigkeitsverteilung_ges:
                if größe != np.inf:
                    dic_id_geschwindigkeitsverteilung = dic_id_geschwindigkeitsverteilung_ges[größe]
                    relative_anzahl = berechne_relative_anzahl(dic_id_geschwindigkeitsverteilung)
                    dic_relative_anzahl[größe].append(relative_anzahl)
            relative_anzahl_alle_größen.append(berechne_relative_anzahl(dic_id_geschwindigkeitsverteilung_alle_größen))
            
        x, keys = [], list(dic_id_geschwindigkeitsverteilung)
        for i in range(len(dic_id_geschwindigkeitsverteilung)):
            if keys[i] == np.inf:
                string = str(keys[i-1])+ " bis " + str(keys[i])
            elif keys[i] == -np.inf:
                string =  str(keys[i]) + " bis " + str(np.round(keys[i+1] - step_size_gesch, 2))
            else:
                string = str(np.round(keys[i] - step_size_gesch, 2)) + " bis " + str(keys[i])
            x.append(string)
            
        for größe in tropfengröße:
            bar_width = 0.25
            fig, ax = plt.subplots(figsize=(16,9))
            ax.bar(np.arange(len(x)) - bar_width, dic_relative_anzahl[größe][0], width=bar_width, label=elem[0])
            ax.bar(np.arange(len(x)), dic_relative_anzahl[größe][1], width=bar_width, label=elem[1])
            ax.bar(np.arange(len(x)) + bar_width, dic_relative_anzahl[größe][2], width=bar_width, label=elem[2])

            ax.set_xlabel('Geschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
            ax.set_ylabel('Relative Anzahl ID aus dic_id [%]', fontsize = font_size_plot, fontname = "Arial")
            ax.set_title('Vergleich relative Anzahl an IDs, größe: ' + str(np.round(größe - step_size, 2)) + " bis " + str(größe) + " mm", fontsize = font_size_plot)
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(x)
            ax.legend(fontsize = legend_fontsize)

            ax.tick_params(direction="in")
            plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
            plt.yticks(fontsize = font_size_plot, fontname = "Arial")
            fig.tight_layout()
            plt.subplots_adjust(bottom = 0.3)
            fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\lastfall_"+str(größe)+"_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
            fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\lastfall_"+str(größe)+"_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
            fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\lastfall_"+str(größe)+"_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
            #plt.show()
            plt.close()

        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width, relative_anzahl_alle_größen[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)), relative_anzahl_alle_größen[1], width=bar_width, label=elem[1])
        ax.bar(np.arange(len(x)) + bar_width, relative_anzahl_alle_größen[2], width=bar_width, label=elem[2])

        ax.set_xlabel('Geschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Anzahl ID aus dic_id [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich relative Anzahl an IDs, größe: alle größen', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\lastfall_alle_größen_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\lastfall_alle_größen_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\lastfall_alle_größen_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()

def plot_phasenverhältnis_geschwindigkeitsverteilung():            
    for elem in list_alle_phasenvergleiche:
        dic_relative_anzahl = {}
        for größe in tropfengröße:
            dic_relative_anzahl[größe] = []
        relative_anzahl_alle_größen = []
        
        for VERWENDETE_DATEN in elem:
            dic_id_geschwindigkeitsverteilung_ges = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_ges_"+VERWENDETE_DATEN, "rb"))
            dic_id_geschwindigkeitsverteilung_alle_größen = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_alle_größen_"+VERWENDETE_DATEN, "rb"))
            
            for größe in dic_id_geschwindigkeitsverteilung_ges:
                if größe != np.inf:
                    dic_id_geschwindigkeitsverteilung = dic_id_geschwindigkeitsverteilung_ges[größe]
                    relative_anzahl = berechne_relative_anzahl(dic_id_geschwindigkeitsverteilung)
                    dic_relative_anzahl[größe].append(relative_anzahl)
            relative_anzahl_alle_größen.append(berechne_relative_anzahl(dic_id_geschwindigkeitsverteilung_alle_größen))

        x, keys = [], list(dic_id_geschwindigkeitsverteilung)
        for i in range(len(dic_id_geschwindigkeitsverteilung)):
            if keys[i] == np.inf:
                string = str(keys[i-1])+ " bis " + str(keys[i])
            elif keys[i] == -np.inf:
                string =  str(keys[i]) + " bis " + str(np.round(keys[i+1] - step_size_gesch, 2))
            else:
                string = str(np.round(keys[i] - step_size_gesch, 2)) + " bis " + str(keys[i])
            x.append(string)
            
        for größe in tropfengröße:
            bar_width = 0.25
            fig, ax = plt.subplots(figsize=(16,9))
            ax.bar(np.arange(len(x)) - bar_width, dic_relative_anzahl[größe][0], width=bar_width, label=elem[0])
            ax.bar(np.arange(len(x)), dic_relative_anzahl[größe][1], width=bar_width, label=elem[1])
            ax.bar(np.arange(len(x)) + bar_width, dic_relative_anzahl[größe][2], width=bar_width, label=elem[2])

            ax.set_xlabel('Geschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
            ax.set_ylabel('Relative Anzahl ID aus dic_id [%]', fontsize = font_size_plot, fontname = "Arial")
            ax.set_title('Vergleich relative Anzahl an IDs, größe: ' + str(np.round(größe - step_size, 2)) + " bis " + str(größe)+ " mm", fontsize = font_size_plot)
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(x)
            ax.legend(fontsize = legend_fontsize)

            ax.tick_params(direction="in")
            plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
            plt.yticks(fontsize = font_size_plot, fontname = "Arial")
            fig.tight_layout()
            plt.subplots_adjust(bottom = 0.3)
            fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\phasenvergleich_"+str(größe)+"_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
            fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\phasenvergleich_"+str(größe)+"_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
            fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\phasenvergleich_"+str(größe)+"_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
            plt.close()

        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width, relative_anzahl_alle_größen[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)), relative_anzahl_alle_größen[1], width=bar_width, label=elem[1])
        ax.bar(np.arange(len(x)) + bar_width, relative_anzahl_alle_größen[2], width=bar_width, label=elem[2])

        ax.set_xlabel('Geschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Anzahl ID aus dic_id [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich relative Anzahl an IDs, größe: alle größen', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\phasenvergleich_alle_größen_"+ elem[0] + ".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\phasenvergleich_alle_größen_"+ elem[1] + ".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\phasenvergleich_alle_größen_"+ elem[2] + ".png", dpi = 300, bbox_inches='tight')
        plt.close()

list_alle_lastfall_vergleiche, list_vergleich_oben_unten, list_alle_phasenvergleiche = erzeuge_vergleiche()
vergleich_oben_unten()

plot_lastfall_geschwindigkeitsverteilung()
plot_phasenverhältnis_geschwindigkeitsverteilung()

vergleich_lastfall()
vergleich_phasenverhältnis()


#40 = Blaue
#80 = Grun
#90 = Rot
