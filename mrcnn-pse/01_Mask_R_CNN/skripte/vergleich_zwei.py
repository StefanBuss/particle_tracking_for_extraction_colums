from PIL import Image, ImageDraw, ImageFont
import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import os

ROOT_DIR = "C:\\Users\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"


font_size_plot = 30
legend_fontsize = 24
step_size_gesch, untere_grenze_gesch, obere_grenze_gesch = 10, -100, 160
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

tropfengröße = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)

def load_data(VERWENDETE_DATEN):
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


def morphologie_eines_tropfens(r):
    height = abs(r[2]-r[0])
    width = abs(r[3]-r[1])
    center_y = abs(r[0] + height/2)
    center_x = abs(r[1] + width/2)
    durchmesser = (height + width)/2

    return [height, width, center_y, center_x, durchmesser]


def größenverteilung_id(VERWENDETE_DATEN):
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

def avg_durchmesser_id(ID,dic_id,list_r):
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

def Volumen_berechnung_kugel(durchmesser):
    return np.pi / 6 * durchmesser**3

def Volumenanteil_pro_groeßenklasse(VERWENDETE_DATEN,list_r):
    dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "rb"))
    dic_id = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
    
    v_ges = 0
    for ID in dic_id:
        avg_durchmesser = avg_durchmesser_id(ID,dic_id,list_r)
        v_ges += Volumen_berechnung_kugel(avg_durchmesser)

    v_rel = []
    for größe in dic_id_groeßenverteilung:
        if größe != np.inf:
            v_größe = 0
            for ID in dic_id_groeßenverteilung[größe]:
                avg_durchmesser = avg_durchmesser_id(ID,dic_id,list_r)
                v_größe += Volumen_berechnung_kugel(avg_durchmesser)

            v_rel.append(v_größe/v_ges)

    return v_rel


def plot_kameraposition():
    for elem in list_vergleich_oben_unten:
        list_relatives_volumen_id = []
        list_releative_anzahl_id = []
        for VERWENDETE_DATEN in elem:
            list_r, list_elem, ende = load_data(VERWENDETE_DATEN)
            list_releative_anzahl_id.append(größenverteilung_id(VERWENDETE_DATEN))
            list_relatives_volumen_id.append(Volumenanteil_pro_groeßenklasse(VERWENDETE_DATEN,list_r))
        tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
        for i in range(len(tropfengröße)):
            if tropfengröße[i] != np.inf:
                string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
            else:
                string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
            x.append(string)


        bar_width = 0.4
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width/2, list_releative_anzahl_id[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)) + bar_width/2, list_releative_anzahl_id[1], width=bar_width, label=elem[1])

        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Anzahl [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relative Anzahl von IDs - Kameraposition', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\kameraposition_relative_anzahl_id_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\kameraposition_relative_anzahl_id_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()

        


        bar_width = 0.4
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width/2, list_relatives_volumen_id[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)) + bar_width/2, list_relatives_volumen_id[1], width=bar_width, label=elem[1])

        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Volumen [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relatives Volumen von IDs - Kameraposition', fontsize = font_size_plot, fontname = "Arial")
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\kameraposition_list_relatives_volumen_id_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\kameraposition_list_relatives_volumen_id_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()

        
def plot_lastfall():
    for elem in list_alle_lastfall_vergleiche:
        list_relatives_volumen_id = []
        list_releative_anzahl_id = []
        for VERWENDETE_DATEN in elem:
            list_r, list_elem, ende = load_data(VERWENDETE_DATEN)
            list_releative_anzahl_id.append(größenverteilung_id(VERWENDETE_DATEN))
            list_relatives_volumen_id.append(Volumenanteil_pro_groeßenklasse(VERWENDETE_DATEN,list_r))

        tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
        for i in range(len(tropfengröße)):
            if tropfengröße[i] != np.inf:
                string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
            else:
                string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
            x.append(string)
            

        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width, list_releative_anzahl_id[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)), list_releative_anzahl_id[1], width=bar_width, label=elem[1])
        ax.bar(np.arange(len(x)) + bar_width, list_releative_anzahl_id[2], width=bar_width, label=elem[2])

        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Anzahl [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relative Anzahl von IDs - Lastfall', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\lastfall_relative_anzahl_id_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\lastfall_relative_anzahl_id_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\lastfall_relative_anzahl_id_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()


        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width, list_relatives_volumen_id[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)), list_relatives_volumen_id[1], width=bar_width, label=elem[1])
        ax.bar(np.arange(len(x)) + bar_width, list_relatives_volumen_id[2], width=bar_width, label=elem[2])

        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Volumen [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relatives Volumen von IDs - Lastfall', fontsize = font_size_plot, fontname = "Arial")
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\lastfall_list_relatives_volumen_id_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\lastfall_list_relatives_volumen_id_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\lastfall_list_relatives_volumen_id_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()
            
            
def plot_phasenverhältnis():            
    for elem in list_alle_phasenvergleiche:
        list_relatives_volumen_id = []
        list_releative_anzahl_id = []
        for VERWENDETE_DATEN in elem:
            list_r, list_elem, ende = load_data(VERWENDETE_DATEN)
            list_releative_anzahl_id.append(größenverteilung_id(VERWENDETE_DATEN))
            list_relatives_volumen_id.append(Volumenanteil_pro_groeßenklasse(VERWENDETE_DATEN,list_r))


        tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
        for i in range(len(tropfengröße)):
            if tropfengröße[i] != np.inf:
                string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
            else:
                string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
            x.append(string)

        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width, list_releative_anzahl_id[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)), list_releative_anzahl_id[1], width=bar_width, label=elem[1])
        ax.bar(np.arange(len(x)) + bar_width, list_releative_anzahl_id[2], width=bar_width, label=elem[2])

        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Anzahl [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relative Anzahl von IDs - Phasenverhältnis', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\phasenverhältnis_relative_anzahl_id_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\phasenverhältnis_relative_anzahl_id_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\phasenverhältnis_relative_anzahl_id_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()




        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(16,9))
        ax.bar(np.arange(len(x)) - bar_width, list_relatives_volumen_id[0], width=bar_width, label=elem[0])
        ax.bar(np.arange(len(x)), list_relatives_volumen_id[1], width=bar_width, label=elem[1])
        ax.bar(np.arange(len(x)) + bar_width, list_relatives_volumen_id[2], width=bar_width, label=elem[2])

        ax.set_xlabel('Tropfengröße [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Volumen [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relatives Volumen von IDs - Phasenverhältnis', fontsize = font_size_plot, fontname = "Arial")
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.legend(fontsize = legend_fontsize)

        ax.tick_params(direction="in")
        plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
        plt.yticks(fontsize = font_size_plot, fontname = "Arial")
        fig.tight_layout()
        plt.subplots_adjust(bottom = 0.3)
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\phasenverhältnis_list_relatives_volumen_id_"+elem[0]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\phasenverhältnis_list_relatives_volumen_id_"+elem[1]+".png", dpi = 300, bbox_inches='tight')
        fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\phasenverhältnis_list_relatives_volumen_id_"+elem[2]+".png", dpi = 300, bbox_inches='tight')
        #plt.show()
        plt.close()
      
list_alle_lastfall_vergleiche, list_vergleich_oben_unten, list_alle_phasenvergleiche = erzeuge_vergleiche()
plot_kameraposition()
plot_lastfall()
plot_phasenverhältnis()
