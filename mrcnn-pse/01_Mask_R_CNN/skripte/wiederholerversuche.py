import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 
import shutil

font_size_plot = 30
legend_fontsize = 24
ROOT_DIR = "C:\\Users\\\stefa\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN"
step_size_gesch, untere_grenze_gesch, obere_grenze_gesch = 10, -100, 160
step_size, untere_grenze, obere_grenze = 0.2, 0, 3
remove_data = True
step_list = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)

Wiederholerversuche = [["20221124_Oben_1,2_1_40", "20221130_Oben_1,2_1_40"],
                ["20221124_Oben_1,2_1_80", "20221130_Oben_1,2_1_80"],
                ["20221124_Oben_1,2_1_90", "20221130_Oben_1,2_1_90"],
                ["20221124_Oben_1,2_1_100", "20221130_Oben_1,2_1_100"],
                ["20221124_Unten_1,2_1_40", "20221130_Unten_1,2_1_40"],
                ["20221124_Unten_1,2_1_80", "20221130_Unten_1,2_1_80"],
                ["20221124_Unten_1,2_1_90", "20221130_Unten_1,2_1_90"],
                ["20221124_Unten_1,2_1_100", "20221130_Unten_1,2_1_100"]]

def plot_gesch_wiederholder(y1,y2,x,elem,speicher_name, max_anzahl,größe):
    bar_width = 0.35
    if größe != None:
        p_text = "größe: " + str(np.round(größe-step_size,2)) + " bis " + str(größe) + " mm"
    else:
        p_text = "größe: alle größen"

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(np.arange(len(x)) - bar_width/2, y1, width=bar_width, label=elem[0])
    ax.bar(np.arange(len(x)) + bar_width/2, y2, width=bar_width, label=elem[1])

    ax.set_xlabel('Geschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
    ax.set_ylabel('Relative Anzahl ID aus dic_id [%]', fontsize = font_size_plot, fontname = "Arial")
    ax.set_title('Vergleich relative Anzahl an IDs, ' + p_text, fontsize = font_size_plot)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.set_ylim([0, max_anzahl])
    ax.legend(fontsize = legend_fontsize)

    ax.tick_params(direction="in")
    plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
    plt.yticks(fontsize = font_size_plot, fontname = "Arial")
    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\wiederholerversuch\\"+speicher_name[0]+".png", dpi = 300, bbox_inches='tight')
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\wiederholerversuch\\"+speicher_name[1]+".png", dpi = 300, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_geschwindigkeitsverteilung_wiederholerversuch():
    for elem in Wiederholerversuche:
        dic_id_geschwindigkeitsverteilung_ges_erster = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_ges_"+elem[0], "rb"))
        dic_id_geschwindigkeitsverteilung_alle_größen_erster = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_alle_größen_"+elem[0], "rb"))

        dic_id_geschwindigkeitsverteilung_ges_zweiter = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_ges_"+elem[1], "rb"))
        dic_id_geschwindigkeitsverteilung_alle_größen_zweiter = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_alle_größen_"+elem[1], "rb"))

        x, keys = [], list(dic_id_geschwindigkeitsverteilung_alle_größen_erster)
        for i in range(len(dic_id_geschwindigkeitsverteilung_alle_größen_erster)):
            if keys[i] == np.inf:
                string = str(keys[i-1])+ " bis " + str(keys[i])
            elif keys[i] == -np.inf:
                string =  str(keys[i]) + " bis " + str(np.round(keys[i+1] - step_size_gesch, 2))
            else:
                string = str(np.round(keys[i] - step_size_gesch, 2)) + " bis " + str(keys[i])
            x.append(string)

        try:
            os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\wiederholerversuch")
            os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\wiederholerversuch")
        except:
            if remove_data == True:
                shutil.rmtree(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\wiederholerversuch")
                os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\wiederholerversuch")
                shutil.rmtree(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\wiederholerversuch")
                os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\wiederholerversuch")
            else:
                sys.exit("Datei gibt es schon")

        alle_y1 = []
        alle_y2 = []
        for i in step_list:
            dic_eins = dic_id_geschwindigkeitsverteilung_ges_erster[i]
            dic_zwei = dic_id_geschwindigkeitsverteilung_ges_zweiter[i]
            y1_abs = list(dic_eins.values())
            y2_abs = list(dic_zwei.values())

            anzahl_y1, anzahl_y2 = 0,0
            for ii in range(len(y1_abs)):
                anzahl_y1 += len(y1_abs[ii])
                anzahl_y2 += len(y2_abs[ii])
            y1, y2 = [], []
            for ii in range(len(y1_abs)):
                y1.append(len(y1_abs[ii])/anzahl_y1)
                y2.append(len(y2_abs[ii])/anzahl_y2)

            alle_y1.append(y1)
            alle_y2.append(y2)

        max_anzahl = 0
        for i in range(len(alle_y1)):
            if max_anzahl < max(max(alle_y1[i]), max(alle_y2[i])):
                max_anzahl = max(max(alle_y1[i]), max(alle_y2[i]))
                                                                
        for i in range(len(step_list)):
            plot_gesch_wiederholder(alle_y1[i],alle_y2[i],x,elem,["wiederholderversuch_"+str(step_list[i])+"_"+elem[0],"wiederholderversuch_"+str(step_list[i])+"_"+elem[1]], max_anzahl, step_list[i])

        dic_eins = dic_id_geschwindigkeitsverteilung_alle_größen_erster
        dic_zwei = dic_id_geschwindigkeitsverteilung_alle_größen_zweiter
        y1_abs = list(dic_eins.values())
        y2_abs = list(dic_zwei.values())

        anzahl_y1, anzahl_y2 = 0,0
        for ii in range(len(y1_abs)):
            anzahl_y1 += len(y1_abs[ii])
            anzahl_y2 += len(y2_abs[ii])
        y1, y2 = [], []
        for ii in range(len(y1_abs)):
            y1.append(len(y1_abs[ii])/anzahl_y1)
            y2.append(len(y2_abs[ii])/anzahl_y2)
                
        plot_gesch_wiederholder(y1,y2,x,elem,["wiederholderversuch_alle_größen_"+elem[0],"wiederholderversuch_alle_größen_"+elem[1]],max(max(y1),max(y2)), None)

def plot_modell_wiederholder(y1,y2,x,elem,speicher_name,modus):
   
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(np.arange(len(x)) - bar_width/2, y1, width=bar_width, label=elem[0])
    ax.bar(np.arange(len(x)) + bar_width/2, y2, width=bar_width, label=elem[1])

    if modus == 1:
        ax.set_xlabel('Durchmesser [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Tropfenaufstiegsgeschwindigkeit [mm/s]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich gemessene Aufstiegsgeschwindigkeit - Wiederholerversuch', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
    if modus == 2:
        ax.set_xlabel('Durchmesser [mm]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_ylabel('Relative Anzahl ID [%]', fontsize = font_size_plot, fontname = "Arial")
        ax.set_title('Vergleich Relative Anzahl ID - Wiederholerversuch', fontsize = font_size_plot)
        ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.legend(fontsize = legend_fontsize)

    ax.tick_params(direction="in")
    plt.xticks(rotation=45, ha='right', fontsize = font_size_plot, fontname = "Arial")
    plt.yticks(fontsize = font_size_plot, fontname = "Arial")
    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\wiederholerversuch\\"+speicher_name[0]+".png", dpi = 300, bbox_inches='tight')
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\wiederholerversuch\\"+speicher_name[1]+".png", dpi = 300, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_geschwindigkeitsverteilung_durchmesser():
    for elem in Wiederholerversuche:
        M_dis_erster = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+elem[0]+"\\M_dis", "rb"))
        M_dis_zweiter = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+elem[1]+"\\M_dis", "rb"))

        obere_grenze_betrachteter_bereich = 2.8
        untere_grenze_betrachteter_bereich = 0.4
    
        tropfengröße = list(M_dis_erster[:,0])
        erste_index = tropfengröße.index(untere_grenze_betrachteter_bereich)
        letzter_index = tropfengröße.index(obere_grenze_betrachteter_bereich)

        x = []
        for i, größe in enumerate(step_list[erste_index:letzter_index+1]):
            x.append(str(np.round(größe - step_size, 2)) + " bis " + str(größe))

        anzahl_tropfen_erster, v_y_erster = list(M_dis_erster[:,1])[erste_index:letzter_index+1], list(M_dis_erster[:,2])[erste_index:letzter_index+1]
        anzahl_tropfen_zweiter, v_y_zweiter = list(M_dis_zweiter[:,1])[erste_index:letzter_index+1], list(M_dis_zweiter[:,2])[erste_index:letzter_index+1]

        relative_tropfen_anzahl_erster = []
        relative_tropfen_anzahl_zweiter = []

        summe_tropfen_erster = sum(anzahl_tropfen_erster)
        summe_tropfen_zweiter = sum(anzahl_tropfen_zweiter)
        
        for i in range(len(anzahl_tropfen_erster)):
            relative_tropfen_anzahl_erster.append(anzahl_tropfen_erster[i]/summe_tropfen_erster)
            relative_tropfen_anzahl_zweiter.append(anzahl_tropfen_zweiter[i]/summe_tropfen_zweiter)

        plot_modell_wiederholder(v_y_erster,v_y_zweiter,x,elem,["v_y_vergleich_"+elem[0],"v_y_vergleich_"+elem[1]],1)
        plot_modell_wiederholder(relative_tropfen_anzahl_erster,relative_tropfen_anzahl_zweiter,x,elem,["anzahl_tropfen_zweiter_vergleich_"+elem[0],"anzahl_tropfen_zweiter_vergleich_"+elem[1]],2)

plot_geschwindigkeitsverteilung_wiederholerversuch()
plot_geschwindigkeitsverteilung_durchmesser()

