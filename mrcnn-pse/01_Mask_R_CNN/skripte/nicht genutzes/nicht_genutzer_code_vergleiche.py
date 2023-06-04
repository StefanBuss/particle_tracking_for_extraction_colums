"""
    print(list_vergleich_oben_unten)
    for elem in list_vergleich_oben_unten:
        verwendete_daten_oben = elem[0]
        verwendete_daten_unten = elem[1]
        
        image1 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + verwendete_daten_oben + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        image2 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + verwendete_daten_unten + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")

        size = image1.size

        new_size = (size[0], 2 * size[1])
        new_image = Image.new("RGB", new_size)

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, size[1]))

        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + verwendete_daten_oben + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_oben_unten.png")
        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + verwendete_daten_unten + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_oben_unten.png")

    print("ende vergleich oben unten\n")        
    """



 """
    print(list_alle_lastfall_vergleiche)
    for elem in list_alle_lastfall_vergleiche:
        image1 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        image2 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        image3 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        
        size = image1.size

        new_size = (size[0], 3 * size[1])
        new_image = Image.new("RGB", new_size)

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, size[1]))
        new_image.paste(image3, (0, 2 * size[1]))

        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_lastfall.png")
        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_lastfall.png")
        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_lastfall.png")
    print("ende vergleich lastfall\n") 
    """



  """
    print(list_alle_phasenvergleiche)
    for elem in list_alle_phasenvergleiche:
        image1 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        image2 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        image3 = Image.open(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\AnzahlderTropfen_Geschwindigkeit_RGB_None.png")
        
        size = image1.size

        new_size = (size[0], 3 * size[1])
        new_image = Image.new("RGB", new_size)

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, size[1]))
        new_image.paste(image3, (0, 2 * size[1]))

        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[0] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_phasenverhältnis.png")
        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[1] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_phasenverhältnis.png")
        new_image.save(ROOT_DIR + "\\datasets\\output_analyseddata\\" + elem[2] + "\\AnzahlderTropfen_Geschwindigkeit_RGB\\vergleich_phasenverhältnis.png")
    print("ende vergleich phasenverhältnis\n")
    #print(combinations)
    """
