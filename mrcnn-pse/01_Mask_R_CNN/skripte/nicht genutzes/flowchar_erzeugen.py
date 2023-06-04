from pyflowchart import *


datei_name = "C:\\Users\\StefanvenderBuss\\Desktop\\BA_git\\mrcnn-pse\\01_Mask_R_CNN\\samples\\droplet\\test.py"

with open(datei_name, "r") as file:
    code = file.read()

fc = Flowchart.from_code(code)

print(fc.flowchart())
