import os

##### https://stackoverflow.com/questions/51232600/

args = [5]#,2,3,4,5]
# args = [5]
for arg in args:
    os.system("python div_calc.py --seed {}".format(arg))
    os.system("python main_script_phi_e.py --seed {}".format(arg))
