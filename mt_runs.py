import os

##### https://stackoverflow.com/questions/51232600/

# args = [1,2,3,4,5]
args = [5]
for arg in args:
    os.system("python main_mt.py --seed {}".format(arg))

# args = [2,3,4,5]
# for arg in args:
#     os.system("python ./optim_prob/div_calc.py --seed {}".format(arg))
#     os.system("python ./optim_prob/main_script_phi_e.py --seed {}".format(arg))
#     os.system("python ./main_mt.py --seed {}".format(arg))
