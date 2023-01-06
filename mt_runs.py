import os

##### https://stackoverflow.com/questions/51232600/

args = [1,2,3,4,5]
for arg in args:
    os.system("python main_mt.py --seed {}".format(arg))