# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 13:35:40 2023

@author: ch5b2
"""

import os

##### https://stackoverflow.com/questions/51232600/

args = [1,2,3,4,5]
for arg in args:
    os.system("python compare_mt_iclr.py --seed {}".format(arg))
