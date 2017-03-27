# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:18:18 2017

@author: avergara
"""
import pandas as pd
diabetes = pd.read_csv("diabetes.csv", delimiter=';')

print diabetes.describe()