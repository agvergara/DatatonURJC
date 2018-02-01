#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:47:08 2017

@author: avergara
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
import csv

ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42

#SHUFFLE THIS DATA, BECAUSE RANDOM IS COOL
train_data = pd.read_csv("traindata.csv", delimiter=',')
test_data = pd.read_csv("testdata.csv", delimiter = ',')

#DROP THAT ROWS
cols = ["Dia", "Dia_Semana", "Arboles", "Carretera", "Tipo_Est",
       "Pk", "Km"]

train_data = train_data[list(train_data)[4:]]
train_data = train_data.drop(cols, axis=1)

test_data = test_data[list(test_data)[4:]]
test_data = test_data.drop(cols, axis=1)

train_data["Y"] = np.where((train_data["N_Muertos"] > 0) | (train_data["N_Graves"] > 0) | (train_data["N_Leves"] > 0), 1, 0) 

#Gradient Boosting, because Boosting is F-U-N
model = GradientBoostingClassifier()

del train_data["N_Muertos"]
del train_data["N_Leves"]
del train_data["N_Graves"]

X = train_data[list(train_data)[:-1]]
Y = train_data["Y"]

#----------------------------------------------------------
#Training data
#Split traning data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Just transform those NaN values into something USEFULL!
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X_train)

X_train_imp = imp.transform(X_train)

#FIT
model.fit(X_train_imp, y_train)

X_test_imp = imp.transform(X_test)
#PREDICT
y_pred_test = model.predict(X_test_imp)

print "Accuracy -> ", accuracy_score(y_test, y_pred_test)
print "F-Score -> ", f1_score(y_test, y_pred_test)
#---------------------------------------------------------

#Test data

#Just transform those NaN values into something USEFULL!
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X)
X_imp = imp.transform(X)

model.fit(X_imp, Y)

test_data_imp = imp.transform(test_data)
y_pred = model.predict(test_data_imp)

id = 1

with open('finaltest.csv' , 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Prediction'])
    for i in y_pred:
        writer.writerow([id,i])
        id = id + 1

print "DONE"
           


