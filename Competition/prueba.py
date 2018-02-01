#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:02:06 2017

@author: root
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:47:08 2017

@author: avergara
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import csv

train_data = pd.read_csv("traindata.csv", delimiter=',')
test_data = pd.read_csv("testdata.csv", delimiter = ',')

cols = ["Dia", "Hora", "Mes", "Carretera", "Pk", "Km", "Tipo_Est",
       "Hm", "Sentido", "GPS_z", "IMD", "Arboles_Metros_Calzada",
       "Colision_Vehiculo_Obstaculo_1", "Colision_Vehiculo_Obstaculo_2"]

train_data = train_data[list(train_data)[4:]]
train_data = train_data.drop(cols, axis=1)
train_data.fillna(value=0, inplace=True)

test_data = test_data[list(test_data)[4:]]
test_data = test_data.drop(cols, axis=1)
test_data.fillna(value=0, inplace=True)

#Separar los datos de train en test y train para verlo

#Incluimos varias features a revisar
train_data["Y"] = np.where((train_data["N_Muertos"] > 0) | (train_data["N_Graves"] > 0) | (train_data["N_Leves"] > 0), 1, 0) 

#Modelo

model = SVC(kernel="poly", decision_function_shape="ovr", random_state=42, C = 0.3)

del train_data["N_Muertos"]
del train_data["N_Leves"]
del train_data["N_Graves"]

X = train_data[list(train_data)[:-1]]
Y = train_data["Y"]

#----------------------------------------------------------
#Esto sirve para ver el accuracy
#Separar los datos de train en test y train para verlo
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#Modelo
scalar = StandardScaler()
X_train_n = scalar.fit_transform(X_train)
X_test_n = scalar.fit_transform(X_test)

model.fit(X_train_n, y_train)

y_pred_test = model.predict(X_test_n)

print classification_report(y_test, y_pred_test)
#---------------------------------------------------------
"""
# Hago el fit y predict sobre los datos de test

model.fit(X_test, Y)

y_pred = model.predict(test_data)

id = 1

with open('testfinal.csv' , 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Prediction'])
    for i in y_pred:
        writer.writerow([id,i])
        id = id + 1
"""
print "DONE"
           


