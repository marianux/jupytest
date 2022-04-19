#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:24:22 2021

@author: mariano
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


dd = pd.read_excel("Resultados3.xlsx")


X = dd[['WT2', 'VQ', 'VT1', 'VT2' ]].to_numpy()
Y = dd[['target' ]].to_numpy()

clf = LinearDiscriminantAnalysis()
clf.fit(X, Y)

ConfusionMatrixDisplay.from_estimator(clf, X, Y)
