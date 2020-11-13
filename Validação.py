#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:00:47 2020

@author: usuario
"""


import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



# Read image
im3 = cv2.imread("Infra3103_2019.tif").astype(np.float)
# Read image
im3a = cv2.imread("Vermelho3103_2019.tif").astype(np.float)
im3b = cv2.imread("Verde3103_2019.tif").astype(np.float)
im3c = cv2.imread("Azul3103_2019.tif").astype(np.float)
I3= im3[:, 0, 0]
R3= im3a[:, 0, 0]
G3= im3b[:, 0, 0]
B3= im3c[:, 0, 0]

I3 = (I3 - R3) / (I3 + R3)
M = 0.7556 - (0.0129 * R3) + (0.0024 * G3) + (0.0046 * B3)
#M = (255 * (M - (min(M))))/(max(M)-min(M))
#M = M.astype(np.uint8)

Corr3 = stats.pearsonr(I3, M)
print(Corr3)
Erro = np.int8(I3 - M)
soma = sum(Erro)
ErroMedio = soma/100
print(ErroMedio)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(I3, M))
print(rms)

M2 = 1.0968 - (0.0175*R3)
#M2 = (255 * (M2 - (min(M2))))/(max(M2)-min(M2))
#M2 = M2.astype(np.uint8)

Corr3 = stats.pearsonr(I3, M2)
print(Corr3)
Erro = np.int8(I3 - M2)
soma = sum(Erro)
ErroMedio = soma/100
print(ErroMedio)


rms = sqrt(mean_squared_error(I3, M2))
print(rms)

M3 = 72.8943 + (1.7677*G3)
M3 = (255 * (M3 - (min(M3))))/(max(M3)-min(M3))
M3 = M3.astype(np.uint8)

Corr3 = stats.pearsonr(I3, M3)
print(Corr3)
Erro = np.int8(I3 - M3)
soma = sum(Erro)
ErroMedio = soma/100
print(ErroMedio)


rms = sqrt(mean_squared_error(I3, M3))
print(rms)

M4 = 71.4530 + (2.3865*B3)
M4 = (255 * (M4 - (min(M4))))/(max(M4)-min(M4))
M4 = M4.astype(np.uint8)

Corr3 = stats.pearsonr(I3, M4)
print(Corr3)
Erro = np.int8(I3 - M4)
soma = sum(Erro)
ErroMedio = soma/100
print(ErroMedio)

rms = sqrt(mean_squared_error(I3, M4))
print(rms)



