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


y=20
x=15
h=64
w=63
# Read image
im3 = cv2.imread("Infra3103_2019.tif").astype(np.float)
im3 = im3[y:y+h, x:x+w]
# Read image
im3a = cv2.imread("Vermelho3103_2019.tif").astype(np.float)
im3a = im3a[y:y+h, x:x+w]
im3b = cv2.imread("Verde3103_2019.tif").astype(np.float)
im3b = im3b[y:y+h, x:x+w]
im3c = cv2.imread("Azul3103_2019.tif").astype(np.float)
im3c = im3c[y:y+h, x:x+w]
I3a= im3[0, :, 0]
I3b= im3[10, :, 0]
I3c= im3[20, :, 0]
I3d= im3[30, :, 0]
I3e= im3[40, :, 0]
I3 = [*I3a, *I3b, *I3c, *I3d, *I3e]
I3 = np.array(I3)
#R3= im3a[0, :, 0]
R3a= im3a[0, :, 0]
R3b= im3a[10, :, 0]
R3c= im3a[20, :, 0]
R3d= im3a[30, :, 0]
R3e= im3a[40, :, 0]
R3 = [*R3a, *R3b, *R3c, *R3d, *R3e]
R3 = np.array(R3)
#G3= im3b[0, :, 0]
G3a= im3b[0, :, 0]
G3b= im3b[10, :, 0]
G3c= im3b[20, :, 0]
G3d= im3b[30, :, 0]
G3e= im3b[40, :, 0]
G3 = [*G3a, *G3b, *G3c, *G3d, *G3e]
G3 = np.array(G3)
#B3= im3c[0, :, 0]
B3a= im3c[0, :, 0]
B3b= im3c[10, :, 0]
B3c= im3c[20, :, 0]
B3d= im3c[30, :, 0]
B3e= im3c[40, :, 0]
B3 = [*B3a, *B3b, *B3c, *B3d, *B3e]
B3 = np.array(B3)
I3 = (I3 - R3) / (I3 + R3)
M = 0.8841 - (0.0131 * R3) + (0.0023 * G3) + (0.0018 * B3)

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
