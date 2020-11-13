#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:49:13 2020

@author: usuario
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas as pd

y=20
x=15
h=64
w=63

im3 = cv2.imread("Infra3103_2019.tif").astype(np.float)
im3 = im3[y:y+h, x:x+w]
im3a = cv2.imread("Vermelho3103_2019.tif").astype(np.float)
im3a = im3a[y:y+h, x:x+w]
im3b = cv2.imread("Verde3103_2019.tif").astype(np.float)
im3b = im3b[y:y+h, x:x+w]
im3c = cv2.imread("Azul3103_2019.tif").astype(np.float)
im3c = im3c[y:y+h, x:x+w]
I= im3[:, 0, 0]
R= im3a[:, 0, 0]
G= im3b[:, 0, 0]
B= im3c[:, 0, 0]

with np.errstate(invalid='ignore'):
    NDVIc = 0.7556 - (0.0129*R) + (0.0024*G) + (0.0046*B)
    NDVI = (I - R)/(I + R)
    r = R / (R + G + B)
    g = G / (R + G + B)
    b = B / (R + G + B)
    ExG = (2*g) - r - b
    ExGR = ExG - ((1.4*r) - g)
    VEG = g / ((r**0.667) * (b ** 0.333))
    CIVE = (0.441 * r) - (0.881 * g) + (0.385 * b) + 18.78745
    COM = 0.25 * ExG + 0.30 * ExGR + 0.33 * CIVE + 0.12 * VEG
    RGBVI = ((G * G) - (R * B))/((G * G) + (R * B))
    GLI = ((2*G) - R - B) / ((2*G) + R + B)
    VARI = (G - R) / (G + R - B)
    MPRI = (G - R) / (G + R)
    TGI = G - (0.39 * R) - (0.61 * B)
    RGVBI = (G - (B * R)) / ((G * G) + (B * R))
    MGVRI = ((G * G) - (R * R))/ ((G * G) + (R * R))

df = pd.DataFrame({'NDVI': NDVI, 'NDVIc': NDVIc, 'r': r, 'g': g, 'b': b, 'ExG': ExG, 'ExGR': ExGR, 'VEG': VEG, 'CIVE': CIVE, 'COM': COM, 'RGBVI': RGBVI, 'GLI': GLI, 'VARI': VARI, 'MPRI': MPRI, 'TGI': TGI, 'RGVBI': RGVBI, 'MGVRI': MGVRI})
#df = pd.DataFrame({'NDVIc': NDVIc, 'r': r, 'g': g, 'b': b, 'ExG': ExG, 'ExGR': ExGR, 'VEG': VEG, 'CIVE': CIVE, 'COM': COM, 'RGBVI': RGBVI, 'GLI': GLI, 'VARI': VARI, 'MPRI': MPRI, 'TGI': TGI, 'RGVBI': RGVBI, 'MGVRI': MGVRI})
corr = df.corr()
import seaborn as sns
#sns.heatmap(corr, xticklabels=corr.columns, cmap = "RdYlGn")

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, linewidths=.5, cmap = 'RdYlGn')

    