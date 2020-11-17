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
I= im3[:, :, 0]
R= im3a[:, :, 0]
G= im3b[:, :, 0]
B= im3c[:, :, 0]

with np.errstate(invalid='ignore'):
    NDVI = (I - R) / (I + R)
    media0 = np.average(NDVI, axis=0)
    NDVIc = 0.7669 - (0.0132*R) + (0.0017*G) + (0.0056*B)
    media = np.average(NDVIc, axis=0)
    r = R / (R + G + B)
    media2 = np.average(r, axis=0)
    g = G / (R + G + B)
    media3 = np.average(g, axis=0)
    b = B / (R + G + B)
    media4 = np.average(b, axis=0)
    ExG = (2*g) - r - b
    media5 = np.average(ExG, axis=0)
    ExGR = ExG - ((1.4*r) - g)
    media6 = np.average(ExGR, axis=0)
    VEG = g / ((r**0.667) * (b ** 0.333))
    media7 = np.average(VEG, axis=0)
    CIVE = (0.441 * r) - (0.881 * g) + (0.385 * b) + 18.78745
    media8 = np.average(CIVE, axis=0)
    COM = (0.25 * ExG) + (0.30 * ExGR) + (0.33 * CIVE) + (0.12 * VEG)
    media9 = np.average(COM, axis=0)
    RGBVI = ((G * G) - (R * B))/((G * G) + (R * B))
    media10 = np.average(RGBVI, axis=0)
    GLI = ((2*G) - R - B) / ((2*G) + R + B)
    media11 = np.average(GLI, axis=0)
    VARI = (G - R) / (G + R - B)
    media12 = np.average(VARI, axis=0)
    MPRI = (G - R) / (G + R)
    media13 = np.average(MPRI, axis=0)
    TGI = G - (0.39 * R) - (0.61 * B)
    media14 = np.average(TGI, axis=0)
    RGVBI = (G - (B * R)) / ((G * G) + (B * R))
    media15 = np.average(RGVBI, axis=0)
    MGVRI = ((G * G) - (R * R))/ ((G * G) + (R * R))
    media16 = np.average(MGVRI, axis=0)

imgplot = plt.imshow(NDVI)
imgplot.set_cmap('RdYlGn')
min1 = NDVIc[np.isfinite(NDVI)].min()
max1 = NDVIc[np.isfinite(NDVI)].max() 
plt.clim(min1, max1)
plt.colorbar()
plt.axis('off')
plt.title('NDVI')
pylab.show()

imgplot = plt.imshow(NDVIc)
imgplot.set_cmap('RdYlGn')
min1 = NDVIc[np.isfinite(NDVIc)].min()
max1 = NDVIc[np.isfinite(NDVIc)].max() 
plt.clim(min1, max1)
plt.colorbar()
plt.axis('off')
plt.title('NDVIc')
pylab.show()

imgplot = plt.imshow(r)
imgplot.set_cmap('RdYlGn')
#min1 = r[np.isfinite(r)].min()
#max1 = r[np.isfinite(r)].max() 
plt.clim(0, 1)
plt.colorbar()
plt.axis('off')
plt.title('R normalizado')
pylab.show()

imgplot = plt.imshow(g)
imgplot.set_cmap('RdYlGn')
#min1 = g[np.isfinite(g)].min()
#max1 = g[np.isfinite(g)].max() 
plt.clim(0, 1)
plt.colorbar()
plt.axis('off')
plt.title('G normalizado')
pylab.show()

imgplot = plt.imshow(b)
imgplot.set_cmap('RdYlGn')
#min1 = b[np.isfinite(b)].min()
#max1 = b[np.isfinite(b)].max() 
plt.clim(0, 1)
plt.colorbar()
plt.axis('off')
plt.title('B normalizado')
pylab.show()

imgplot = plt.imshow(ExG)
imgplot.set_cmap('RdYlGn') 
min2 = ExG[np.isfinite(ExG)].min()
max2 = ExG[np.isfinite(ExG)].max()
plt.clim(min2, max2)
plt.colorbar()
plt.axis('off')
plt.title('ExG')
pylab.show()

imgplot = plt.imshow(ExGR)
imgplot.set_cmap('RdYlGn') 
min3 = ExGR[np.isfinite(ExGR)].min()
max3 = ExGR[np.isfinite(ExGR)].max()
plt.clim(min3, max3)
plt.colorbar()
plt.axis('off')
plt.title('ExGR')
pylab.show()

imgplot = plt.imshow(VEG)
imgplot.set_cmap('RdYlGn')
min4 = VEG[np.isfinite(VEG)].min()
max4 = VEG[np.isfinite(VEG)].max() 
plt.clim(min4, max4)
plt.colorbar()
plt.axis('off')
plt.title('VEG')
pylab.show()

imgplot = plt.imshow(CIVE)
imgplot.set_cmap('RdYlGn') 
min5 = CIVE[np.isfinite(CIVE)].min()
max5 = CIVE[np.isfinite(CIVE)].max()
plt.clim(min5, max5)
plt.colorbar()
plt.axis('off')
plt.title('CIVE')
pylab.show()

imgplot = plt.imshow(COM)
imgplot.set_cmap('RdYlGn') 
min6 = COM[np.isfinite(COM)].min()
max6 = COM[np.isfinite(COM)].max()
plt.clim(min6, max6)
plt.colorbar()
plt.axis('off')
plt.title('COM')
pylab.show()

imgplot = plt.imshow(RGBVI)
imgplot.set_cmap('RdYlGn') 
min7 = RGBVI[np.isfinite(RGBVI)].min()
max7 = RGBVI[np.isfinite(RGBVI)].max()
plt.clim(min7, max7)
plt.colorbar()
plt.axis('off')
plt.title('RGBVI')
pylab.show()

imgplot = plt.imshow(GLI)
imgplot.set_cmap('RdYlGn') 
min8 = GLI[np.isfinite(GLI)].min()
max8 = GLI[np.isfinite(GLI)].max()
plt.clim(min8, max8)
plt.colorbar()
plt.axis('off')
plt.title('GLI')
pylab.show()

imgplot = plt.imshow(VARI)
imgplot.set_cmap('RdYlGn') 
min9 = VARI[np.isfinite(VARI)].min()
max9 = VARI[np.isfinite(VARI)].max()
plt.clim(min9, max9)
plt.colorbar()
plt.axis('off')
plt.title('VARI')
pylab.show()

imgplot = plt.imshow(MPRI)
imgplot.set_cmap('RdYlGn') 
min10 = MPRI[np.isfinite(MPRI)].min()
max10 = MPRI[np.isfinite(MPRI)].max()
plt.clim(min10, max10)
plt.colorbar()
plt.axis('off')
plt.title('MPRI')
pylab.show()

imgplot = plt.imshow(TGI)
imgplot.set_cmap('RdYlGn') 
min11 = TGI[np.isfinite(TGI)].min()
max11 = TGI[np.isfinite(TGI)].max()
plt.clim(min11, max11)
plt.colorbar()
plt.axis('off')
plt.title('TGI')
pylab.show()

imgplot = plt.imshow(RGVBI)
imgplot.set_cmap('RdYlGn')
min12 = RGVBI[np.isfinite(RGVBI)].min()
max12 = RGVBI[np.isfinite(RGVBI)].max() 
plt.clim(min12, max12)
plt.colorbar()
plt.axis('off')
plt.title('RGVBI')
pylab.show()

imgplot = plt.imshow(MGVRI)
imgplot.set_cmap('RdYlGn')
min13 = MGVRI[np.isfinite(MGVRI)].min()
max13 = MGVRI[np.isfinite(MGVRI)].max() 
plt.clim(min13, max13)
plt.colorbar()
plt.axis('off')
plt.title('MGVRI')
pylab.show()

df = pd.DataFrame({'NDVI': media0, 'NDVIc': media, 'r': media2, 'g': media3, 'b': media4, 'ExG': media5, 'ExGR': media6, 'VEG': media7, 'CIVE': media8, 'COM': media9, 'RGBVI': media10, 'GLI': media11, 'VARI': media12, 'MPRI': media13, 'TGI': media14, 'RGVBI': media15, 'MGVRI': media16})
#df = pd.DataFrame({'NDVIc': NDVIc, 'r': r, 'g': g, 'b': b, 'ExG': ExG, 'ExGR': ExGR, 'VEG': VEG, 'CIVE': CIVE, 'COM': COM, 'RGBVI': RGBVI, 'GLI': GLI, 'VARI': VARI, 'MPRI': MPRI, 'TGI': TGI, 'RGVBI': RGVBI, 'MGVRI': MGVRI})
corr = df.corr()
import seaborn as sns
#sns.heatmap(corr, xticklabels=corr.columns, cmap = "RdYlGn")

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, linewidths=.5, cmap = 'RdYlGn')