#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 16:03:14 2019

@author: usuario
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab

im = cv2.imread("60m_area.jpg").astype(np.float)


MPRI = ((im[:, :, 1] - im[:, :, 2])/(im[:, :, 1]+ im[:, :, 2]))
minMPRI = MPRI[np.isfinite(MPRI)].min()
maxMPRI = MPRI[np.isfinite(MPRI)].max()

print('Maximo =', maxMPRI)
print('Minimo =', minMPRI)
print('MPRI =', MPRI[np.isfinite(MPRI)].mean())

Gn = ((im[:, :, 1])/(im[:, :, 1]+ im[:, :, 2]+im[:, :, 0]))
minGn = Gn[np.isfinite(Gn)].min()
maxGn = Gn[np.isfinite(Gn)].max()

print('Maximo =', maxGn)
print('Minimo =', minGn)
print('Gn =', Gn[np.isfinite(Gn)].mean())

Rn = ((im[:, :, 2])/(im[:, :, 1]+ im[:, :, 2]+im[:, :, 0]))
minRn = Rn[np.isfinite(Rn)].min()
maxRn = Rn[np.isfinite(Rn)].max()

print('Maximo =', maxRn)
print('Minimo =', minRn)
print('Rn =', Rn[np.isfinite(Rn)].mean())

Bn = ((im[:, :, 0])/(im[:, :, 1]+ im[:, :, 2]+im[:, :, 0]))
minBn = Bn[np.isfinite(Bn)].min()
maxBn = Bn[np.isfinite(Bn)].max()

print('Maximo =', maxBn)
print('Minimo =', minBn)
print('Rn =', Bn[np.isfinite(Bn)].mean())

imgplot = plt.imshow(MPRI)
imgplot.set_cmap('RdYlGn') 
plt.clim(-0.5, 0.5)
plt.colorbar()
plt.axis('off')
pylab.show()

pylab.show()

t=np.linspace(minMPRI, maxMPRI, 10)
t2=np.linspace(5.381, 8.779, 10)

EQ1 = (t - 0.28) / 0.18
FC = 1- ((0.795 - t)/(0.795 - 0.687))
EQ2 = -2 * (np.log((1 - FC)))
EQ3= 0.0211 * np.exp(7.9336* t)
EQ4 = 4.546 * (t ** 3.496) 
EM1 = ((min(EQ1)-1.26)+(max(EQ1)-1.86))/2
print("Erro medio da EQ1 vale: ", EM1)
EM2 = ((min(EQ2)-1.26)+(max(EQ2)-1.86))/2
print("Erro medio da EQ2 vale: ", EM2)
EM3 = ((min(EQ3)-1.26)+(max(EQ3)-1.86))/2
print("Erro medio da EQ3 vale: ", EM3)
EM4 = ((min(EQ4)-1.26)+(max(EQ4)-1.86))/2
print("Erro medio da EQ2 vale: ", EM4)
EQM1 = EM1 ** 2
print("Erro quadrado medio da EQ1 vale: ", EQM1)
EQM2 = EM2 ** 2
print("Erro quadrado medio da EQ2 vale: ", EQM2)
EQM3 = EM3 ** 2
print("Erro quadrado medio da EQ3 vale: ", EQM3)
EQM4 = EM4 ** 2
print("Erro quadrado medio da EQ4 vale: ", EQM4)


plt.plot(t, EQ1)
plt.show()
plt.plot(t, EQ2)
plt.show()
plt.plot(t, EQ3)
plt.show()
plt.plot(t, EQ4)
plt.show()
cv2.waitKey(0)

cv2.destroyAllWindows()