#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 29 13:40:28 2020

@author: usuario
"""

import cv2
import numpy as np
from scipy import stats

# Read image
im0 = cv2.imread("Infra2111_2019.tif").astype(np.float)
y=20
x=15
h=64
w=63
im0 = im0[y:y+h, x:x+w]
im0a = cv2.imread("Vermelho2111_2019.tif").astype(np.float)
im0a = im0a[y:y+h, x:x+w]
im0b = cv2.imread("Verde2111_2019.tif").astype(np.float)
im0b = im0b[y:y+h, x:x+w]
im0c = cv2.imread("Azul2111_2019.tif").astype(np.float)
im0c = im0c[y:y+h, x:x+w]
I0= im0[0, :, 0]
R0= im0a[0, :, 0]
G0= im0b[0, :, 0]
B0= im0c[0, :, 0]
with np.errstate(invalid='ignore'):
    NDVI = (I0 - R0) / (I0 + R0)
#NDVI = np.asarray(NDVI)

# Read image
im1 = cv2.imread("Infra2612_2019.tif").astype(np.float)

im1 = im1[y:y+h, x:x+w]
im1a = cv2.imread("Vermelho2612_2019.tif").astype(np.float)
im1a = im1a[y:y+h, x:x+w]
im1b = cv2.imread("Verde2612_2019.tif").astype(np.float)
im1b = im1b[y:y+h, x:x+w]
im1c = cv2.imread("Azul2612_2019.tif").astype(np.float)
im1c = im1c[y:y+h, x:x+w]
I1= im1[0, :, 0]
R1= im1a[0, :, 0]
G1= im1b[0, :, 0]
B1= im1c[0, :, 0]
with np.errstate(invalid='ignore'):
    NDVI1 = ((I1 - R1) / (I1 + R1))
#NDVI1 = np.asarray(NDVI1)


# Read image
im2 = cv2.imread("Infra2501_2020.tif").astype(np.float)
im2 = im2[y:y+h, x:x+w]
im2a = cv2.imread("Vermelho2501_2020.tif").astype(np.float)
im2a = im2a[y:y+h, x:x+w]
im2b = cv2.imread("Verde2501_2020.tif").astype(np.float)
im2b = im2b[y:y+h, x:x+w]
im2c = cv2.imread("Azul2501_2020.tif").astype(np.float)
im2c = im2c[y:y+h, x:x+w]
I2= im2[0, :, 0]
R2= im2a[0, :, 0]
G2= im2b[0, :, 0]
B2= im2c[0, :, 0]
with np.errstate(invalid='ignore'):
    NDVI2 = (I2 - R2) / (I2 + R2)
#NDVI2 = np.asarray(NDVI2)
  
# Read image
im3 = cv2.imread("Infra0902_2019.tif").astype(np.float)
im3 = im3[y:y+h, x:x+w]
im3a = cv2.imread("Vermelho0902_2019.tif").astype(np.float)
im3a = im3a[y:y+h, x:x+w]
im3b = cv2.imread("Verde0902_2019.tif").astype(np.float)
im3b = im3b[y:y+h, x:x+w]
im3c = cv2.imread("Azul0902_2019.tif").astype(np.float)
im3c = im3c[y:y+h, x:x+w]
I3= im3[0, :, 0]
R3= im3a[0, :, 0]
G3= im3b[0, :, 0]
B3= im3c[0, :, 0]
with np.errstate(invalid='ignore'):
    NDVI3 = (I3 - R3) / (I3 + R3)
#NDVI3 = np.asarray(NDVI3)

# Read image
im4 = cv2.imread("Infra1501_2019.tif").astype(np.float)
im4 = im4[y:y+h, x:x+w]
im4a = cv2.imread("Vermelho1501_2019.tif").astype(np.float)
im4a = im4a[y:y+h, x:x+w]
im4b = cv2.imread("Verde1501_2019.tif").astype(np.float)
im4b = im4b[y:y+h, x:x+w]
im4c = cv2.imread("Azul1501_2019.tif").astype(np.float)
im4c = im4c[y:y+h, x:x+w]
I4= im4[0, :, 0]
R4= im4a[0, :, 0]
G4= im4b[0, :, 0]
B4= im4c[0, :, 0]
with np.errstate(invalid='ignore'):
    NDVI4 = (I4 - R4) / (I4 + R4)
#NDVI3 = np.asarray(NDVI3)

R = [*R0, *R1, *R2, *R3, *R4]
G = [*G0, *G1, *G2, *G3, *G4]
B = [*B0, *B1, *B2, *B3, *B4]
I = [*NDVI, *NDVI1, *NDVI2, *NDVI3, *NDVI4]

Corr = stats.pearsonr(R, I)
print(Corr)
Corr2 = stats.pearsonr(G, I)
print(Corr2)
Corr3 = stats.pearsonr(B, I)
print(Corr3)

import statsmodels.api as sm

x = np.column_stack((R, G, B))  #Agrupa as variaveis preditorass

x = sm.add_constant(x, prepend=True) #Adiciona a coluna das constantes

res = sm.OLS(I,x).fit() #Cria e ajusta o modelo

#print(res.params)

#print(res.bse)

print(res.summary())

X_sm = sm.add_constant(R)# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo
results = sm.OLS(I, X_sm).fit()# mostrando as estatísticas do modelo
results.summary()# mostrando as previsões para o mesmo conjunto passado
results.predict(X_sm)
print(results.summary())

X_sm = sm.add_constant(G)# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo
results = sm.OLS(I, X_sm).fit()# mostrando as estatísticas do modelo
results.summary()# mostrando as previsões para o mesmo conjunto passado
results.predict(X_sm)
print(results.summary())

X_sm = sm.add_constant(B)# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo
results = sm.OLS(I, X_sm).fit()# mostrando as estatísticas do modelo
results.summary()# mostrando as previsões para o mesmo conjunto passado
results.predict(X_sm)
print(results.summary())
