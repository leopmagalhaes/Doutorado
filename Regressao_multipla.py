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
I0a= im0[0, :, 0]
I0b= im0[13, :, 0]
I0c= im0[26, :, 0]
I0d= im0[39, :, 0]
I0e= im0[52, :, 0]
I0 = [*I0a, *I0b, *I0c, *I0d, *I0e]
I0 = np.array(I0)
#R3= im3a[0, :, 0]
R3a= im0a[0, :, 0]
R3b= im0a[13, :, 0]
R3c= im0a[26, :, 0]
R3d= im0a[39, :, 0]
R3e= im0a[52, :, 0]
R0 = [*R3a, *R3b, *R3c, *R3d, *R3e]
R0 = np.array(R0)
#G3= im3b[0, :, 0]
G3a= im0b[0, :, 0]
G3b= im0b[13, :, 0]
G3c= im0b[26, :, 0]
G3d= im0b[39, :, 0]
G3e= im0b[52, :, 0]
G0 = [*G3a, *G3b, *G3c, *G3d, *G3e]
G0 = np.array(G0)
#B3= im3c[0, :, 0]
B3a= im0c[0, :, 0]
B3b= im0c[13, :, 0]
B3c= im0c[26, :, 0]
B3d= im0c[39, :, 0]
B3e= im0c[52, :, 0]
B0 = [*B3a, *B3b, *B3c, *B3d, *B3e]
B0 = np.array(B0)
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
I0a= im1[0, :, 0]
I0b= im1[13, :, 0]
I0c= im1[26, :, 0]
I0d= im1[39, :, 0]
I0e= im1[52, :, 0]
I1 = [*I0a, *I0b, *I0c, *I0d, *I0e]
I1 = np.array(I1)
#R3= im3a[0, :, 0]
R3a= im1a[0, :, 0]
R3b= im1a[13, :, 0]
R3c= im1a[26, :, 0]
R3d= im1a[39, :, 0]
R3e= im1a[52, :, 0]
R1 = [*R3a, *R3b, *R3c, *R3d, *R3e]
R1 = np.array(R1)
#G3= im3b[0, :, 0]
G3a= im0b[0, :, 0]
G3b= im0b[13, :, 0]
G3c= im0b[26, :, 0]
G3d= im0b[39, :, 0]
G3e= im0b[52, :, 0]
G1 = [*G3a, *G3b, *G3c, *G3d, *G3e]
G1 = np.array(G1)
#B3= im3c[0, :, 0]
B3a= im0c[0, :, 0]
B3b= im0c[13, :, 0]
B3c= im0c[26, :, 0]
B3d= im0c[39, :, 0]
B3e= im0c[52, :, 0]
B1 = [*B3a, *B3b, *B3c, *B3d, *B3e]
B1 = np.array(B1)
with np.errstate(invalid='ignore'):
    NDVI1 = (I1 - R1) / (I1 + R1)
#NDVI = np.asarray(NDVI)


# Read image
im2 = cv2.imread("Infra2501_2020.tif").astype(np.float)
im2 = im2[y:y+h, x:x+w]
im2a = cv2.imread("Vermelho2501_2020.tif").astype(np.float)
im2a = im2a[y:y+h, x:x+w]
im2b = cv2.imread("Verde2501_2020.tif").astype(np.float)
im2b = im2b[y:y+h, x:x+w]
im2c = cv2.imread("Azul2501_2020.tif").astype(np.float)
im2c = im2c[y:y+h, x:x+w]
I0a= im2[0, :, 0]
I0b= im2[13, :, 0]
I0c= im2[26, :, 0]
I0d= im2[39, :, 0]
I0e= im2[52, :, 0]
I2 = [*I0a, *I0b, *I0c, *I0d, *I0e]
I2 = np.array(I2)
#R3= im3a[0, :, 0]
R3a= im2a[0, :, 0]
R3b= im2a[13, :, 0]
R3c= im2a[26, :, 0]
R3d= im2a[39, :, 0]
R3e= im2a[52, :, 0]
R2 = [*R3a, *R3b, *R3c, *R3d, *R3e]
R2 = np.array(R2)
#G3= im3b[0, :, 0]
G3a= im2b[0, :, 0]
G3b= im2b[13, :, 0]
G3c= im2b[26, :, 0]
G3d= im2b[39, :, 0]
G3e= im2b[52, :, 0]
G2 = [*G3a, *G3b, *G3c, *G3d, *G3e]
G2 = np.array(G2)
#B3= im3c[0, :, 0]
B3a= im2c[0, :, 0]
B3b= im2c[13, :, 0]
B3c= im2c[26, :, 0]
B3d= im2c[39, :, 0]
B3e= im2c[52, :, 0]
B2 = [*B3a, *B3b, *B3c, *B3d, *B3e]
B2 = np.array(B2)
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
I0a= im3[0, :, 0]
I0b= im3[13, :, 0]
I0c= im3[26, :, 0]
I0d= im3[39, :, 0]
I0e= im3[52, :, 0]
I3 = [*I0a, *I0b, *I0c, *I0d, *I0e]
I3 = np.array(I3)
#R3= im3a[0, :, 0]
R3a= im3a[0, :, 0]
R3b= im3a[13, :, 0]
R3c= im3a[26, :, 0]
R3d= im3a[39, :, 0]
R3e= im3a[52, :, 0]
R3 = [*R3a, *R3b, *R3c, *R3d, *R3e]
R3 = np.array(R3)
#G3= im3b[0, :, 0]
G3a= im3b[0, :, 0]
G3b= im3b[13, :, 0]
G3c= im3b[26, :, 0]
G3d= im3b[39, :, 0]
G3e= im3b[52, :, 0]
G3 = [*G3a, *G3b, *G3c, *G3d, *G3e]
G3 = np.array(G3)
#B3= im3c[0, :, 0]
B3a= im3c[0, :, 0]
B3b= im3c[13, :, 0]
B3c= im3c[26, :, 0]
B3d= im3c[39, :, 0]
B3e= im3c[52, :, 0]
B3 = [*B3a, *B3b, *B3c, *B3d, *B3e]
B3 = np.array(B3)
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
I0a= im4[0, :, 0]
I0b= im4[13, :, 0]
I0c= im4[26, :, 0]
I0d= im4[39, :, 0]
I0e= im4[52, :, 0]
I4 = [*I0a, *I0b, *I0c, *I0d, *I0e]
I4 = np.array(I4)
#R3= im3a[0, :, 0]
R3a= im4a[0, :, 0]
R3b= im4a[13, :, 0]
R3c= im4a[26, :, 0]
R3d= im4a[39, :, 0]
R3e= im4a[52, :, 0]
R4 = [*R3a, *R3b, *R3c, *R3d, *R3e]
R4 = np.array(R4)
#G3= im3b[0, :, 0]
G3a= im4b[0, :, 0]
G3b= im4b[13, :, 0]
G3c= im4b[26, :, 0]
G3d= im4b[39, :, 0]
G3e= im4b[52, :, 0]
G4 = [*G3a, *G3b, *G3c, *G3d, *G3e]
G4 = np.array(G4)
#B3= im3c[0, :, 0]
B3a= im4c[0, :, 0]
B3b= im4c[13, :, 0]
B3c= im4c[26, :, 0]
B3d= im4c[39, :, 0]
B3e= im4c[52, :, 0]
B4 = [*B3a, *B3b, *B3c, *B3d, *B3e]
B4 = np.array(B4)
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
