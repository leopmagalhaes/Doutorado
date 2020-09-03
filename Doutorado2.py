#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:37:20 2020

@author: usuario
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import math


def MLf(z, a):
    """Mittag-Leffler function
    """
    k = np.arange(100).reshape(-1, 1)
    E = ((7.9336*z)**k) / gamma(a*k + 1)
    return 0.0211*(np.sum(E, axis=0))

x = np.arange(0.340, 0.341, 0.01)
f = (0.0211*np.exp(7.9336*x))
#plt.plot(x, f, label="alpha = 1" )
#print("Max com alfa igual a", 1, "vale: ", max(f))
#print("Min com alfa igual a", 1, "vale: ", min(f))
#EM1 = ((min(f)-1.26)+(max(f)-1.86))/2
#print("Erro medio com alfa igual a 1 vale: ", EM1)
#print(" ")
#y = (x/0.6869)**(1/0.1809)
#print("Max pelo modelo vale: ", max(y))
#print("Min pelo modeko vale: ", min(y))
#plt.plot(x, y)
#plt.title('Modelo da tese')
#plt.figure(figsize=(10,5))

for i in np.arange(2,3,0.1):
    plt.plot(x, MLf(x, i/2), label="alpha = "+str(round(i/2,3)))
    print("Max com alfa igual a: ", round(i/2,3), "vale: ", round(max(MLf(x, i/2)),3), "e o min vale: ", round(min(MLf(x, i/2)),3))
    #EM1 = np.sum(((MLf(x, i/2))-y))/10
    #EM2 = np.sum(((MLf(x, i/2))-y)**2)/10
    #d1 = np.sum(((abs(MLf(x, i/2)-np.mean(y)))+(abs((y-np.mean(y)))))**2)
    #d = 1 - (EM2)/d1
    #print("Erro medio: ", round(EM1,3))
    #print("Erro quadrado medio: ", round(EM2,3))
    #print("O indice de concordancia de Willmott (d): ", round(d,3))
    #correlation_matrix = np.corrcoef(MLf(x,i/2), y)
    #correlation_xy = correlation_matrix[0,1]
    #r_squared = correlation_xy**2
    #r_ajustado = 1 - (((10-1)/(10-2))*(1-r_squared))
    #raiz = math.sqrt(r_squared)
    #c = raiz*d
    #print("O indice de confiança de Camargo (c): ", round(c,3))
    #print("O valor do R2 é: ", round(r_squared,4))
    #print("O valor do R2 ajustado é: ", round(r_ajustado,4))
    #print(" ")
plt.legend()
#plt.ylim(-5, 5); plt.xlim(-55, 15); plt.grid()