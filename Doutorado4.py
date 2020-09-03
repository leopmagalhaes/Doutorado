#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:00:26 2020

@author: usuario
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

a = float(input('Digite valor de a: '))
b = float(input('Digite valor de b: '))
c = float(input('Digite valor de c: '))

t = np.arange(0.336, 0.35, 0.01)

y = a*(t**2)-(b*t)+c

EM1 = max(y)
print("O max vale: ", EM1)
plt.figure(figsize=(10,5))
plt.plot(t, y)
plt.show()

plt.figure(figsize=(10,5))

for i in range(1, 5):
    alfa = i/3
    d = gamma(4)
    e = gamma(3 - alfa + 1)
    r = ((d/e)/3) * (t**(3-alfa))
    d1 = gamma(3)
    e1 = gamma(2 - alfa + 1)
    r1 = ((d1/e1)/2) * (t**(2-alfa))
    d2 = gamma(2)
    e2 = gamma(1 - alfa + 1)
    r2 = ((d2/e2)/1) * (t**(1-alfa))
    y1 = (a*r) - (b*r1) + (c*r2)
    EM2 = max(y1)
    print("O max vale: ", EM2)
    plt.plot(t, y1, label="alpha = "+str(i/3))
plt.legend()
plt.ylim(20, 160); plt.xlim(0.336, 0.35)