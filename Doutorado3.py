#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:41:39 2020

@author: usuario
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

a = float(input('Digite valor de a: '))
b = float(input('Digite valor de b: '))
c = float(input('Digite valor de c: '))

t = np.arange(0, 36, 1)

gompertz_int_1 = np.exp(-c*t)
gompertz_int_2 = -np.log(b)*gompertz_int_1
gompertz_int_3 = a*np.exp(gompertz_int_2)
gompertz_int = gompertz_int_3
EM1 = max(gompertz_int)
print("O max vale: ", EM1)
plt.figure(figsize=(10,5))
plt.plot(t, gompertz_int)
plt.show()


plt.figure(figsize=(10,5))

for i in range(1, 5):
    alfa = i/3
    k = np.arange(100)
    d = -c*(np.power(t,alfa))
    gompertz_frac_1 = (np.polynomial.polynomial.polyval(d, 1/gamma(alfa*k + 1)))
    gompertz_frac_2 = -np.log(b)*gompertz_frac_1
    gompertz_frac_3 = a*np.exp(gompertz_frac_2)
    gompertz_frac = gompertz_frac_3
    EM2 = max(gompertz_frac)
    print("O max vale: ", EM2)
    plt.plot(t, gompertz_frac, label="alpha = "+str(i/3))
plt.legend()
plt.ylim(2, 16); plt.xlim(0, 10)
