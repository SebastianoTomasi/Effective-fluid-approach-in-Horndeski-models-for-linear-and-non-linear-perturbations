# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:42:50 2022

@author: sebas
"""
import numpy as np
from numpy import log
from numpy import log10
from numpy import exp
from numpy import tanh



def lcdm0(z,w_i=-1,w_f=None,z_t=None,gamma=None):
    return w_i
lcdm=np.vectorize(lcdm0)

    
def cpl(z,w_i=0,w_f=-1,z_t=2,gamma=10):
    cpl= w_f+(z/(1+z))*w_i
    return cpl

    
def de_eos_1(z,w_i=-0.0,w_f=-1,z_t=0.5,gamma=10):
    """delta_z=2*(1+z_t)*sinh(2/gamma)"""
    de_eos_1=0.5*(w_i+w_f)-0.5*(w_i-w_f)*tanh(gamma*log((1+z_t)/(1+z)))
    return de_eos_1



def de_eos_2(z,w_i=-0.2,w_f=-1,z_t=1,gamma=10):
    q=gamma
    de_eos_2=w_f+(w_i-w_f)*(z/z_t)**q/(1+(z/z_t)**q)
    return de_eos_2

    
def de_eos_3(z,w_i=-0.4,w_f=-1,z_t=2,gamma=0.1):
    delta=gamma
    de_eos_3=w_i+(w_f-w_i)/(1+exp((z-z_t)/delta))
    return de_eos_3    


def de_eos_4(z,w_i=-0.6,w_f=-1,z_t=3,gamma=0.015):
    a=1/(z+1)
    a_m=1/(z_t+1)
    delta_m=gamma
    de_eos_4z=w_f+(w_i-w_f)*(1+exp(a_m/delta_m))/(1+exp((-a+a_m)/delta_m))*(1-exp((-a+1)/delta_m))/(1-exp(1/delta_m))
    return de_eos_4z

    
def de_eos_5(z,w_i=None,w_f=None,z_t=0.8,gamma=1.7):
    de_eos_5=-gamma/(3*log(10))*(1+tanh(gamma*log10((1+z)/(1+z_t))))-1
    return de_eos_5

def de_eos_6(z,w_i=None,w_f=None,z_t=None,gamma=None):
    de_eos_6=-1/(3*log(10))*(1+tanh(log10(1+z)))-1
    return de_eos_6

    
    