# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:54 2023

@author: sebas
"""


import numpy as np
from numpy import log
from numpy import sqrt
from numpy import exp

import simulation_parameters as params

import sys
sys.path.append("../utility_modules")
import numerical_methods as mynm





#%%
"""Matter density parameter in function of a"""
def matter_density_evolution_a(a):
    matter_density_evolution_a= params.omega_matter_now/a**3
    return matter_density_evolution_a

"""Radiation density parameter in function of a"""
def rad_density_evolution_a(a):
    rad_density_evolution_a= params.omega_rad_now/a**4
    return rad_density_evolution_a

"""Finds the function g(a) for the dark energy."""
def solve_dark_density_evolution_numerical_a():
    """Definition of the integrand in the exponent."""
    def de_exponent(a):
        de_exponent = (1+params.de_eos_a(a))/a
        return de_exponent
    """Compute dark_density_evolution_a(a)=omega_dark_now*g(a) """
    de_exponent_integral=mynm.integrate(f=de_exponent,a=params.a_max,b=params.a_min,n=params.n,Fa=0)
    scale_parameter_values=de_exponent_integral[0]
    dark_density_evolution_numerical_a=np.array([scale_parameter_values,params.omega_dark_now*exp(-3*de_exponent_integral[1])])
    
    return dark_density_evolution_numerical_a



"""Derivative of the dark energy eos"""
def de_eos_derivative_a(a):
    de_eos_derivative_a=mynm.derivate(params.de_eos_a, 1/params.n, a)
    return de_eos_derivative_a


"""Approximation for the dark density evolution """
def step_eos(a):#Step eos
    a_t=1/(1+params.trans_z)
    if a < a_t:
        return params.w_i
    else:
        return params.w_f
    
def step_dark_density_evolution_a(a):
    a_t=1/(1+params.trans_z)
    if a_t>1:
        raise Exception("trans_z must be a positive number")
    if a>=a_t:
        return params.omega_dark_now*a**(-3*(1+params.w_f))
    if a<a_t:
        return  params.omega_dark_now*a**(-3*(1+params.w_i))*a_t**(-3*(params.w_f-params.w_i))
    


def time_a_LCDM(a):#Age of the universe in LCDM
    r=params.omega_dark_now/params.omega_matter_now
    time_a_LCDM=-log(-2*a**(3/2)*sqrt(r)*sqrt(a**3*r+1)+ 2*a**3*r+1)/(3*sqrt(params.omega_dark_now))
    return time_a_LCDM/params.hubble_constant

"""Conversion between redshift <-> scale parameter"""
def z_a(a):
    z_a=1/a-1
    return z_a
def a_z(z):
    a_z=1/(1+z)
    return a_z