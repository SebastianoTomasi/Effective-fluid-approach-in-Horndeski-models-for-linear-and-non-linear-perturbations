# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:34:51 2022

@author: Sebastiano Tomasi
"""
import numpy as np
from numpy import sqrt
import scipy as sp

import sys
sys.path.append("../data_modules")
import simulation_parameters as params



#%%
"""Solving the density contrast perturbation equation for matter 

In our case the differential equation is:
x'' + a(t)x' + b(t)x = c(t)
wich can be rewritten as a first order system:
y=[x,x']
y'[0]=f(t,y)[0]= y[1]
y'[1]=f(t,y)[1]=c(t)-a(t)*y[1]-b(t)*y[0]

Compute the coefficients of the pert eq. in term of a
Define the w_eff(a)=effective_eos_a for a universe with matter and DE"""


#%%
"""Definition of the differential equation:"""
def linear_density_contrast_eq(a,y):
    """This function defines the LINEAR matter density contrast 
    differential equation whene dark energy perturbations are set to zero,
    but the effects of dark energy are still considered in the background.
    y[0]=\delta_m
    y[1]=\delta_m' 
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\delta_m''
    """
    fun=[0]*2
    fun[0] = y[1]
    fun[1] = -3/(2*a)*(1-effective_eos_a(a))*y[1] \
        +3/(2*a**2)*omega_matter_a(a)*y[0]
    return np.array(fun)
#%%
"""Integrator"""
def solve(friedmann_sol=None):
    """Solves the linear matter density contrast differential equation:
        input:
            - friedmann_sol result of the friedmann solver
        output:
            - [[a_1,..,a_n],[delta_1,...,delta_n]]"""
    """Interpolate the friedmann results """
    global effective_eos_a
    effective_eos_a=sp.interpolate.interp1d(friedmann_sol.effective_eos_numerical_a[0], friedmann_sol.effective_eos_numerical_a[1],
                                       fill_value="extrapolate", assume_sorted=True)
    global omega_matter_a
    omega_matter_a=sp.interpolate.interp1d(friedmann_sol.matter_density_parameter_numerical_a[0], friedmann_sol.matter_density_parameter_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)


    """Defining the integration parameters"""
    atol=1e-9
    rtol=1e-8
    a_min= params.scale_at_lss#We start at last scattering 
    a_max= params.a_max #Use the a_max from the other numerical methods
    
    """Compute the exponent of the growing mode at last scattering"""
    n=round(0.25*(-1+3*effective_eos_a(a_min)+sqrt((1-3*effective_eos_a(a_min))**2+24*omega_matter_a(a_min))),3)

    delta_m_tls=(a_min)**n
    d_delta_m_tls=n*delta_m_tls/a_min


    init_cond=[delta_m_tls,d_delta_m_tls]# [Position(t=t_min),Speed(t_tmin)]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(linear_density_contrast_eq,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    linear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    return linear_matter_density_contrast_numerical_a






