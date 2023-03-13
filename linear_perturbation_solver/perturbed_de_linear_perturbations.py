# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:06:11 2023

@author: Sebastiano Tomasi
"""
import numpy as np
from numpy import sqrt
import scipy as sp

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func



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

def linear_density_contrast_eq_c_eff(a,y):
    """This function defines the LINEAR matter density contrast differential equation for clustering dark energy.
    y[0]=\delta_m
    y[1]=\theta 
    y[2]=\delta_de
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\theta'
    fun[2]=y'[2]=\delta_de'
    """
    fun=[0]*3
    fun[0] = -y[1]/a
    fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-3/(2*a)*(omega_matter_a(a)*y[0]+omega_dark_a(a)*y[2]*(1+3*params.c_eff(a)))
    fun[2] = -3/a*(params.c_eff(a)-params.de_eos_a(a))*y[2]-(1+params.de_eos_a(a))*y[1]/a
    return np.array(fun)



def linear_density_contrast_eq(a,y):
    """This function defines the LINEAR matter density contrast 
    differential equation whene dark energy perturbations are set to zero,
    but the effects of dark energy are still considered in the background.
    y[0]=\delta_m
    y[1]=\delta_m' 
    y[2]=\delta_de
    y[3]=\delta_de'
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\delta_m''
    fun[2]=y'[2]=\delta_de'
    fun[3]=y'[3]=\delta_de''
    """
    fun=[0]*4
    fun[0] = y[1]
    fun[1] = -3/(2*a)*(1-effective_eos_a(a))*y[1] \
        +3/(2*a**2)*(omega_matter_a(a)*y[0]+omega_dark_a(a)*y[2]*(1+3*params.de_eos_a(a)))
    fun[2]=y[3]
    fun[3]=-(3/(2*a)*(1-effective_eos_a(a))-cosm_func.de_eos_derivative_a(a)/(1+params.de_eos_a(a)))*y[3]\
        +3/(2*a**2)*(1+params.de_eos_a(a))*(omega_matter_a(a)*y[0]+omega_dark_a(a)*y[2]*(1+3*params.de_eos_a(a)))
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
    effective_eos_a=sp.interpolate.interp1d(friedmann_sol.effective_eos_numerical_a[0], 
                                            friedmann_sol.effective_eos_numerical_a[1],
                                       fill_value="extrapolate", assume_sorted=True)
    global omega_matter_a
    omega_matter_a=sp.interpolate.interp1d(friedmann_sol.matter_density_parameter_numerical_a[0], 
                                           friedmann_sol.matter_density_parameter_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    global omega_dark_a
    omega_dark_a=sp.interpolate.interp1d(friedmann_sol.dark_density_parameter_numerical_a[0], 
                                         friedmann_sol.dark_density_parameter_numerical_a[1],
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
    delta_de_tls=(1+params.de_eos_a(a_min))*delta_m_tls
    d_delta_de_tls=n*delta_de_tls/a_min


    init_cond=[delta_m_tls,d_delta_m_tls,delta_de_tls,d_delta_de_tls]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(linear_density_contrast_eq,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    linear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    linear_de_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[2]])
    return [linear_matter_density_contrast_numerical_a,linear_de_density_contrast_numerical_a]

def solve_c_eff(friedmann_sol=None):
    global effective_eos_a
    effective_eos_a=sp.interpolate.interp1d(friedmann_sol.effective_eos_numerical_a[0], 
                                            friedmann_sol.effective_eos_numerical_a[1],
                                       fill_value="extrapolate", assume_sorted=True)
    global omega_matter_a
    omega_matter_a=sp.interpolate.interp1d(friedmann_sol.matter_density_parameter_numerical_a[0], 
                                           friedmann_sol.matter_density_parameter_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    global omega_dark_a
    omega_dark_a=sp.interpolate.interp1d(friedmann_sol.dark_density_parameter_numerical_a[0], 
                                         friedmann_sol.dark_density_parameter_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    
    """Defining the integration parameters"""
    atol=1e-9
    rtol=1e-8

    a_min= params.scale_at_lss
    a_max= params.a_max
    n=round(0.25*(-1+sqrt(24*omega_matter_a(a_min)+(1-3*effective_eos_a(a_min))**2)+3*effective_eos_a(a_min)),4)

    delta_m_ini=a_min**n#Initial perturbation value
    theta_ini=-n*delta_m_ini#Initial value for theta
    delta_de_ini=n*(1+params.de_eos_a(a_min))*delta_m_ini/(n+3*(params.c_eff(a_min)-params.de_eos_a(a_min)))#Initial DE perturbation first derivative w.r.t scale parameter value
    
    init_cond=[delta_m_ini,theta_ini,delta_de_ini]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(linear_density_contrast_eq_c_eff,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    linear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    linear_de_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[2]])
    
    return [linear_matter_density_contrast_numerical_a,linear_de_density_contrast_numerical_a]




