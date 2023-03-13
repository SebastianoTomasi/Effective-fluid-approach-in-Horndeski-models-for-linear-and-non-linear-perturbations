# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:32:54 2023

@author: Sebastiano Tomasi
"""
import numpy as np
from numpy import sqrt

import scipy as sp
import sys

sys.path.append("../data_modules")
from data_classes import friedmann_sol
import cosmological_functions as cosm_func
import simulation_parameters as params

sys.path.append("../utility_modules")
import numerical_methods as mynm



#%%
"""Those functions must be defined here since dark_density_evolution_a(a) is computed here."""
def friedmann_equation_integrand(a):
    friedmann_equation_integrand = 1/(a*sqrt(cosm_func.matter_density_evolution_a(a) + dark_density_evolution_a(a)+cosm_func.rad_density_evolution_a(a)))
    return friedmann_equation_integrand


def rescaled_hubble_function_a(a):#Usually denoted by E(a)=(H/H0)
    rescaled_hubble_function_a=sqrt(cosm_func.matter_density_evolution_a(a) + dark_density_evolution_a(a)+cosm_func.rad_density_evolution_a(a))
    return rescaled_hubble_function_a

#%%


"""Create the class object to store the results"""
result=friedmann_sol()

def solve(time_domain=False):
    """This function solve for the background equations, given the parameters in the simulation_parameters module.
        input:
            - time_domain if True, computes all the bakground quantities also as function of time. """
    
    """Compute the dark density evolution"""
    dark_density_evolution_numerical_a=cosm_func.solve_dark_density_evolution_numerical_a()
    scale_parameter_values=dark_density_evolution_numerical_a[0]
    global dark_density_evolution_a #Must be global, or the the function defined outside solve() will not see it.
    dark_density_evolution_a=sp.interpolate.interp1d(scale_parameter_values, dark_density_evolution_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=False)
    
    """Compute also the de_eos numerically in order to plot it."""
    de_eos_numerical_a=np.array([scale_parameter_values,params.de_eos_a(scale_parameter_values)])
    
    """Effective eos"""
    result.effective_eos_numerical_a=[scale_parameter_values,
                                      (params.de_eos_a(scale_parameter_values)*dark_density_evolution_numerical_a[1]+  \
                                       cosm_func.rad_density_evolution_a(scale_parameter_values)/3)/    \
                                          (cosm_func.matter_density_evolution_a(scale_parameter_values)+   \
                                         dark_density_evolution_numerical_a[1]+   \
                                        cosm_func.rad_density_evolution_a(scale_parameter_values))]
    
    """Approximate dark density evolution using a step transition."""
    aux=[]
    for i in scale_parameter_values:
        aux.append(cosm_func.step_dark_density_evolution_a(i))
    appox_dark_density_evolution_numerical_a=np.array([scale_parameter_values,aux])
    
    """Rescaled hubble function H/H0, we need it to compute the density parameters"""
    rescaled_hubble_functions_numerical_a=np.array([scale_parameter_values,
                                          rescaled_hubble_function_a(scale_parameter_values)])
    
    dark_density_parameter_numerical_a=np.array([scale_parameter_values,
              dark_density_evolution_a(scale_parameter_values)/rescaled_hubble_functions_numerical_a[1]**2])
    matter_density_parameter_numerical_a=np.array([scale_parameter_values,
              cosm_func.matter_density_evolution_a(scale_parameter_values)/rescaled_hubble_functions_numerical_a[1]**2])
    rad_density_numerical_a=np.array([scale_parameter_values,
              cosm_func.rad_density_evolution_a(scale_parameter_values)/rescaled_hubble_functions_numerical_a[1]**2])
    
    """Saving result in the Friedmann results class"""
    result.dark_density_evolution_numerical_a=dark_density_evolution_numerical_a
    result.appox_dark_density_evolution_numerical_a=appox_dark_density_evolution_numerical_a
    result.rescaled_hubble_functions_numerical_a=rescaled_hubble_functions_numerical_a
    result.dark_density_parameter_numerical_a=dark_density_parameter_numerical_a
    result.matter_density_parameter_numerical_a=matter_density_parameter_numerical_a
    result.rad_density_numerical_a=rad_density_numerical_a
    result.de_eos_numerical_a=de_eos_numerical_a
    
    if time_domain:
        """Integrate the friedmann equation"""
        hubble_constant_times_t = mynm.integrate(f=friedmann_equation_integrand,a=params.a_max, b=params.a_min, n=params.n)
        
    
        
        """Compute t(a)==time_a"""
        time_minus_universe_age=hubble_constant_times_t[1]/params.hubble_constant
        universe_age = -time_minus_universe_age[0]#In giga yeras
        time=time_minus_universe_age+universe_age
        
        """Compare the age of the universe of the model to LCDM"""
        # lcdm_universe_age= cosm_func.time_a_LCDM(1)
        # print("MODEL: t0=",round(universe_age,3)," Gy     LCDM: t0=",round(lcdm_universe_age,3)," Gy")
        
        """Invert t(a) to obtain a(t). In more detail we have a(t) s.t a(t0)=1 
        where t_0 is the age of the universe"""
        scale_parameter_numerical_t=np.array([time,scale_parameter_values])
        
        """Compute the hubble function H(t)=\dot{a}/a and the second derivative of a"""
        scale_parameter_derivative=mynm.Nderivate(scale_parameter_numerical_t)
        hubble_function_numerical_t=np.array([time,scale_parameter_derivative[1]/scale_parameter_values])
        
        """We can now calculate the deceleration parameter today: q_0"""
        scale_parameter_2derivative=mynm.Nderivate(scale_parameter_derivative)
        dec_param_now=-scale_parameter_2derivative[1][-1]/params.hubble_constant**2
        # print("MODEL: q0 = ",round(dec_param_now,3),"""    LCDM: q0 = """,round(omega_matter_now/2-omega_dark_now,3))
        result.scale_parameter_numerical_t=scale_parameter_numerical_t
        result.hubble_function_numerical_t=hubble_function_numerical_t
        result.universe_age=universe_age
        result.deceleration_parameter=dec_param_now

    
    return result









