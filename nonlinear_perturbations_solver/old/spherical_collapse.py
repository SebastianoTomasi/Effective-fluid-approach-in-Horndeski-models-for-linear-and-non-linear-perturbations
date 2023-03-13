# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:51:08 2023

@author: sebas
"""
import numpy as np
from numpy import sin
from numpy import cos
from numpy import log
from numpy import exp
from numpy import e
from numpy import pi
from numpy import sqrt
from numpy import sinh
from numpy import cosh
from numpy import arcsinh
from numpy import arccosh

import scipy as sp
import matplotlib.pyplot as pl

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func

sys.path.append("../utility_modules")
import numerical_methods as mynm
import plotting_functions as mypl

sys.path.append("../friedmann_solver")
import friedmann_solver as fr_sol



#%%
# friedmann_sol=fr_sol.solve()

#%%


def rescaled_hubble_function_a(a):#Usually denoted by E(a)=(H/H0)
    rescaled_hubble_function_a=sqrt(cosm_func.matter_density_evolution_a(a) + dark_density_evolution_a(a)+cosm_func.rad_density_evolution_a(a))
    return rescaled_hubble_function_a

def matter_density_parameter(a):
    matter_density_parameter=cosm_func.matter_density_evolution_a(a)/(rescaled_hubble_function_a(a)**2)
    return matter_density_parameter

def dark_density_parameter(a):
    dark_density_parameter=dark_density_evolution_a(a)/(rescaled_hubble_function_a(a)**2)
    return dark_density_parameter



"""Definition of the ODE system to solve: """
def fun(a,y):
    fun=[0]*4
    """Matter perturbations"""
    fun[0] = y[1]
    fun[1] = -3/(2*a)*(1-effective_eos_a(a))*y[1] + 4/3*y[1]**2/(1+y[0]) \
        +3/(2*a**2)*(1+y[0])*(matter_density_parameter(a)*y[0]+dark_density_parameter(a)*y[2]*(1+3*params.de_eos_a(a)))
       
    """DE perturbations"""    
    fun[2]=y[3]
    fun[3]=-(3/(2*a)*(1-effective_eos_a(a))-cosm_func.de_eos_derivative_a(a)/(1+params.de_eos_a(a)))*y[3]\
        +(4+3*params.de_eos_a(a))/(3*(1+params.de_eos_a(a)))*y[3]**2/(1+y[2])\
        +3/(2*a**2)*(1+params.de_eos_a(a))*(1+y[2])*(matter_density_parameter(a)*y[0]+dark_density_parameter(a)*y[2]*(1+3*params.de_eos_a(a)))
    return np.array(fun)


#%%
def solve(friedmann_sol=None):        
    global effective_eos_a
    effective_eos_numerical_a=friedmann_sol.effective_eos_numerical_a
    effective_eos_a=sp.interpolate.interp1d(effective_eos_numerical_a[0], effective_eos_numerical_a[1],
                                            fill_value="extrapolate", assume_sorted=True)
    
    global dark_density_evolution_a
    dark_density_evolution_a=sp.interpolate.interp1d(friedmann_sol.dark_density_evolution_numerical_a[0], friedmann_sol.dark_density_evolution_numerical_a[1],
                                                     fill_value="extrapolate", assume_sorted=True)
    matter_density_parameter=sp.interpolate.interp1d(friedmann_sol.matter_density_parameter_numerical_a[0], friedmann_sol.matter_density_parameter_numerical_a[1],
                                                     fill_value="extrapolate", assume_sorted=True)
    
    """Defining the integration parameters"""
    atol=1e-12
    rtol=1e-13
    t_min= params.scale_at_lss#We start at last scattering 
    t_max= params.a_max #1.062 for ta, 1 1.69 for collapse
    
    """Compute the exponent of the growing mode at last scattering"""
    n1=round(0.25*(-1+3*effective_eos_a(t_min)+sqrt((1-3*effective_eos_a(t_min))**2+24)),4)
    print("Exp. old:",n1)
    n1=round(0.25*(-1+np.sqrt(24*matter_density_parameter(t_min)+(1-3*effective_eos_a(t_min))**2)+3*effective_eos_a(t_min)),4)
    print("Exponent at ls = ",n1)
    
    delta_m_tls=(params.scale_at_lss)**n1
    d_delta_m_tls=n1*(params.scale_at_lss)**(n1-1)
    delta_de_tls=params.scale_at_lss*(1+params.de_eos_a(params.scale_at_lss))#Adiabatic initial condition
    d_delta_de_tls=n1*(1+params.de_eos_a(params.scale_at_lss))+params.scale_at_lss*cosm_func.de_eos_derivative_a(params.scale_at_lss)
    
    init_cond=[delta_m_tls,d_delta_m_tls,delta_de_tls,d_delta_de_tls]# [Position(t=t_min),Speed(t_tmin)]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(fun,t_span=(t_min,t_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    
    matter_density_contrast_numerical=np.array([list(rk4_result.t),rk4_result.y[0]])
    dark_energy_density_contrast_numerical=np.array([list(rk4_result.t),rk4_result.y[2]])
    result=[matter_density_contrast_numerical,dark_energy_density_contrast_numerical]
    
    # res2=mynm.rk4(fun, (t_min,t_max), init_cond, n)
    
    return result

#%%

# res=solve(friedmann_sol)
# matter_density_contrast_numerical=res[0]
# dark_energy_density_contrast_numerical=res[1]

#%%

    
# mypl.m_plot([matter_density_contrast_numerical,dark_energy_density_contrast_numerical],
#             func_to_compare=lambda x:x,
#             legend=("matter","de","matterEdS"),
#             # xscale="log",yscale="log",
#             xscale="linear",yscale="linear",
#             dotted=False,ylim=None)
# pl.show()

    
# mypl.m_plot(matter_density_contrast_numerical,
#             func_to_compare=lambda x:x,
#             legend=("matter","matterEdS"),
#             xscale="log",yscale="log",dotted=False,ylim=None)
# pl.show()

# growth_evolution_matter=[]
# growth_evolution_matter=[1/matter_density_contrast_numerical[0]-1,matter_density_contrast_numerical[1]]

# growth_evolution_de=[]
# growth_evolution_de=[1/dark_energy_density_contrast_numerical[0]-1,dark_energy_density_contrast_numerical[1]]

# mypl.m_plot([growth_evolution_matter,growth_evolution_de],"$z$","$\delta_m(z)(1+z)$",
#         "Matter growth evolution $\delta_m(z)(1+z)$",
#         legend=("$\delta_m(z)$","$\delta_{de}(z)$","$\delta_{lin}^{EdS}$"),
#         func_to_compare=lambda z:1/(1+z),
#         save=False,
#         name="growth_evolution_matter_z",
#         xscale="log")
# pl.show()

