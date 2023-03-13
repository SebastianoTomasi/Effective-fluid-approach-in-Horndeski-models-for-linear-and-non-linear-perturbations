# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:36:38 2022

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
import matplotlib.ticker as ticker

import sys
sys.path.append("../data_modules")
sys.path.append("../utility_modules")
sys.path.append("../friedmann_solver")

import friedmann_solver as fr_sol
import numerical_methods as mynm
import plotting_functions as mypl
from cosmological_functions import *
from simulation_parameters import *
from data_classes import quintessence_sol
#%%
"""We need the dark density evolution for the particular dark energy"""
dark_density_evolution_numerical_a=fr_sol.solve_dark_density_evolution_numerical_a(de_eos_a, a_min,a_max, n)
scale_parameter_values=dark_density_evolution_numerical_a[0]
dark_density_evolution_a=sp.interpolate.interp1d(scale_parameter_values, dark_density_evolution_numerical_a[1],
                                    fill_value="extrapolate", assume_sorted=True)

"""Definitions of some functions """

def potential_tilde_a(a):
    potential_a=dark_density_evolution_a(a)*(1-de_eos_a(a))
    return potential_a

def approx_potential_tilde_a(a):
    a_t=1/(1+trans_z)
    if a_t>1:
        raise Exception("z_t must be >=0")
    if a>=a_t:
        return omega_dark_now*a**(-3*(1+w_f))*(1-w_f)
    if a<a_t:
        return  omega_dark_now*a**(-3*(1+w_i))*a_t**(-3*(w_f-w_i))*(1-w_i)

def phi_tilde_integrand_a(a):
    w=de_eos_a(a)
    if w<-1:
        raise Exception("The program only deals with quintessence, hence -1<w<1")
    phi_tilde_integrand_a=sqrt((a*dark_density_evolution_a(a)*(1+w))/ \
                               (omega_matter_now+a**3*dark_density_evolution_a(a)+omega_rad_now/a))
    return phi_tilde_integrand_a

def approx_phi_tilde_integrand_a(a):
    w=step_eos(a)
    if w<-1:
        raise Exception("The program only deals with quintessence, hence -1<w<1")
    approx_phi_tilde_integrand_a=sqrt((a*step_dark_density_evolution_a(a)*(1+w))/ \
                               (omega_matter_now+a**3*step_dark_density_evolution_a(a)+omega_rad_now/a))
    return approx_phi_tilde_integrand_a

# x=np.linspace(1e-5,1,500)
# y0=[]
# y1=[]
# for i in x:
#     y0.append(approx_phi_tilde_integrand_a(i))
#     y1.append(phi_tilde_integrand_a(i))
# mypl.m_plot([[x,y1],[x,y0]],yscale="linear",xlim=(0.2,1),ylim=(-1,2))

#%%


def solve(dark_density_evolution_a=dark_density_evolution_a):
    """Create a quintessence solution object to store the solutions"""
    result=quintessence_sol()
    
    """Evaluate the potential on the scale parameter values"""
    potential_tilde_numerical_a=np.array([scale_parameter_values,potential_tilde_a(scale_parameter_values)])
    
    """Compute the integral for phi_tilde"""
    phi_tilde_numerical_a=mynm.integrate(a_max,a_min,n,phi_tilde_integrand_a)

    """Compute V(phi) by inverting phi(a)->a(phi) => V(a)->V(a(phi))=V(phi):"""
    potential_tilde_numerical_phi=np.array([phi_tilde_numerical_a[1],potential_tilde_numerical_a[1]])
    
    """Approximation of the eos with a step function: """
    """Compute an approximation for the dark density evolution """
    aux=[]
    for i in scale_parameter_values:
        aux.append(step_dark_density_evolution_a(i))
    approx_dark_density_evolution_numerical_a=np.array([scale_parameter_values,aux])
    # mypl.plot([dark_density_evolution_numerical_a,approx_dark_density_evolution_numerical_a],legend=("dde","approx. dde"))
              
    """ Compute the approximate integral for phi_tilde"""
    approx_phi_tilde_numerical_a=mynm.integrate(a_max,a_min,n,approx_phi_tilde_integrand_a)
    
    """Approx. for the potential"""
    aux=[]
    for i in scale_parameter_values:
        aux.append(approx_potential_tilde_a(i))
    approx_potential_tilde_numerical_a=np.array([scale_parameter_values,aux])
    
    """Invert the approximate phi(a) and obtain the approx. V(phi)"""
    approx_potential_tilde_numerical_phi=np.array([approx_phi_tilde_numerical_a[1],approx_potential_tilde_numerical_a[1]])
    
    # """Convert to physical units to get V and phi without tilde"""
    # phi_numerical_a=[scale_parameter_values,0]
    # potential_numerical_a=[scale_parameter_values,0]
    # approx_potential_numerical_a=[scale_parameter_values,0]
    # phi_numerical_a[1]=phi_tilde_numerical_a[1]*sqrt(3/(8*pi*G))
    # potential_numerical_a[1]=potential_tilde_numerical_a[1]*3*(hubble_constant/Gy)**2/(16*pi*G)
    # approx_potential_numerical_a[1]=approx_potential_tilde_numerical_a[1]*3*(hubble_constant/Gy)**2/(16*pi*G)
    
    """Save the results and return tham"""
    result.phi_tilde_numerical_a=phi_tilde_numerical_a
    result.approx_phi_tilde_numerical_a=approx_phi_tilde_numerical_a
    result.potential_tilde_numerical_a=potential_tilde_numerical_a
    result.approx_potential_tilde_numerical_a=approx_potential_tilde_numerical_a
    result.potential_tilde_numerical_phi=potential_tilde_numerical_phi
    result.approx_potential_tilde_numerical_phi=approx_potential_tilde_numerical_phi
    
    return result
#%%

res=solve()

#%%
"""PLOTTING"""

save = True

# """POTENTIAL of a"""
# mypl.m_plot(res[1],
#             xlabel="$a$",
#             ylabel="${V}(a)$",
#             title="Scalar field potential ${V}(a)$",
#             legend=("Dark energy","Matter"),
#             func_to_compare=lambda a :3*(hubble_constant/Gy)**2/(16*pi*G)*omega_dark_now/(a**3),
#             save=save,name=None,xscale="linear",yscale="log",
#             xlim=None,ylim=None, dotted=False)
# pl.show()
# """SCALAR FIELD of a """
# mypl.m_plot(res[0],
#             xlabel="$a$",
#             ylabel="$\~{\phi} (a)$",
#             title="Scalar field $\~{\phi} (a)$",
#             legend=("$\~{\phi}$","$\~{\phi}_m$"),
#             func_to_compare=lambda a :sqrt(3/(8*pi*G))*sqrt(omega_dark_now)*log(a),
#             save=False,name=None,xscale="linear",yscale="linear",
#             xlim=None,ylim=None, dotted=False)
# pl.show()
# """POTENTIAL TILDE of phi"""
# mypl.m_plot([res[0][1],res[1][1]],
#             xlabel="$\~{\phi}$",
#             ylabel="$\~{V}(\~{\phi})$",
#             title="Potential",
#             legend=("$\~{V}(\~{\phi})$","approx $\~{V}(\~{\phi})$","$\~{V}(\~{\phi})_m$"),
#             save=save,name="phi_tilde_numerical_phi",xscale="linear",yscale="log",
#             xlim=None,ylim=None, dotted=False)
# pl.show()




#%%




"""POTENTIAL TILDE of a"""
mypl.m_plot([res.potential_tilde_numerical_a,res.approx_potential_tilde_numerical_a],
            xlabel="$a$",
            ylabel="$\~{V}(a)$",
            title="Scalar field potential $\~{V}(a)$",
            legend=("True","Step approx."),
            # func_to_compare=lambda a :omega_dark_now/(a**3),
            save=save,name="potential_tilde_a",xscale="linear",yscale="log",
            xlim=None,ylim=None, dotted=False)
pl.show()



"""SCALAR FIELD TILDE of a """
mypl.m_plot([res.phi_tilde_numerical_a,res.approx_phi_tilde_numerical_a],
            xlabel="$a$",
            ylabel="$\~{\phi} (a)$",
            title="Scalar field $\~{\phi} (a)$",
            # legend=("$\~{\phi}$","$\~{\phi}_m$"),
            legend=("True","Step approx."),
            # func_to_compare=lambda a :sqrt(omega_dark_now)*log(a),
            save=save,name="scalar_field_tilde_a",xscale="linear",yscale="linear",
            xlim=None,ylim=(-3,0.5), dotted=False)
pl.show()

    

"""POTENTIAL TILDE of phi"""
mypl.m_plot([res.potential_tilde_numerical_phi,res.approx_potential_tilde_numerical_phi],
            xlabel="$\~{\phi}$",
            ylabel="$\~{V}(\~{\phi})$",
            title="Scalar field potential $\~{V}(\~{\phi})$",
            legend=("True","Step approx."),
            # legend=("$\~{V}(\~{\phi})$","approx $\~{V}(\~{\phi})$","$\~{V}(\~{\phi})_m$"),
            # func_to_compare=lambda x :1.05*x/x,
            # func_to_compare=lambda x :omega_dark_now*exp(-3*x/sqrt(omega_dark_now)),
            save=save,name="potential_tilde_phi",xscale="linear",yscale="log",
            xlim=None,ylim=None, dotted=False)
pl.show()

mypl.save_run_parameters()
