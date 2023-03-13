# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:03:20 2023

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

import numerical_methods as mynm
import plotting_functions as mypl
import dark_energy_eos as eos
import lcdm_model

#%%
"""Physical constants """
c=299792458#m s-1
G=6.67430e-11# m3 kg-1 s-2

"""Unit conversions"""
Gy=31557600*10**9# s
Mpc=3.085677581e22# m



""" Cosmological parameters """
w_i=-1
w_f=-0.2
z_t=0.5
gamma=5
phi_tilde_now=0
omega_matter_now=0.31
hubble_constant_standard_units = 67.4# km s-1 Mpc-1
omega_rad_now=0*round(0.2473/(hubble_constant_standard_units)**2,5)

scale_at_lss=1/(1+1089.0)#t_ls=304171
scale_at_equivalence=1/(24000*omega_matter_now*0.674**2)

"""Derived constants"""
omega_matter_now=omega_matter_now-omega_rad_now
omega_dark_now = 1-omega_matter_now
hubble_constant = hubble_constant_standard_units*1e3*Gy/Mpc# Gy-1
"""H[s]=H[Gy]/Gy -> H in sec. = H in Gy divided by 1Gy in seconds"""
critical_density = 3*hubble_constant**2/(8*pi*G*Gy**2)#kg m-3

"""Numerical integration parameters"""
n = int(1e4)#Number of steps used in mynm.integrate... 
a_min=1e-5#Minumum value of a. Ideally should be zero
a_max=1
print(f"Run parameters:  N= {n:.1e}","    ","a_min=",a_min,"    a_max=",a_max)
#%%

"""Matter density evolution in function of a"""
def matter_density_evolution_a(a):
    matter_density_evolution_a= omega_matter_now/a**3
    return matter_density_evolution_a

"""Radiation density evolution in function of a"""
def rad_density_evolution_a(a):
    rad_density_evolution_a= omega_rad_now/a**4
    return rad_density_evolution_a

def de_exponent(a):
    de_exponent = (1+de_eos(a))/a
    return de_exponent



init_value=0
final_value=2.5
n_samples=2
var_par=2#[0->w_i; 1->w_f; 2->z_t; 3->gamma]
params=[-0.2,-1,0.5,10]#[w_i, w_f, z_t, gamma]
step=(final_value-init_value)/n_samples
params[var_par]=init_value

params_names=["w_i","w_f","z_t","\Delta"]
phis_tilde_numerical_a=[]
potentials_tilde_numerical_phi=[]
legend=[]
for i in range(n_samples+1):
    print("    varying: ",params_names[var_par],"    i = ",i,"    value = ",round(params[var_par],3))
    legend.append("$"+params_names[var_par]+"$="+str(round(params[var_par],3)))
                  # +"$-> a_t=$"+str(round(1/(1+params[var_par]),2))  )#To convert the z_t to a_t
    """ de_eos is the w parameter: the ratio between P/rho"""   
    def de_eos(a):
        z=1/a-1
        de_eos=eos.de_eos_1(z,w_i=params[0],w_f=params[1],z_t=params[2],gamma=params[3])
        return de_eos

    """Compute dark_density_evolution(a) == dark_density_evolution_a """
    de_exponent_integral=mynm.integrate(1,a_min,n,de_exponent)
    scale_parameter_values=de_exponent_integral[0]
    dark_density_evolution_numerical_a=np.array([scale_parameter_values,omega_dark_now*exp(-3*de_exponent_integral[1])])
    
    # mypl.m_plot(dark_density_evolution_numerical_a,title="Dark density evolution",
    #             func_to_compare=lambda a:omega_dark_now*a**(-3*(1+w_i+w_f))*exp(-3*w_i*(1-a)),dotted=False,
    #             xscale="log",yscale="log")
    # pl.show()
    
    dark_density_evolution_a=sp.interpolate.interp1d(scale_parameter_values, dark_density_evolution_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    
    """Integrate the friedmann equation to get a(t)"""
    def friedmann_equation_integrand(a):
        friedmann_equation_integrand = 1/(a*sqrt(matter_density_evolution_a(a) + dark_density_evolution_a(a)+rad_density_evolution_a(a)))
        return friedmann_equation_integrand
    
    hubble_constant_times_t = mynm.integrate(a_max, a_min, n, friedmann_equation_integrand)
    
    """Compute t(a)==time_a"""
    time_minus_universe_age_a=hubble_constant_times_t[1]/hubble_constant
    universe_age = -time_minus_universe_age_a[0]#In giga yeras
    time_a=time_minus_universe_age_a+universe_age
    scale_parameter_numerical_t=np.array([time_a,scale_parameter_values])


    """Compute the potential and the scalar field as functions of a and t"""
    def potential_tilde_a(a):
        potential_tilde_a=dark_density_evolution_a(a)*(1-de_eos(a))
        return potential_tilde_a
    potential_tilde_numerical_a=np.array([scale_parameter_values,potential_tilde_a(scale_parameter_values)])
    potential_tilde_numerical_t=np.array([time_a,potential_tilde_numerical_a[1]])
    
    def phi_tilde_integrand_a(a):
        phi_tilde_integrand_a=sqrt((a*dark_density_evolution_a(a)*(1+de_eos(a)))/ \
                                   (omega_matter_now+a**3*dark_density_evolution_a(a)+omega_rad_now/a))
        return phi_tilde_integrand_a
    
    phi_tilde_numerical_a=mynm.integrate(1,a_min,n,phi_tilde_integrand_a)
    phi_tilde_numerical_a[1]=phi_tilde_numerical_a[1]+phi_tilde_now
    phi_tilde_a=sp.interpolate.interp1d(scale_parameter_values, phi_tilde_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    
    phi_tilde_numerical_t=np.array([time_a,phi_tilde_numerical_a[1]])
    
    
    phi_tilde_numerical_phi=np.array([phi_tilde_numerical_a[1],potential_tilde_numerical_a[1]])
    
    """Approximate the eos with a step function: 
    Compute an approximation for the dark density evolution """
    def approx_dark_density_evolution_a(a):
        a_t=1/(1+z_t)
        if a_t>1:
            raise Exception("z_t must be >=0")
        if a>=a_t:
            return omega_dark_now*a**(-3*(1+w_f))
        if a<a_t:
            return  omega_dark_now*a**(-3*(1+w_i))*a_t**(-3*(w_f-w_i))
    aux=[]
    for i in scale_parameter_values:
        aux.append(approx_dark_density_evolution_a(i))
    approx_dark_density_evolution_numerical_a=np.array([scale_parameter_values,aux])
    
    mypl.m_plot([dark_density_evolution_numerical_a,approx_dark_density_evolution_numerical_a],title="Dark density evolution apporximation",
              legend=("True","Step approximation"),yscale="log",xscale="log",
              xlim=(1e-1,1e0),ylim=(0.1,1e4))
    pl.show()
    
    
    def approx_potential_tilde_a(a):
        a_t=1/(1+z_t)
        if a_t>1:
            raise Exception("z_t must be >=0")
        if a>=a_t:
            return omega_dark_now*a**(-3*(1+w_f))*(1-w_f)
        if a<a_t:
            return  omega_dark_now*a**(-3*(1+w_i))*a_t**(-3*(w_f-w_i))*(1-w_i)
    aux=[]
    for i in scale_parameter_values:
        aux.append(approx_potential_tilde_a(i))
    approx_potential_tilde_numerical_a=np.array([scale_parameter_values,aux])
    
    mypl.m_plot([potential_tilde_numerical_a,approx_potential_tilde_numerical_a],title="Potential approx. $\~{V}(a)$",
              legend=("True","Step approximation"),yscale="log",xscale="log",
              xlim=(1e-1,1e0),ylim=(0.1,1e4))
    pl.show()

#%%
save = False



"""SCALAR FIELD TILDE"""
mypl.m_plot(phis_tilde_numerical_a,
            xlabel="$a$",
            ylabel="$\~{\phi} (a)$",
            title="Scalar field $\~{\phi} (a)$",
            legend=("$\~{\phi}$","$\~{\phi}_m$"),
            func_to_compare=lambda a :sqrt(omega_dark_now)*log(a),
            save=False,name=None,xscale="log",yscale="linear",
            xlim=None,ylim=None, dotted=True)
pl.show()
    



"""POTENTIAL OF PHI"""
mypl.m_plot(potentials_tilde_numerical_phi,
            xlabel="$\~{\phi}$",
            ylabel="$\~{V}(\~{\phi})$",
            title="Potential",
            legend=("$\~{V}(\~{\phi})$","$\~{V}(\~{\phi})_m$"),
            func_to_compare=lambda x :omega_dark_now*exp(-3*x/sqrt(omega_dark_now)),
            save=save,name="phi_tilde_numerical_phi",xscale="linear",yscale="log",
            xlim=None,ylim=None, dotted=True)
pl.show()