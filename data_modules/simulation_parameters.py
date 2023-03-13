# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:55:08 2023

@author: sebas
"""
import dark_energy_eos as eos
import constants as const
from numpy import pi


"""From here you can tweak the simulation parameters"""

cosmological_parameters={"omega_matter_now":0.3,
                         "hubble_constant_standard_units":67.4,
                         "omega_rad_now":0,#round(0.2473/(67.4)**2,5)
                         "scale_at_lss":1/(1+1089.0)}

"""Equation of state for the dark energy"""
dark_energy_parameters={"w_i":-0.4,
                        "w_f":-1,
                        "trans_steepness":-2,
                        "trans_z":0.5,
                        "de_eos":"de_eos_5",
                        "c_eff_value":0}


"""Paremeters used to perform integrations with numerical_methods.integrate"""
integration_parameters={"n":int(1e5),
                        "a_min":9e-6,
                        "a_max":1}

dark_energy_eos_selected=dark_energy_parameters["de_eos"]
exec("de_eos_z=eos." + dark_energy_eos_selected )
def de_eos_a(a):
    z=1/a-1
    de_eos_a=de_eos_z(z,w_i,w_f,trans_z,trans_steepness)
    return de_eos_a

def c_eff(a):
    c_eff=c_eff_value
    return c_eff




#%%
"""Generate the variables
by defining them as key=value """
for key, value in cosmological_parameters.items():
    globals()[key]=value
    
for key,value in integration_parameters.items():
    globals()[key]=value
    
for key,value in dark_energy_parameters.items():
    globals()[key]=value
#%%


"""Parameters derived from the one already given"""

scale_at_equivalence=1/(2.4*omega_matter_now*hubble_constant_standard_units**2)
omega_matter_now=omega_matter_now-omega_rad_now
omega_dark_now = 1-omega_matter_now
hubble_constant = hubble_constant_standard_units*1e3*const.Gy/const.Mpc# Gy-1
"""H[s]=H[Gy]/Gy -> H in sec. = H in Gy divided by 1Gy in seconds"""
critical_density = 3*hubble_constant**2/(8*pi*const.G*const.Gy**2)#kg m-3