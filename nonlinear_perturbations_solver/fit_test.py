# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:47:40 2023

@author: Sebastiano Tomasi
"""
import scipy as sp
import numpy as np

import matplotlib.pyplot as pl

import numpy as np
import os

import sys

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func
import constants as const

sys.path.append("../utility_modules")
import numerical_methods as mynm
import plotting_functions as mypl
import import_export as myie

sys.path.append("../friedmann_solver")
import friedmann_solver as fr_sol
#%%

save=True

this_run_specifier_0="nonlinear_perturbations"
this_run_specifier_1="perturbed_de"
this_run_specifier_2="de_eos_2"
this_run_specifier_3="trans_steepness"

name_specifiers_0=["nonlinear_perturbations","linear_perturbations"]
name_specifiers_1=["plots","unperturbed_de","perturbed_de","LCDM","EDS"]
name_specifiers_2=["de_eos_"+str(i) for i in range(1,8)]
# name_specifiers_3=['w_i','w_f','trans_steepness','trans_z']
name_specifiers_3=[this_run_specifier_3]

lcdm_virialization_overdensities_star=myie.import_from_txt_twocolumns("../data/"+this_run_specifier_0+"/LCDM/zeta_vir_star")









#%%

def hubble_ratio(x,c,s):
    hubble_ratio=s*(params.omega_matter_now+params.omega_dark_now/(1+x)**3)/params.omega_matter_now+c
    # hubble_ratio=s*params.omega_dark_now/(params.omega_dark_now*(1+x)**3)+c
    return hubble_ratio
#%%

z_c=lcdm_virialization_overdensities_star[0]
zeta_vir=lcdm_virialization_overdensities_star[1]


pars=[106,71]

res = sp.optimize.curve_fit(hubble_ratio, z_c, zeta_vir, p0=pars)
new_pars=res[0]
print("new pars:",new_pars)
errors=np.sqrt(np.diag(res[1]))
print("Deviations:",errors)
print("percentage errors:",errors/new_pars)
W=np.linspace(z_c[0],z_c[-1],200)
    
fig1 = pl.figure(1,figsize=(10,6))
# pl.xscale('log')
pl.plot(z_c,zeta_vir,'ob')
# pl.plot(W,hubble_ratio(W, pars[0],pars[1]),'b')#Initial curve
pl.plot(W, hubble_ratio(W,new_pars[0],new_pars[1]),'r')#Fitting curve
pl.title("",fontsize=16)
pl.xlabel("$z_c$",fontsize=12)
pl.ylabel("$\zeta^*_{vir}$",fontsize=12)
pl.legend(('Punti', 'Curva teorica', 'Curva reale'),
        shadow=True, loc="best", handlelength=1.5, fontsize=12)
pl.draw()
pl.show()