# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:22:40 2023

@author: sebas
"""

import numpy as np
import os

import sys

sys.path.append("../utility_modules")
import plotting_functions as mypl
import import_export as myie
import lcdm_model as lcdm
#%%

save=False

this_run_specifier_0="nonlinear_perturbations"
this_run_specifier_1="perturbed_de"
this_run_specifier_2="de_eos_5"
this_run_specifier_3="trans_z"

name_specifiers_0=["nonlinear_perturbations","linear_perturbations"]
name_specifiers_1=["plots","unperturbed_de","perturbed_de","LCDM","EDS"]
name_specifiers_2=["de_eos_"+str(i) for i in range(1,8)]
# name_specifiers_3=['w_i','w_f','trans_steepness','trans_z']
name_specifiers_3=[this_run_specifier_3]






#%%

"""Import the LCDM to compare evrithing else aganist."""
lcdm_linear_matter_density_contrasts_at_collapse_z=myie.import_from_txt_twocolumns("../data/"+this_run_specifier_0+"/LCDM/delta_c")
lcdm_virialization_overdensities_star=myie.import_from_txt_twocolumns("../data/"+this_run_specifier_0+"/LCDM/zeta_vir_star")

#%%

save_plot_to_path="../data"+"/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/plots/"+this_run_specifier_3+"/"
get_data_from_path="../data"+"/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/"+this_run_specifier_3+"/"

try:
    linear_matter_density_contrasts_at_collapse_z=myie.import_from_txt_multicolumn(get_data_from_path+"delta_c")
    virialization_overdensities_star= myie.import_from_txt_multicolumn(get_data_from_path+"zeta_vir_star")
    effective_eos_a=myie.import_from_txt_multicolumn(get_data_from_path+ "effective_eos")
    de_eos_a=myie.import_from_txt_multicolumn(get_data_from_path+"dark_energy_eos")
    omega_matter_a=myie.import_from_txt_multicolumn(get_data_from_path+"omega_matter")
    growth_factos_a=myie.import_from_txt_multicolumn(get_data_from_path+"growth_factors_0")
except FileNotFoundError:
    print("Error: file not found. Check if the data is available.")
    

"""Append the lcdm model curves"""
linear_matter_density_contrasts_at_collapse_z.append(lcdm_linear_matter_density_contrasts_at_collapse_z)
virialization_overdensities_star.append(lcdm_virialization_overdensities_star)

"""Set up the legend"""
params_names={"w_i":"w_i",
            "w_f":"w_f",
            "trans_steepness":"\Gamma",
            "trans_z":"z_t",
            "de_eos_a":this_run_specifier_2}


"""Import the values of the varied parameter and build the legend"""
var_par_values=myie.import_varied_parameter_values(get_data_from_path+"delta_c",this_run_specifier_3)
legend=[]
for i in var_par_values:
    legend.append("$"+params_names[this_run_specifier_3]+"$="+str(round(i,3)))
legend.append("$\Lambda CDM$")    

"""Do the actual plotting"""
mypl.m_plot(linear_matter_density_contrasts_at_collapse_z,legend=legend,
            # title="Linearly extrapolated matter density contrast at collapse.",
            title="",
            xlabel="$z_c$",ylabel="$\delta_c$",
            save=save,
            name="delta_c",
            save_plot_dir=save_plot_to_path)

mypl.m_plot(virialization_overdensities_star,legend=legend,
            # title="Matter overdensity at virialization.",
            title="",
            xlabel="$z_c$",ylabel="$\zeta_{vir}^*$",
            save=save,
            name="zeta_vir_star",
            save_plot_dir=save_plot_to_path)

mypl.m_plot(effective_eos_a,legend=legend,
            # title="Effective eos",
            title="",
            func_to_compare=lcdm.effective_eos_a,
            xlabel="$a$",ylabel="$w_{eff}$",
            save=save,
            name="effective_eos",
            save_plot_dir=save_plot_to_path)

mypl.m_plot(de_eos_a,legend=legend,
            # title="eos: "+this_run_specifier_2+".",
            title="",
            xlabel="$a$",ylabel="$w$",
            save=save,
            name="dark_energy_eos",
            save_plot_dir=save_plot_to_path)

mypl.m_plot(omega_matter_a,legend=legend,
            # title="Matter density parameter.",
            title="",
            func_to_compare=lcdm.omega_matter_a,
            xlabel="$a$",ylabel="$\Omega_m$",
            save=save,
            name="omega_matter",
            xscale="log",
            save_plot_dir=save_plot_to_path)
if save:
    myie.save_header(myie.import_header(get_data_from_path+"/delta_c"),save_plot_to_path+"/"+"simulation_parameters")































