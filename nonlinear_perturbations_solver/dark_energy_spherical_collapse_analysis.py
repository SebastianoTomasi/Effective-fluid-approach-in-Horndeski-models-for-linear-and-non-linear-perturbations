# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:29:28 2023

@author: sebas
"""


import matplotlib.pyplot as pl
import numpy as np

import sys
sys.path.append("../data_modules")
import simulation_parameters as params

sys.path.append("../utility_modules")
import plotting_functions as mypl
import import_export as myie

sys.path.append("../friedmann_solver")
import friedmann_solver as fr_sol

sys.path.append("../perturbation_solver")


import time
start_time = time.time()
#%%

"""
This module computes the nonlinear perturbations for the dark energy 
model and parameters specified in the simulation_parameters module. 
The variable var_par is the parameter of the eos we want to vary. We
vary it in the range [init_value,final_value] with (n_samples+1) points.
The specifiers must be correctly named according to the folder structure of 
the method import_export.generate_data_folders().
If one want to switch from unperturbed dark energy to perturbed, can change the value
of the boolean does_de_cluster to True."""

does_de_cluster=True


number_of_points=50#Number of data points computed for delta_c and zeta_vir_star
var_par="trans_z" # can be = 'w_i';'w_f';'trans_steepness';'trans_z';
init_value=-2.9
final_value=-0.5
n_samples=1

"""Print out the other parameters to double check their values."""
for key,value in params.dark_energy_parameters.items():
    print(f"{key}: {value}")


#%%

"""Checks: if the number of samples is choosen as zero, we solve 
for only one var_par value """
if n_samples>0:
    varied_par_values=np.linspace(init_value,final_value,n_samples)
    
    varied_par_values=[0.1,1,4]
    # varied_par_values=[0,-0.023,-0.047,-0.07,-0.3,-0.9]
    # varied_par_values=[1,2,5]
    n_samples=len(varied_par_values)
elif n_samples==0:
    varied_par_values=[init_value]
else:
    raise Exception("The number of samples must be greater or equal to zero")

"""We save the original value of the varied parameter, then
we change the parameter values in the params namespace so
that all moduels notify the change."""
exec("reset_param_value=params."+var_par)#Save the original inital value to reset it
#%%

"""Check if de can cluster or not"""
if does_de_cluster:
    import clustering_de_perturbations as sph_c
    this_run_specifier_1="perturbed_de"
elif not does_de_cluster:
    import bakground_de_spherical_collapse as sph_c
    this_run_specifier_1="unperturbed_de"
    


"""Construct the path where to save the data."""
this_run_specifier_0="nonlinear_perturbations"
this_run_specifier_2=params.dark_energy_eos_selected
this_run_specifier_3=var_par


save_data_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/"+this_run_specifier_3+"/"
save_plot_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/plots/"+this_run_specifier_3+"/"


#%%
"""Actual cycle on the parameter values."""

params_names={"w_i":"w_i",#Used to build the legend
            "w_f":"w_f",
            "trans_steepness":"\Gamma",
            "trans_z":"z_t",
            "de_eos_a":params.dark_energy_eos_selected}

linear_matter_density_contrasts_at_collapse_z=[]#Used to store results.
virialization_overdensities_star_z=[]
effective_eos_a=[]
omega_matter_a=[]
de_eos_a=[]
legend=[]
for i in range(len(varied_par_values)):
    exec("params."+var_par+"="+str(varied_par_values[i]))#Update the var_par value across the modules.
    """Construct the legend"""
    legend.append("$"+params_names[var_par]+"$="+str(round(varied_par_values[i],3)))
    
    """Call the friedmann solver.
    If time_domain=True then it will print out the deceleration
    parameter and the age of the universe for the model considered."""      
    time_domain=False
    friedmann_sol=fr_sol.solve(time_domain=time_domain)
    
    """Save some bakground results."""
    de_eos_a.append(friedmann_sol.de_eos_numerical_a)
    effective_eos_a.append(friedmann_sol.effective_eos_numerical_a)
    omega_matter_a.append(friedmann_sol.matter_density_parameter_numerical_a)
    
    """Call the spherical collapse solver""" 
    nonlin_pertrubations=sph_c.solve(friedmann_sol,number_of_points)
    linear_matter_density_contrasts_at_collapse_z.append(nonlin_pertrubations[0])
    virialization_overdensities_star_z.append(nonlin_pertrubations[1])
    
    """Save to a txt the growth evolutions."""
    myie.save_to_txt_multicolumn(nonlin_pertrubations[2],save_data_to_path + "growth_factors_"+str(i),
                                 var_par,varied_par_values)
    
    """Print a table with some informations to keep track where is the cycle at."""
    labels=["Var_par:"+var_par,"t_0","q_0"]
    if time_domain:
        values=[round(varied_par_values[i],3),round(friedmann_sol.universe_age,3),round(friedmann_sol.deceleration_parameter,3)]
    else:
        values=[round(varied_par_values[i],3),"#","#"]
    if i==0:
        print("*" * 50)
        print("{:<27} {:<12} {:<12}".format(*labels))
        print("*" * 50)
        print("{:<27} {:<12} {:<12}".format(*values))
        print("*" * 50)
    else:
        print("{:<27} {:<12} {:<12}".format(*values))
        print("*" * 50)
    
exec("params."+var_par+"=reset_param_value")#Reset the original value 
#%%

"""Save all data to .txt files in the correct folders"""
myie.save_to_txt_multicolumn(linear_matter_density_contrasts_at_collapse_z, save_data_to_path+"delta_c",var_par,varied_par_values)
myie.save_to_txt_multicolumn(virialization_overdensities_star_z, save_data_to_path+"zeta_vir_star",var_par,varied_par_values)
myie.save_to_txt_multicolumn(effective_eos_a, save_data_to_path+"effective_eos",var_par,varied_par_values)
myie.save_to_txt_multicolumn(de_eos_a, save_data_to_path+"dark_energy_eos",var_par,varied_par_values)
myie.save_to_txt_multicolumn(omega_matter_a, save_data_to_path+"omega_matter",var_par,varied_par_values)

#%%
"""PLOTTING """
save=True
"""Linear matter density contrast at collapse"""
mypl.m_plot(linear_matter_density_contrasts_at_collapse_z,"$z$","$\delta_c(z)$",
        "Linear density contrast at collapse",
        legend=legend,
        # func_to_compare=lambda x: x,
        save=save,dotted=False,
        name="linear_matter_density_contrast_collapse_z",
        )
pl.show()


mypl.m_plot(virialization_overdensities_star_z,"$z$","$\zeta_{vir}^*(z)$",
        "Overdensity at virialization",
        legend=legend,
        save=save,dotted=False,
        name="overdensity_at_virialization_star",
        )
pl.show()


print("Time taken to complete the execution: {:.2f} seconds".format(time.time() - start_time))