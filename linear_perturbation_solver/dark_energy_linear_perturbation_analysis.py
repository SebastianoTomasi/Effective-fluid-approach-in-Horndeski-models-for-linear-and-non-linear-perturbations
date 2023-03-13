# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:23:56 2022

@author: sebas
"""
import numpy as np
from numpy import sqrt

import scipy as sp
import matplotlib.pyplot as pl

import sys
sys.path.append("../data_modules")
import simulation_parameters as params


sys.path.append("../utility_modules")
import plotting_functions as mypl
import lcdm_model
import cosmological_functions as cosm_func
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


var_par="trans_z" # can be = 'w_i';'w_f';'trans_steepness';'trans_z';
init_value=0
final_value=10
n_samples=0

"""Print out the other parameters to double check their values."""
for key,value in params.dark_energy_parameters.items():
    print(f"{key}: {value}")
    
#%%

"""Checks: if the number of samples is choosen as zero, we solve 
for only one var_par value """
if n_samples>0:
    varied_par_values=np.linspace(init_value,final_value,n_samples)
    
    # varied_par_values=[0.3,1.2,3,5]
    # varied_par_values=[0,-0.023,-0.047,-0.07,-0.3,-0.9]
    # varied_par_values=[0.1,1,2,10]
    # n_samples=len(varied_par_values)
    
elif n_samples==0:
    varied_par_values=[init_value]
    step=0
else:
    raise Exception("The number of samples must be greater or equal to zero")

"""We save the original value of the varied parameter, then
we change the parameter values in the params namespace so
that all moduels notify the change."""
exec("reset_param_value=params."+var_par)#Save the original inital value to reset it

#%%
"""Check if de can cluster or not"""
if does_de_cluster:
    this_run_specifier_1="perturbed_de"
    import perturbed_de_linear_perturbations as pert_solve
elif not does_de_cluster:
    this_run_specifier_1="unperturbed_de"
    import unperturbed_de_linear_perturbations as pert_solve

"""Construct the path where to save the data."""
this_run_specifier_0="linear_perturbations"
this_run_specifier_2=params.dark_energy_eos_selected
this_run_specifier_3=var_par


save_data_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/"+this_run_specifier_3+"/"
save_plot_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/plots/"+this_run_specifier_3+"/"


#%%
params_names={"w_i":"w_i",
            "w_f":"w_f",
            "trans_steepness":"\Gamma",
            "trans_z":"z_t",
            "de_eos":"de_eos_1"}

matter_density_contrasts_numerical_a=[]
dark_density_contrast_numerical_a=[]
growth_evolutions_z=[]
equations_of_state=[]
matter_density_parameters=[]
dark_density_parameters=[]
effective_eos=[]
hubble_functions=[]
legend=[]
for i in range(len(varied_par_values)):
    exec("params."+var_par+"="+str(varied_par_values[i]))#Update the var_par value across the modules.
    legend.append("$"+params_names[var_par]+"$="+str(round(varied_par_values[i],3)))
    
    time_domain=False             
    friedmann_sol=fr_sol.solve(time_domain=time_domain)
    
    """Build some intersting quantities matrices to plot: H(a),w(a),dark_density_evolution(a),omega_m(a)..."""
    equations_of_state.append(np.array(friedmann_sol.de_eos_numerical_a))
    matter_density_parameters.append(np.array(friedmann_sol.matter_density_parameter_numerical_a))
    dark_density_parameters.append(np.array(friedmann_sol.dark_density_parameter_numerical_a))
    effective_eos.append(np.array(friedmann_sol.effective_eos_numerical_a))
    hubble_functions.append(np.array(friedmann_sol.rescaled_hubble_functions_numerical_a))
    """Compute the perturbations and store the result"""
    if does_de_cluster:
        aux=np.array(pert_solve.solve_c_eff(friedmann_sol))#Can use solve or solve_c_eff
        matter_density_contrasts_numerical_a.append(aux[0])
        growth_evolutions_z.append([cosm_func.z_a(aux[0][0]),aux[1][1]/(aux[1][1][-1]*aux[0][0])])
        dark_density_contrast_numerical_a.append(aux[1])
    elif not does_de_cluster:
        aux=np.array(pert_solve.solve(friedmann_sol))
        matter_density_contrasts_numerical_a.append(aux)
        growth_evolutions_z.append([cosm_func.z_a(aux[0]),aux[1]/(aux[1][-1]*aux[0])])
    
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
myie.save_to_txt_multicolumn(growth_evolutions_z, save_data_to_path+"growth_evolution",var_par,varied_par_values)
myie.save_to_txt_multicolumn(matter_density_contrasts_numerical_a, save_data_to_path+"delta_m",var_par,varied_par_values)
myie.save_to_txt_multicolumn(effective_eos, save_data_to_path+"effective_eos",var_par,varied_par_values)
myie.save_to_txt_multicolumn(equations_of_state, save_data_to_path+"dark_energy_eos",var_par,varied_par_values)
myie.save_to_txt_multicolumn(matter_density_parameters, save_data_to_path+"omega_matter",var_par,varied_par_values)

#%%
"""PLOTTING """
save=True#Save the plots?
# legend.append("$EdS$")
legend.append("$\Lambda CDM$")


"""Use the lcdm_model to do comparisons"""
LCDM=lcdm_model.LCDM()
linear_perturbations_lcdm_numerical_a=LCDM.compute_perturbations()
lcdm_scale_parameter=linear_perturbations_lcdm_numerical_a[0]
linear_perturbations_lcdm_a=sp.interpolate.interp1d(lcdm_scale_parameter, 
                                linear_perturbations_lcdm_numerical_a[1],
                                   fill_value="extrapolate", assume_sorted=False)

#%%
"""EQUATIONS OF STATE"""
mypl.m_plot(equations_of_state,"$a$","$w$",
        title="",
        legend=legend,
        func_to_compare=LCDM.cosmological_constant_eos_a,
        save=save,
        save_plot_dir=save_plot_to_path,
        # xscale="log",
        name="dark_energy_eos",
        yscale="linear")
pl.show()

# hubble_function_z=[]
# for i in range(len(hubble_functions)):
#     hubble_function_z.append([cosm_func.z_a(hubble_functions[i][0]),
#                               hubble_functions[i][1]])
# """Hubble functions"""
# mypl.m_plot(hubble_function_z,"$z$",r"$\frac{H}{H_0}$",
#         title="",
#         legend=legend,
#         # func_to_compare=LCDM.cosmological_constant_eos_a,
#         save=save,
#         save_plot_dir=save_plot_to_path,
#         # xscale="log",
#         name="hubble_functions_detail",
#         # yscale="log",
#         # xscale="log",
#         ylim=(0,2.5),
#         xlim=(0,1)
#         )
# pl.show()
# """Hubble functions"""
# mypl.m_plot(hubble_functions,"$z$",r"$\frac{H}{H_0}$",
#         title="",
#         legend=legend,
#         # func_to_compare=LCDM.cosmological_constant_eos_a,
#         save=save,
#         save_plot_dir=save_plot_to_path,
#         # xscale="log",
#         name="hubble_functions",
#         yscale="log",
#         xscale="log",
#         )
# pl.show()

"""EFFECTIVE EOS"""
mypl.m_plot(effective_eos,"$a$","$w_{eff}$",
        title="",
        legend=legend,
        func_to_compare=LCDM.effective_eos_a,
        save=save,
        save_plot_dir=save_plot_to_path,
        # xlim=(0,1/(z_t+1)),
        # xscale="log",
        name="effective_eos")
pl.show()


"""MATTER DENSITY PARAMETERS"""
mypl.m_plot(matter_density_parameters,"$a$","$\Omega_{m}$",
            title="",
        legend=legend,
        func_to_compare=LCDM.omega_matter_a,
        save=save,
        save_plot_dir=save_plot_to_path,
        name="matter_density_parameter",
        xscale="log",
        yscale="linear")
pl.show()

if does_de_cluster:
    mypl.m_plot(dark_density_contrast_numerical_a,"$a$","$\delta_{de}$",
                title="",
            legend=legend,
            func_to_compare=None,
            save=save,
            save_plot_dir=save_plot_to_path,
            dotted=False,
            name="dark_contrast_matter_a",
            )
    pl.show()

"""Matter perturbations on the total energy density"""
matter_density_perturbations_hat_a=[]
for i in range(len(matter_density_contrasts_numerical_a)):
    a=matter_density_contrasts_numerical_a[i][0]
    omega_m=sp.interpolate.interp1d(matter_density_parameters[i][0], 
                                    matter_density_parameters[i][1],
                                    fill_value="extrapolate", assume_sorted=True)
    omega_de=sp.interpolate.interp1d(dark_density_parameters[i][0], 
                                    dark_density_parameters[i][1],
                                    fill_value="extrapolate", assume_sorted=True)
    matter_density_perturbations_hat_a.append([a,
    omega_m(a)*matter_density_contrasts_numerical_a[i][1]])
mypl.m_plot(matter_density_perturbations_hat_a,"$a$","$\Omega_m\delta_m$",
            title="",
        legend=legend,
        func_to_compare=None,
        save=save,
        save_plot_dir=save_plot_to_path,
        dotted=False,
        name="total_density_contrast",
        )
pl.show()

"""MATTER DENSITY CONTRAST"""
# matter_density_contrasts_numerical_a.append(linear_perturbations_eds_numerical_a)
mypl.m_plot(matter_density_contrasts_numerical_a,"$a$","$\delta_m$",
            title="",
        legend=legend,
        func_to_compare=linear_perturbations_lcdm_a,
        save=save,
        save_plot_dir=save_plot_to_path,
        dotted=False,
        name="density_contrast_matter_a",
        # xlim=(0.9,1),ylim=(0.7,0.85)
        )
pl.show()



"""GROWTH EVOLUTION as a function of z"""
# linear_perturbations_lcdm_z=sp.interpolate.interp1d(cosm_func.z_a(lcdm_scale_parameter),
#                                 linear_perturbations_lcdm_numerical_a[1]/(linear_perturbations_lcdm_numerical_a[1][-1]*lcdm_scale_parameter),
#                                     fill_value="extrapolate", assume_sorted=False)

# mypl.m_plot(growth_evolutions_z,"$z$","$D_+$",
#             title="",
#         legend=legend,
#         func_to_compare=linear_perturbations_lcdm_z,
#         xlim=(1e-1,1e3),
#         save=save,
#         # dotted=True,
#         save_plot_dir=save_plot_to_path,
#         name="growth_evolution_matter_z",
#         xscale="log"
#         )
# pl.show()

if save:
    mypl.save_run_parameters(save_plot_to_path)
print("Time taken to complete the execution: {:.2f} seconds".format(time.time() - start_time))


