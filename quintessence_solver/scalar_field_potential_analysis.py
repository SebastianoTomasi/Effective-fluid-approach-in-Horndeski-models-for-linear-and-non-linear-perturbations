# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:51:29 2023

@author: sebas
"""

import numpy as np
import matplotlib.pyplot as pl

import quintessence_solver 

import sys

sys.path.append("../utility_modules")
import numerical_methods as mynm

sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func
import data_classes 

sys.path.append("../utility_modules")
import plotting_functions as mypl
import import_export as myie











#%%


"""We want to cycle on some values of the DE eos eq. parameters and solve for each
value the pert. eq. Then plot all the graphs."""
"""We vary the parameter corresponding to var_par in params from it's value minus 
var_perc to plus var_perc in n_samples """



var_par="trans_steepness" # can be = 'w_i';'w_f';'trans_steepness';'trans_z';
init_value=-0.8
final_value=-1
n_samples=2

"""Print out the other parameters to double check their values."""
for key,value in params.dark_energy_parameters.items():
    print(f"{key}: {value}")
    
#%%

"""Checks: if the number of samples is choosen as zero, we solve 
for only one var_par value """
if n_samples>0:
    varied_par_values=np.linspace(init_value,final_value,n_samples)
    
    # varied_par_values=[0.3,1.2,3,5]
    varied_par_values=[0.1,1,2,10]
    n_samples=len(varied_par_values)
else:
    raise Exception("The number of samples must be greater than zero")

"""We save the original value of the varied parameter, then
we change the parameter values in the params namespace so
that all moduels notify the change."""
exec("reset_param_value=params."+var_par)#Save the original inital value to reset it

#%%
params_names={"w_i":"w_i",
                   "w_f":"w_f",
                   'trans_z':"z_t",
                   'trans_steepness':"\Gamma"}

phi_tilde_numerical_a=[]
approx_phi_tilde_numerical_a=[]
potential_tilde_numerical_a=[]
approx_potential_tilde_numerical_a=[]
potential_tilde_numerical_phi=[]
approx_potential_tilde_numerical_phi=[]

scale_param_values=np.linspace(params.a_min,params.a_max,500)
dark_energy_eos=[]

legend=[]
for i in range(len(varied_par_values)):
    exec("params."+var_par+"="+str(varied_par_values[i]))#Update the var_par value across the modules.
    legend.append("$"+params_names[var_par]+"$="+str(round(varied_par_values[i],3)))
                  
    quintessence_sol=quintessence_solver.solve(False)
    
    """Build some intersting quantities matrices to plot: H(a),w(a),dark_density_evolution(a),omega_m(a)..."""
    phi_tilde_numerical_a.append(np.array(quintessence_sol.phi_tilde_numerical_a))
    approx_phi_tilde_numerical_a.append(np.array(quintessence_sol.approx_phi_tilde_numerical_a))
    potential_tilde_numerical_a.append(np.array(quintessence_sol.potential_tilde_numerical_a))
    approx_potential_tilde_numerical_a.append(np.array(quintessence_sol.approx_potential_tilde_numerical_a))
    potential_tilde_numerical_phi.append(np.array(quintessence_sol.potential_tilde_numerical_phi))
    approx_potential_tilde_numerical_phi.append(np.array(quintessence_sol.approx_potential_tilde_numerical_phi))
    
    dark_energy_eos.append(np.array([scale_param_values,params.de_eos_a(scale_param_values)]))
    
    """Print a table with some informations to keep track where is the cycle at."""
    labels=["Var_par:"+var_par,"t_0","q_0"]

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
"""PLOTTING """
save=False#Save the plots?
this_run_specifier_0="quintessence_potential"
this_run_specifier_1=params.dark_energy_eos_selected
this_run_specifier_2=var_par


# save_data_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/"
save_plot_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/plots/"+this_run_specifier_2+"/"
#%%



"""DARK ENERGY EOS """
mypl.m_plot(dark_energy_eos,
            xlabel="$a$",
            ylabel="$w$",
            title="",
            legend=legend,
            save_plot_dir=save_plot_to_path,
            save=save,name="de_eos",
            xscale="linear",yscale="linear",
            xlim=None,ylim=None, dotted=False)
pl.show()

"""SCALAR FIELD TILDE of a """
mypl.m_plot(phi_tilde_numerical_a,
            xlabel="$a$",
            ylabel="$\~{\phi} (a)$",
            title="",
            legend=legend,
            save_plot_dir=save_plot_to_path,
            save=save,name="scalar_field_tilde_a",
            xscale="linear",yscale="linear",
            xlim=None,ylim=None, dotted=False)
pl.show()

    
"""POTENTIAL TILDE of phi"""
mypl.m_plot(potential_tilde_numerical_phi,
            xlabel="$\~{\phi}$",
            ylabel="$\~{V}(\~{\phi})$",
            title="",
            legend=legend,
            save_plot_dir=save_plot_to_path,
            save=save,name="potential_tilde_phi",xscale="linear",yscale="log",
            xlim=None,ylim=None, dotted=False)
pl.show()

# """POTENTIAL TILDE of phi"""
# mypl.m_plot(approx_potential_tilde_numerical_phi,
#             xlabel="$\~{\phi}$",
#             ylabel="$\~{V}(\~{\phi})$",
#             title="Approximate $\~{V}(\~{\phi})$",
#             legend=legend,
#             # legend=("$\~{V}(\~{\phi})$","approx $\~{V}(\~{\phi})$","$\~{V}(\~{\phi})_m$"),
#             # func_to_compare=lambda x :1.05*x/x,
#             # func_to_compare=lambda x :omega_dark_now*exp(-3*x/sqrt(omega_dark_now)),
#             save=save,name="potential_tilde_phi_detail",yscale="log",
#             xlim=(1.2,1.5),ylim=(0,5), 
#             dotted=False)
# pl.show()


# """POTENTIAL TILDE of a"""
# mypl.m_plot(approx_phi_tilde_numerical_a,
#             xlabel="$a$",
#             ylabel="$\~{V}(a)$",
#             title="Approxl $\~{\phi}(a)$",
#             legend=legend,
#             # func_to_compare=lambda a :omega_dark_now/(a**3),
#             save_plot_dir=save_plot_to_path,
#             save=save,name="potential_tilde_a",
#             xscale="linear",
#             xlim=None,ylim=None, dotted=False)
# pl.show()

if save:
    mypl.save_run_parameters(save_plot_to_path)