# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:36:38 2022

@author: sebas
"""
import numpy as np
from numpy import sqrt


import scipy as sp



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

"""We need to declare it here again because if not, it will use the parameter values given in simulation_parameters.
We want to use them, but not when we are cycling over the value of a parameter """


"""Create a quintessence solution object to store the solutions"""
result=data_classes.quintessence_sol()
def solve(compute_approximations=False):
    
    """Compute the dark density evolution"""
    dark_density_evolution_numerical_a=cosm_func.solve_dark_density_evolution_numerical_a()
    scale_parameter_values=dark_density_evolution_numerical_a[0]
    global dark_density_evolution_a
    dark_density_evolution_a=sp.interpolate.interp1d(scale_parameter_values, dark_density_evolution_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    
    
    """Evaluate the potential on the scale parameter values"""
    potential_tilde_numerical_a=np.array([scale_parameter_values,potential_tilde_a(scale_parameter_values)])
    
    """Compute the integral for phi_tilde"""
    phi_tilde_numerical_a=mynm.integrate(phi_tilde_integrand_a,
                                         params.a_max,params.a_min,
                                         n=params.n)
    
    """Translate phi to make it positive"""
    phi_tilde_numerical_a[1]=phi_tilde_numerical_a[1]-np.min(phi_tilde_numerical_a[1])

    """Compute V(phi) by inverting phi(a)->a(phi) => V(a)->V(a(phi))=V(phi):"""
    potential_tilde_numerical_phi=np.array([phi_tilde_numerical_a[1],potential_tilde_numerical_a[1]])
    
    """Approximation of the eos with a step function: """
    if compute_approximations:        
        """ Compute the approximate integral for phi_tilde"""
        approx_phi_tilde_numerical_a=mynm.integrate(approx_phi_tilde_integrand_a,
                                                    params.a_max,params.a_min,
                                                    params.n)
        approx_phi_tilde_numerical_a[1]=approx_phi_tilde_numerical_a[1]-np.min(approx_phi_tilde_numerical_a[1])
        
        """Approx. for the potential"""
        aux=[]
        for i in scale_parameter_values:
            aux.append(approx_potential_tilde_a(i))
        approx_potential_tilde_numerical_a=np.array([scale_parameter_values,aux])
        
        """Invert the approximate phi(a) and obtain the approx. V(phi)"""
        approx_potential_tilde_numerical_phi=np.array([approx_phi_tilde_numerical_a[1],approx_potential_tilde_numerical_a[1]])
        
        """Save the results to return them"""
        result.approx_phi_tilde_numerical_a=approx_phi_tilde_numerical_a
        result.approx_potential_tilde_numerical_a=approx_potential_tilde_numerical_a
        result.approx_potential_tilde_numerical_phi=approx_potential_tilde_numerical_phi
    
    
    
    
    """Convert to physical units to get V and phi without tilde"""
    # phi_numerical_a=[scale_parameter_values,0]
    # potential_numerical_a=[scale_parameter_values,0]
    # approx_potential_numerical_a=[scale_parameter_values,0]
    # phi_numerical_a[1]=phi_tilde_numerical_a[1]*sqrt(3/(8*pi*G))
    # potential_numerical_a[1]=potential_tilde_numerical_a[1]*3*(hubble_constant/Gy)**2/(16*pi*G)
    # approx_potential_numerical_a[1]=approx_potential_tilde_numerical_a[1]*3*(hubble_constant/Gy)**2/(16*pi*G)
    
    """Save the results and return tham"""
    result.phi_tilde_numerical_a=phi_tilde_numerical_a
    result.potential_tilde_numerical_a=potential_tilde_numerical_a
    result.potential_tilde_numerical_phi=potential_tilde_numerical_phi
    

    
    
    return result
#%%
"""Definitions of some functions """


def potential_tilde_a(a):
    potential_a=dark_density_evolution_a(a)*(1-params.de_eos_a(a))
    return potential_a



def phi_tilde_integrand_a(a):
    w=params.de_eos_a(a)
    if w<-1:
        raise Exception("The program only deals with quintessence, hence -1<w<1")
    phi_tilde_integrand_a=sqrt((a*dark_density_evolution_a(a)*(1+w))/ \
                               (params.omega_matter_now+a**3*dark_density_evolution_a(a)+params.omega_rad_now/a))
    return phi_tilde_integrand_a

"""Step approximation"""
def approx_potential_tilde_a(a):
    a_t=1/(1+params.trans_z)
    if a_t>1:
        raise Exception("z_t must be >=0")
    if a>=a_t:
        return params.omega_dark_now*a**(-3*(1+params.w_f))*(1-params.w_f)
    if a<a_t:
        return  params.omega_dark_now*a**(-3*(1+params.w_i))*a_t**(-3*(params.w_f-params.w_i))*(1-params.w_i)

def approx_phi_tilde_integrand_a(a):
    w=cosm_func.step_eos(a)
    if w<-1:
        raise Exception("The program only deals with quintessence, hence -1<w<1")
    approx_phi_tilde_integrand_a=sqrt((a*cosm_func.step_dark_density_evolution_a(a)*(1+w))/ \
                                (params.omega_matter_now+a**3*cosm_func.step_dark_density_evolution_a(a)+params.omega_rad_now/a))
    return approx_phi_tilde_integrand_a


# x=np.linspace(1e-5,1,500)
# y0=[]
# y1=[]
# for i in x:
#     y0.append(approx_phi_tilde_integrand_a(i))
#     y1.append(phi_tilde_integrand_a(i))
# mypl.m_plot([[x,y1],[x,y0]],yscale="linear",xlim=(0.2,1),ylim=(-1,2))

#%%
"""PLOTTING"""

# res=solve()

# save = True

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




# """POTENTIAL TILDE of a"""
# mypl.m_plot([res.potential_tilde_numerical_a,res.approx_potential_tilde_numerical_a],
#             xlabel="$a$",
#             ylabel="$\~{V}(a)$",
#             title="Scalar field potential $\~{V}(a)$",
#             legend=("True","Step approx."),
#             # func_to_compare=lambda a :omega_dark_now/(a**3),
#             save=save,name="potential_tilde_a",xscale="linear",yscale="log",
#             xlim=None,ylim=None, dotted=False)
# pl.show()



# """SCALAR FIELD TILDE of a """
# mypl.m_plot([res.phi_tilde_numerical_a,res.approx_phi_tilde_numerical_a],
#             xlabel="$a$",
#             ylabel="$\~{\phi} (a)$",
#             title="Scalar field $\~{\phi} (a)$",
#             # legend=("$\~{\phi}$","$\~{\phi}_m$"),
#             legend=("True","Step approx."),
#             # func_to_compare=lambda a :sqrt(omega_dark_now)*log(a),
#             save=save,name="scalar_field_tilde_a",xscale="linear",yscale="linear",
#             xlim=None,ylim=(-3,0.5), dotted=False)
# pl.show()

    

# """POTENTIAL TILDE of phi"""
# mypl.m_plot([res.potential_tilde_numerical_phi,res.approx_potential_tilde_numerical_phi],
#             xlabel="$\~{\phi}$",
#             ylabel="$\~{V}(\~{\phi})$",
#             title="Scalar field potential $\~{V}(\~{\phi})$",
#             legend=("True","Step approx."),
#             # legend=("$\~{V}(\~{\phi})$","approx $\~{V}(\~{\phi})$","$\~{V}(\~{\phi})_m$"),
#             # func_to_compare=lambda x :1.05*x/x,
#             # func_to_compare=lambda x :omega_dark_now*exp(-3*x/sqrt(omega_dark_now)),
#             save=save,name="potential_tilde_phi",xscale="linear",yscale="log",
#             xlim=None,ylim=None, dotted=False)
# pl.show()

# mypl.save_run_parameters()
