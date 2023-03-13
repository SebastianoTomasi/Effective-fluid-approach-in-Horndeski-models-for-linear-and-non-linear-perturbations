# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:00:02 2023

@author: sebas
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl

import sys
sys.path.append("../utility_modules")
import plotting_functions as mypl



#%%

"""Definition of the ODE system to solve: """
def fun(a,y):
    fun=[0]*2
    """Matter perturbations"""
    fun[0] = y[1]
    fun[1] = -3/(2*a)*y[1] + 4/3*y[1]**2/(1+y[0]) \
        +3/(2*a**2)*(1+y[0])*y[0]
    return np.array(fun)


#%%
def solve():
    """Defining the integration parameters"""
    atol=1e-12
    rtol=1e-13
    t_min= 1/1090#We start at last scattering 
    t_max= 1#1.585 at vir

    
    delta_m_tls=1/1090
    d_delta_m_tls=1
    
    init_cond=[delta_m_tls,d_delta_m_tls]# [Position(t=t_min),Speed(t_tmin)]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(fun,t_span=(t_min,t_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    eds_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    return eds_matter_density_contrast_numerical_a

#%%

# res=solve()
# eds_matter_density_contrast_numerical_a=res

#%%

    
# mypl.m_plot(eds_matter_density_contrast_numerical_a,
#             func_to_compare=lambda x:x,
#             legend=("matter","de","matterEdS"),
#             # xscale="log",yscale="log",
#             dotted=False,ylim=None)
# pl.show()

    
