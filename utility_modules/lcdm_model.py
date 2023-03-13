# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:30:47 2022

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


from matplotlib.pyplot import plot
import matplotlib.pyplot as pl


import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func
import constants as const



def dark_density_evolution_a(a):
    dark_density_evolution_a=params.omega_dark_now
    return dark_density_evolution_a

def matter_density_evolution_a(a):
    matter_density_evolution_a=params.omega_matter_now/a**3
    return matter_density_evolution_a

def omega_dark_a(a):
    omega_dark_a=dark_density_evolution_a(a)/(params.omega_matter_now/a**3+params.omega_dark_now)
    return omega_dark_a
def omega_matter_a(a):
    omega_matter_a=matter_density_evolution_a(a)/(params.omega_matter_now/a**3+params.omega_dark_now)
    return omega_matter_a

def effective_eos_a(a):
    effective_eos_a=-params.omega_dark_now/(params.omega_matter_now/a**3+params.omega_dark_now)
    return effective_eos_a

class LCDM:
    """Initialization of the LCDM parameters"""
    def __init__(self,model_parameters=[params.omega_matter_now,params.hubble_constant,params.scale_at_lss]):
        self.omega_matter_now = model_parameters[0]
        self.hubble_constant_std = model_parameters[1]
        self.scale_at_ls=model_parameters[2]    
        """Derived parameters:"""
        self.omega_dark_now = 1-self.omega_matter_now
        self.hubble_constant = self.hubble_constant_std*1e3*const.Gy/const.Mpc# Gy-1

        self.w=-1
    """Important functions of the class:"""
    def cosmological_constant_eos_a(self,a):
        cosmological_constant_eos_a=self.w*a/a
        return cosmological_constant_eos_a
        
    def dark_density_evolution_a(self,a):
        dark_density_evolution=self.omega_dark_now*a**(-3*(1+self.w))
        return dark_density_evolution
    
    def matter_density_evolution_a(self,a):
        matter_density_evolution=self.omega_matter_now/a**3
        return matter_density_evolution
    
    def rescaled_hubble_function_a(self,a):
        rescaled_hubble_function_a=sqrt(self.dark_density_evolution_a(a)+self.matter_density_evolution_a(a))
        return rescaled_hubble_function_a
        
    def omega_dark_a(self,a):
        omega_dark=self.dark_density_evolution_a(a)/self.rescaled_hubble_function_a(a)**2
        return omega_dark
    
    def omega_matter_a(self,a):
        omega_matter=self.matter_density_evolution_a(a)/self.rescaled_hubble_function_a(a)**2
        return omega_matter
    
    def effective_eos_a(self,a):
        effective_eos=-self.omega_dark_now/(self.omega_matter_now/a**3+self.omega_dark_now)
        return effective_eos
        
    
    
        
    def compute_perturbations(self):
        
        def eff_eos(a):
            eff_eos=self.omega_dark_a(a)*self.w/(self.omega_matter_a(a)+self.omega_dark_a(a))
            return eff_eos
        
    
        """Definition of the ODE system to solve: """
        def fun(a,y):
            fun=[0,0]
            fun[0] = y[1]
            fun[1] = -3/(2*a)*(1-self.effective_eos_a(a))*y[1] \
                +3/(2*a**2)*self.omega_matter_a(a)*y[0]
            return np.array(fun)
        atol=1e-9
        rtol=1e-10
        a_min= self.scale_at_ls#We start at last scattering 
        a_max= 1 
        delta_tls=self.scale_at_ls
        d_delta_tls=1
        init_cond=[delta_tls,d_delta_tls]

        """Perform the integration """
        rk4_result=sp.integrate.solve_ivp(fun,t_span=(a_min,a_max), y0=init_cond, 
                                          method="RK45",atol=atol,rtol=rtol)

        """Extract the needed informations from the rk4 result"""
        matter_density_contrast_numerical=np.array([list(rk4_result.t),list(rk4_result.y[0])])
        return matter_density_contrast_numerical
            
