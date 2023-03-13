# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:32:41 2023

@author: sebas
"""
"""Class to store the results of the friedmann solver"""
class friedmann_sol:
    def __init__(self):
        self.dark_density_evolution_numerical_a=[]
        self.dark_density_parameter_numerical_a=[]
        self.appox_dark_density_evolution_numerical_a=[]
        self.de_eos_numerical_a=[]
        
        self.scale_parameter_numerical_t=[]
        self.hubble_function_numerical_t=[]
        self.rescaled_hubble_functions_numerical_a=[]

        self.matter_density_parameter_numerical_a=[]
        self.rad_density_numerical_a=[]
        
        self.effective_eos_numerical_a=[]
        
        self.universe_age=None
        self.deceleration_parameter=None
        # self.time_at_eq=None
        pass
    
    
    
"""Class to store the quintessence results"""
class quintessence_sol:
    def __init__(self):
        self.phi_tilde_numerical_a=[]
        self.approx_phi_tilde_numerical_a=[]
        self.potential_tilde_numerical_a=[]
        self.approx_potential_tilde_numerical_a=[]
        self.potential_tilde_numerical_phi=[]
        self.approx_potential_tilde_numerical_phi=[]
        pass