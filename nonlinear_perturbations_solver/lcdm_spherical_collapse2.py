# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:58:19 2023

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
import matplotlib.pyplot as pl

import winsound


import scipy as sp
import matplotlib.pyplot as pl

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func
import constants as const

sys.path.append("../utility_modules")
import numerical_methods as mynm
import plotting_functions as mypl
import lcdm_model as lcdm
import import_export as myie

sys.path.append("../friedmann_solver")
import friedmann_solver as fr_sol


#%%

params.omega_dark_now=0
params.omega_matter_now=1

def linear_contrast(a):
    return  1.68647019984
def virial_overdensity(a):
    return 146.841245384
def virial_overdensity_star(a):
    return 177.65287922
def turn_aroun_overdensity(a):
    return 5.551652
#%%
def infty_minus_nonlinear_delta_c(linear_delta_coll,a_coll,numerical_infty=1e8):
    """Difference between the maxium delta reached and the imposed numerical infinity,
    the goal is to minimize this difference at a_collapse"""
    res=solve(linear_delta_coll,a_coll)
    difference=numerical_infty-res[1][-1]
    return [difference,res]

def find_delta_collapse_star(collapse_scales_a,a=1.68,b=1.7,exact_decimals=4):
    """Input: 
        -collapse_scales_a: can be a number or an array with elements scale parameters.
        -a,b: define the interval [a,b] where the solution is found through bisection
        -exact_decimals: exat decimals of the linear overdensity at collapse"""
    return_scalar=False
    try:
        len(collapse_scales_a)
    except:
        collapse_scales_a=[collapse_scales_a]
        return_scalar=True
        
        
    delta_coll_stars=[]
    density_contrasts=[]
    for i in collapse_scales_a:
        aux=mynm.bisection_(infty_minus_nonlinear_delta_c,a,b,10**(-exact_decimals-1),i)
        density_contrasts.append(aux[1])
        aux=round(aux[0],exact_decimals+1)
        delta_coll_stars.append(aux)
        print("a_c=",round(i,4),"     delta_c_star=",round(aux,exact_decimals+1))
        
    if return_scalar:
        return [delta_coll_stars[0],density_contrasts]
    else:
        return [delta_coll_stars,density_contrasts]
    
def virialization_radius(zeta_ta,a_ta,a_c):
    eta_t=2/zeta_ta*lcdm.omega_dark_a(a_ta)/lcdm.omega_matter_a(a_ta)
    eta_v=2/zeta_ta*(a_ta/a_c)**3*lcdm.omega_dark_a(a_c)/lcdm.omega_matter_a(a_c)
    x_vir=(1-eta_v/2)/(2+eta_t-3*eta_v/2)
    return x_vir
    
def find_virialization_scale(nonlinear_density_contrast_a,zeta_ta,a_ta,a_c):
    
    x_vir=virialization_radius(zeta_ta,a_ta,a_c)
    def fun_to_find_zeros(a):
        fun_to_find_zeros=x_vir-zeta_ta**(1/3)*a/(a_ta*(1+nonlinear_density_contrast_a(a))**(1/3))
        return fun_to_find_zeros
    
    a_vir=mynm.bisection(fun_to_find_zeros, a_ta, 1, tol=1e-9)
    return [a_vir,x_vir]
    
    
def find_turn_around_scale(nonlinear_density_contrast_a,a_coll,tol=1e-6):
   
    def fun_to_minimize(a):
        fun_to_minimize=-a/(1+nonlinear_density_contrast_a(a))**(1/3)
        return fun_to_minimize
    
    bracket_interval=[1e-5,(0.5)**(2/3)*a_coll,a_coll]
    optimize_result=sp.optimize.minimize_scalar(fun_to_minimize,bracket=bracket_interval,  tol=tol)
    a_ta=round(optimize_result.x,int(-np.log10(tol))+1)
    return a_ta
    
#%%
"""Definition of the ODE system to solve: """
def non_lin_pert_eq(a,y):
    fun=[0,0]
    """Matter perturbations"""
    fun[0] = y[1]
    """LCDM"""
    fun[1] = -3/(2*a)*(1-lcdm.effective_eos_a(a))*y[1] + 4/3*y[1]**2/(1+y[0]) \
        +3/(2*a**2)*lcdm.omega_matter_a(a)*y[0]*(1+y[0])  
    return np.array(fun)



def lin_pert_eq(a,y):
    fun=[0,0]
    """Matter perturbations"""
    fun[0] = y[1]
    """LCDM"""
    fun[1] = -3/(2*a)*(1-lcdm.effective_eos_a(a))*y[1] \
        +3/(2*a**2)*lcdm.omega_matter_a(a)*y[0]
    return np.array(fun)


#%%
def solve(delta_coll_star,a_coll):
    """Defining the integration parameters"""
    atol=1e-9
    rtol=1e-8

    t_min= 1e-5
    t_max= a_coll
    n=round(0.25*(-1+np.sqrt(24*lcdm.omega_matter_a(t_min)+(1-3*lcdm.effective_eos_a(t_min))**2)+3*lcdm.effective_eos_a(t_min)),4)
    
    delta_m_tls=delta_coll_star*(t_min/a_coll)**n
    d_delta_m_tls=n*delta_m_tls/t_min
    
    init_cond=[delta_m_tls,d_delta_m_tls]# [Position(t=t_min),Speed(t_tmin)]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(non_lin_pert_eq,t_span=(t_min,t_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    lcdm_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    
    return lcdm_matter_density_contrast_numerical_a



def overdensity_at_ta(delta_coll_star,a_coll,a_ta):
    """Defining the integration parameters"""
    atol=1e-7
    rtol=1e-6

    t_min= 1e-5
    t_max= a_ta
    n=round(0.25*(-1+np.sqrt(24*lcdm.omega_matter_a(t_min)+(1-3*lcdm.effective_eos_a(t_min))**2)+3*lcdm.effective_eos_a(t_min)),4)
    
    delta_m_tls=delta_coll_star*(t_min/a_coll)**n
    d_delta_m_tls=n*delta_m_tls/t_min
    
    init_cond=[delta_m_tls,d_delta_m_tls]# [Position(t=t_min),Speed(t_tmin)]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(non_lin_pert_eq,t_span=(t_min,t_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    # lcdm_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    
    return rk4_result.y[0][-1]



def lin_extrap_coll_contrast(delta_coll_star,a_coll):
    """Defining the integration parameters"""
    atol=1e-12
    rtol=1e-13

    t_min= 1e-5
    t_max= a_coll
    n=round(0.25*(-1+np.sqrt(24*lcdm.omega_matter_a(t_min)+(1-3*lcdm.effective_eos_a(t_min))**2)+3*lcdm.effective_eos_a(t_min)),4)

    delta_m_tls=delta_coll_star*(t_min/a_coll)**n
    d_delta_m_tls=n*delta_m_tls/t_min

    init_cond=[delta_m_tls,d_delta_m_tls]# [Position(t=t_min),Speed(t_tmin)]
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(lin_pert_eq,t_span=(t_min,t_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    # linear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    return rk4_result.y[0][-1]



#%%


collapse_scales_z=np.linspace(0,10,100)
collapse_scales_a=cosm_func.a_z(collapse_scales_z)

# collapse_scales_a=np.linspace(1,0.024390243902439025,10)
# collapse_scales_z=cosm_func.z_a(collapse_scales_a)

aux=find_delta_collapse_star(collapse_scales_a)
delta_c_stars=aux[0]
nonlinear_density_contrasts=aux[1]

linear_overdensity_at_collapse=[]
for i in range(len(delta_c_stars)):
    aux=lin_extrap_coll_contrast(delta_c_stars[i],collapse_scales_a[i])
    print("delta_c: ",round(aux,5))
    linear_overdensity_at_collapse.append(aux)


virialization_overdensities=[]
virialization_overdensities_stars=[]
turn_around_overdensities=[]
for i in range(len(delta_c_stars)):
    aux=nonlinear_density_contrasts[i]
    nonlinear_density_contrast_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)
    
    turn_around_scale=find_turn_around_scale(nonlinear_density_contrast_a,collapse_scales_a[i])
    turn_around_overdensity=np.round(nonlinear_density_contrast_a(turn_around_scale),4)+1
    turn_around_overdensities.append(turn_around_overdensity)
    
    aux=find_virialization_scale(nonlinear_density_contrast_a,turn_around_overdensity, turn_around_scale, collapse_scales_a[i])
    virialization_scale=aux[0]
    x_vir=aux[1]
    virialization_overdensity=np.round(nonlinear_density_contrast_a(virialization_scale),4)+1
    virialization_overdensities.append(virialization_overdensity)
    # print("32=",round((collapse_scales_a[i]/(turn_around_scale*x_vir))**3,3))
    # print("a_c=",round(collapse_scales_a[i],3),"    a_ta=",round(turn_around_scale/collapse_scales_a[i],5),"    x_vir=",round(x_vir,3))
    # print((collapse_scales_a[i]/(turn_around_scale*x_vir))**3,"*",\
    #       turn_around_overdensity,"=",turn_around_overdensity*(collapse_scales_a[i]/(turn_around_scale*x_vir))**3)
    virialization_overdensities_stars.append(turn_around_overdensity*(collapse_scales_a[i]/(turn_around_scale*x_vir))**3)


myie.save_to_txt_twocolumns([collapse_scales_z,linear_overdensity_at_collapse], "delta_c_backgr_de")
myie.save_to_txt_twocolumns([collapse_scales_z,virialization_overdensities_stars], "zeta_vir_star_backgr_de")
myie.save_to_txt_multicolumn(growth_factors,"growth_factors_backgr_de")


#%%

save=False



for i in range(len(lcdm_overdensity_at_virialization_star[0])):
    lcdm_overdensity_at_virialization_star[1][i]=lcdm_overdensity_at_virialization_starss[1][i]*\
        lcdm.omega_matter_a(cosm_func.a_z(lcdm_overdensity_at_virialization_star[0][i]))
    print(cosm_func.a_z(lcdm_overdensity_at_virialization_star[0][i]))



"""Matter overdensity at virialization star"""
mypl.m_plot([lcdm_overdensity_at_virialization_star,eds_overdensity_at_virialization_star],
            title="Virialization overdenisty*",
            # title="EdS virialization overdenisty*",
            
            xlabel="$z_c$", ylabel="$\Omega_m\zeta_{vir}^*$",
            # xscale="log",
            # func_to_compare=virial_overdensity_star,
            legend=("$\Lambda CDM$","EdS","177.653"),
            # legend=("$EdS$","177.653"),
            ylim=(100,200),
            # ylim=(177.55,177.7),
            save=True,
            dotted=True,
            name="matter_overdensity_at_virialization_star_LCDM")
pl.show()

"""Linearly extrapoleted matter overdensity at collapse"""
mypl.m_plot([lcdm_linear_density_contrast_at_collapse,eds_linear_density_contrast_at_collapse],
            title="$\Lambda CDM$ linear contrast at collapse",
            # title="EdS linear contrast at collapse",
            
            xlabel="$z_c$", ylabel="$\delta_c$",
            # xscale="log",
            # func_to_compare=linear_contrast,
            legend=("$\Lambda CDM$","EdS","1.6864"),
            # legend=("EdS","1.6864"),
            
            # ylim=(1.686,1.687),
            ylim=(1.67,1.689),
            
            save=True,
            dotted=True,
            name="linear_contrast_at_collapse_LCDM")
pl.show()












"""Linearly extrapoleted matter overdensity at collapse"""
mypl.m_plot([collapse_scales_z,linear_overdensity_at_collapse],
            title="$\Lambda CDM$ linear contrast at collapse",
            # title="EdS linear contrast at collapse",
            
            xlabel="$z_c$", ylabel="$\delta_c$",
            # xscale="log",
            func_to_compare=linear_contrast,
            legend=("$\Lambda CDM$","1.6864"),
            # legend=("EdS","1.6864"),
            
            # ylim=(1.686,1.687),
            ylim=(1.67,1.689),
            
            save=save,
            dotted=True,
            name="linear_contrast_at_collapse_LCDM")
pl.show()

"""Matter overdensity at virialization"""
mypl.m_plot([collapse_scales_z,virialization_overdensities],
            title="$\Lambda CDM$ virialization overdenisty",
            # title="EdS virialization overdenisty",
            
            xlabel="$z_c$", ylabel="$\zeta_{vir}$",
            # xscale="log",
            func_to_compare=virial_overdensity,
            # func_to_compare=turn_aroun_overdensity,
            
            legend=("$\Lambda CDM$","146.841"),
            # legend=("EdS","146.841"),

            ylim=(130,270),
            # ylim=(146.838,146.845),
            # ylim=(5.52,5.6),
            save=save,
            dotted=True,
            name="matter_overdensity_at_virialization_LCDM")
pl.show()

"""Matter overdensity at virialization star"""
mypl.m_plot([collapse_scales_z,virialization_overdensities_stars],
            title="$\Lambda CDM$ virialization overdenisty*",
            # title="EdS virialization overdenisty*",
            
            xlabel="$z_c$", ylabel="$\zeta_{vir}^*$",
            # xscale="log",
            func_to_compare=virial_overdensity_star,
            legend=("$\Lambda CDM$","177.653"),
            # legend=("$EdS$","177.653"),
            ylim=(150,345),
            # ylim=(177.55,177.7),
            save=save,
            dotted=True,
            name="matter_overdensity_at_virialization_star_LCDM")
pl.show()





# xticks=list(np.round(collapse_scales_a,3))
# mypl.m_plot(nonlinear_density_contrasts,
#             # func_to_compare=lambda x:2.15083*x,
#             legend=None,#("$\Lambda CDM$ non-lin","EdS lin"),
#             xlabel="a",
#             ylabel="$\delta_m$",
#             title="Nonlinear $\Lambda CDM$ density contrast",
#             # xscale="log",
#             yscale="log",
#             # ylim=(0,1e8),
#             dotted=False,
#             x_ticks=xticks,
#             x_ticklables=xticks,
#             name="nonlinear_density_contrast_lcdm",
#             save=save)
# pl.show()

winsound.Beep(440, 1000)
winsound.Beep(880,1000)
winsound.Beep(1760,1000)


