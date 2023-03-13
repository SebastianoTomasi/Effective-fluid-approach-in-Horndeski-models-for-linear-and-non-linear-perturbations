# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:39:15 2023

@author: Sebastiano Tomasi
"""

import numpy as np
import scipy as sp

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func

sys.path.append("../utility_modules")
import numerical_methods as mynm
import import_export as myie


#%%



save_to_txt=False
import_from_txt=False

#%%
"""Definition of some useful functions."""

def virialization_radius(zeta_ta,a_ta,a_c):
    """Implementation of the formula to find the virialization radius"""
    eta_t=2/zeta_ta*omega_dark_a(a_ta)/omega_matter_a(a_ta)
    eta_v=2/zeta_ta*(a_ta/a_c)**3*omega_dark_a(a_c)/omega_matter_a(a_c)
    x_vir=(1-eta_v/2)/(2+eta_t-3*eta_v/2)
    return x_vir
    
def find_virialization_scale(nonlinear_density_contrast_a,zeta_ta,a_ta,a_c):
    """
    Input:
        -nonlinear_density_contrast_a: callable
        -zeta_ta: overdensity at turn around
        -a_ta: scale parameter at turn around
        -a_c: scale parameter at collapse
    Output:
        -[a_vir,x_vir]: scale parameter at virialization and sphere radius at virialization.
    """
    x_vir=virialization_radius(zeta_ta,a_ta,a_c)
    def fun_to_find_zeros(a):
        fun_to_find_zeros=x_vir-zeta_ta**(1/3)*a/(a_ta*(1+nonlinear_density_contrast_a(a))**(1/3))
        return fun_to_find_zeros
    
    a_vir=mynm.bisection(fun_to_find_zeros, a_ta, 1, tol=1e-9)
    return [a_vir,x_vir]

def infty_minus_nonlinear_delta_c(delta_c_star,a_coll,numerical_infty=1e8):
    """Difference between the maxium nonlinear matter density contrast reached and the numerical infinity,
    the goal is to minimize this difference at a_collapse.
    Input:
        -delta_c_star: the initial condition.
        -a_coll: scale parameter at which we want the collapse to happens.
        -numerical_infty: numerical infinity to define the divergence at collapse.
    Output:
        [difference,nonlinear_density_contrast]"""
    nonlinear_density_contrast=solve_nonlinear_perturbations(delta_c_star,a_coll)
    difference=numerical_infty-nonlinear_density_contrast[1][-1]
    return [difference,nonlinear_density_contrast]

def find_delta_collapse_star(collapse_scales_a,a=1.68,b=3.3,exact_decimals=4):
    """
    Finds the initial conditions delta_c^* that result in the sphere collapsing at a_c
    Input: 
        -collapse_scales_a: can be a number or an array of scale parameter values in the range [0,1].
        -a,b: define the interval [a,b] where the solution is found through the bisection alghorithm
        -exact_decimals: exat decimals of the initial condition delta_c^* 
    Output:
        [delta_coll_stars,density_contrast] where density_contrast is a list that contains all the solutions
            returned by infty_minus_nonlinear_delta_c"""
    """Check if only 1 value is given for collapse_scales_a"""
    return_scalar=False
    try:
        len(collapse_scales_a)
    except:
        collapse_scales_a=[collapse_scales_a]
        return_scalar=True
        
        
    delta_coll_stars=[]
    density_contrasts=[]
    for i in collapse_scales_a:
        delta_coll_star,density_contrast=mynm.bisection_(infty_minus_nonlinear_delta_c,a,b,10**(-exact_decimals-1),i)
        
        """We save the nonlinear matter density contrast solutions to return them."""
        density_contrasts.append(density_contrast)
        
        delta_coll_star=round(delta_coll_star,exact_decimals+1)
        delta_coll_stars.append(delta_coll_star)
        print("a_c=",round(i,4),"     delta_c_star=",round(delta_coll_star,exact_decimals+1))
        
    if return_scalar:
        return [delta_coll_stars[0],density_contrasts]
    else:
        return [delta_coll_stars,density_contrasts]
    
    
def find_turn_around_scale(nonlinear_density_contrast_a,a_coll,tol=1e-6):
    
    """Find the turn around scale by minimizing -x, where x is the radius of the sphere.
    Input:
       -nonlinear_density_contrast_a: callable
       -a_coll: scale parameter at collapse
       -tol: tolerance used in the minimization alghoritm
    Output:
       a_ta: scale parameter at turn around
       """
    def fun_to_minimize(a):
        """This function is proportional minus the sphere radius"""
        fun_to_minimize=-a/(1+nonlinear_density_contrast_a(a))**(1/3)
        return fun_to_minimize
    
    bracket_interval=[1e-5,(0.5)**(2/3)*a_coll,a_coll]
    optimize_result=sp.optimize.minimize_scalar(fun_to_minimize,bracket=bracket_interval,  tol=tol)
    a_ta=round(optimize_result.x,int(-np.log10(tol))+1)
    return a_ta


#%%
"""Definition of the ODE system to solve: """

def nonlinear_density_contrast_eq(a,y):# Equations from Pace: https://arxiv.org/pdf/1612.03018.pdf
    """This function defines the NONLINEAR matter density contrast differential equation whene dark energy perturbations are set to zero,
    but the effects of dark energy are still considered in the background.
    y[0]=\delta_m
    y[1]=\theta 
    y[2]=\delta_de
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\theta'
    fun[2]=delta_de'
    """
    fun=[0]*3
    fun[0] = -(1+y[0])*y[1]/a
    fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-y[1]**2/(3*a)-3/(2*a)*(omega_matter_a(a)*y[0]+omega_dark_a(a)*(1+3*params.c_eff(a))*y[2])
    fun[2] = -3/a*(params.c_eff(a)-params.de_eos_a(a))*y[2]-(1+params.de_eos_a(a)+(1+params.c_eff(a))*y[2])*y[1]/a
    
    if y[2]<-1:#Fisicamente non si può andare sotto -1
        # print("delta_de is < -1!!!")
        y[2]=-1
    return np.array(fun)



def linear_density_contrast_eq(a,y):
    """This function defines the LINEAR matter density contrast differential equation whene dark energy perturbations are set to zero,
    but the effects of dark energy are still considered in the background.
    y[0]=\delta_m
    y[1]=\theta 
    y[2]=\delta_de
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\theta'
    fun[3]=delta_de'
    """
    fun=[0]*3
    fun[0] = -y[1]/a
    fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-3/(2*a)*(omega_matter_a(a)*y[0]+omega_dark_a(a)*(1+3*params.c_eff(a))*y[2])
    fun[2] = -3/a*(params.c_eff(a)-params.de_eos_a(a))*y[2]-(1+params.de_eos_a(a))*y[1]/a
    
    return np.array(fun)



#%%
def solve_nonlinear_perturbations(delta_coll_star,a_coll):
    """Define the rk4 integration parameters"""
    atol=1e-9
    rtol=1e-8

    a_min= 1e-5#Where the integration starts
    a_max= a_coll#Where the integration ends
    n=round(0.25*(-1+np.sqrt(24*omega_matter_a(a_min)+(1-3*effective_eos_a(a_min))**2)+3*effective_eos_a(a_min)),4)#Growth factor exponent
    
    delta_m_ini=delta_coll_star*(a_min/a_coll)**n#Initial perturbation value
    theta_ini=-n*delta_m_ini
    delta_de_ini=n*(1+params.de_eos_a(a_min))*delta_m_ini/(n+3*(params.c_eff(a_min)-params.de_eos_a(a_min)))
    init_cond=[delta_m_ini,theta_ini,delta_de_ini]# Initial conditions
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(nonlinear_density_contrast_eq,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the results from the rk4 data structure"""
    nonlinear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    nonlinear_de_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[2]])
    
    return nonlinear_matter_density_contrast_numerical_a #[nonlinear_matter_density_contrast_numerical_a,nonlinear_de_density_contrast_numerical_a]



def solve_growth_factor(delta_coll_star,a_coll):
    """Defining the integration parameters"""
    atol=1e-7
    rtol=1e-6

    a_min= 1e-5
    a_max= a_coll
    n=round(0.25*(-1+np.sqrt(24*omega_matter_a(a_min)+(1-3*effective_eos_a(a_min))**2)+3*effective_eos_a(a_min)),4)

    delta_m_ini=delta_coll_star*(a_min/a_coll)**n#Initial perturbation value
    theta_ini=-n*delta_m_ini
    delta_de_ini=n*(1+params.de_eos_a(a_min))*delta_m_ini/(n+3*(params.c_eff(a_min)-params.de_eos_a(a_min)))
    init_cond=[delta_m_ini,theta_ini,delta_de_ini]# Initial conditions
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(linear_density_contrast_eq,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=atol,rtol=rtol)
    
    """Extract the needed informations from the rk4 result"""
    linear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    return linear_matter_density_contrast_numerical_a



#%%

def solve(friedmann_solution=None,number_of_points=5):
    """Use the friedmann solver to get the effective eos and the omega matter"""

    aux=friedmann_solution.effective_eos_numerical_a
    global effective_eos_a
    effective_eos_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)

    aux=friedmann_solution.matter_density_parameter_numerical_a
    global omega_matter_a
    omega_matter_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)

    aux=friedmann_solution.dark_density_parameter_numerical_a
    global omega_dark_a
    omega_dark_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)
    
    """Define when we want the collapse to happen"""
    collapse_scales_z=np.linspace(0,10,number_of_points)
    collapse_scales_a=cosm_func.a_z(collapse_scales_z)
    
    # # collapse_scales_a=np.linspace(1,0.024390243902439025,number_of_points)
    # # collapse_scales_z=cosm_func.z_a(collapse_scales_a)
    
    """Compute the delta collapse star, that are the initial conditions 
    to obtain the collapse at the collapse scales. Find_delta_collapse_star also returns 
    the nonlinear matter density contrast."""
    delta_c_stars,nonlinear_density_contrasts=find_delta_collapse_star(collapse_scales_a)
    
    
    """Compute the linear density contrast at collapse and the growth factor for each collapse time"""
    linear_density_contrast_at_collapse=[[],[]]
    growth_factors=[]
    for i in range(number_of_points):
        """Compute the grow factor,i.e., the linear density contrast without decaying mode."""
        growth_factor=solve_growth_factor(delta_c_stars[i],collapse_scales_a[i])
        
        """Fill the list that stores all the grow factors"""
        growth_factors.append(growth_factor)
        
        """Fill the list that stores the linearized matter density contrast at collapse"""
        linear_density_contrast_at_collapse[0].append(collapse_scales_z[i])
        linear_density_contrast_at_collapse[1].append(growth_factor[1][-1])
    
    
    """Compute the overdensity at virialization defined in the theis as zeta_m^*"""
    virialization_overdensities_stars=[[],[]]
    for i in range(number_of_points):
        """Interpolate the i-th nonlinear density contrast that correspond to the i-th collapse time"""
        nonlinear_density_contrast_a=sp.interpolate.interp1d(nonlinear_density_contrasts[i][0], nonlinear_density_contrasts[i][1],
                                            fill_value="extrapolate", assume_sorted=False)
        
        """Find the turn around scale parameter and compute the turn around matter overdensity"""
        turn_around_scale=find_turn_around_scale(nonlinear_density_contrast_a,collapse_scales_a[i])
        turn_around_overdensity=np.round(nonlinear_density_contrast_a(turn_around_scale),4)+1
        
        """Find the virialization scale parameter and the radius at virialization x_vir to compute zeta_m^*"""
        virialization_scale,x_vir=find_virialization_scale(nonlinear_density_contrast_a,turn_around_overdensity,
                                                           turn_around_scale, collapse_scales_a[i])
        virialization_overdensities_stars[0].append(collapse_scales_z[i])
        virialization_overdensities_stars[1].append(turn_around_overdensity*(collapse_scales_a[i]/(turn_around_scale*x_vir))**3)
    
        """Debug prints"""
        # print("32=",round((collapse_scales_a[i]/(turn_around_scale*x_vir))**3,3))
        # print("a_c=",round(collapse_scales_a[i],3),"    a_ta=",round(turn_around_scale/collapse_scales_a[i],5),"    x_vir=",round(x_vir,3))
        # print((collapse_scales_a[i]/(turn_around_scale*x_vir))**3,"*",\
        #       turn_around_overdensity,"=",turn_around_overdensity*(collapse_scales_a[i]/(turn_around_scale*x_vir))**3)
    return [linear_density_contrast_at_collapse,virialization_overdensities_stars,growth_factors]

# friedmann_sol=fr_sol.solve()
# res=solve(friedmann_sol,5)
# linear_density_contrast_at_collapse=res[0]
# virialization_overdensities_stars=res[1]
# growth_factors=res[2]

"""Quick check to not mess it up."""
if save_to_txt and import_from_txt:
    raise Exception("Why you want to both save and import at the same time?")
if save_to_txt:    
    myie.save_to_txt_twocolumns(linear_density_contrast_at_collapse, "delta_c_backgr_de")
    myie.save_to_txt_twocolumns(virialization_overdensities_stars, "zeta_vir_star_backgr_de")
    myie.save_to_txt_multicolumn(growth_factors,"growth_factors_backgr_de")
if import_from_txt:
    delta_c_import=myie.import_from_txt_twocolumns("delta_c_backgr_de")
    zeta_vir_import=myie.import_from_txt_twocolumns("zeta_vir_star_backgr_de")
    growth_factors_import=myie.import_from_txt_multicolumn("growth_factors_backgr_de")

#%%
"""PLOTTING"""

save=False

#%%
"""Some constant functions used for the plots"""

def linear_contrast(a):
    return 1.68647019984
def virial_overdensity(a):
    return 146.841245384
def virial_overdensity_star(a):
    return 177.65287922
def turn_aroun_overdensity(a):
    return 5.551652

#%%



# """Linearly extrapoleted matter overdensity at collapse"""
# mypl.m_plot(linear_density_contrast_at_collapse,
#             title="$\Lambda CDM$ linear contrast at collapse",
#             # title="EdS linear contrast at collapse",
            
#             xlabel="$z_c$", ylabel="$\delta_c$",
#             # xscale="log",
#             func_to_compare=linear_contrast,
#             legend=("$\Lambda CDM$","1.6864"),
#             # legend=("EdS","1.6864"),
            
#             # ylim=(1.686,1.687),
#             ylim=(1.67,1.689),
            
#             save=save,
#             dotted=True,
#             name="linear_contrast_at_collapse_LCDM")
# pl.show()


# """Matter overdensity at virialization star"""
# mypl.m_plot(virialization_overdensities_stars,
#             title="$\Lambda CDM$ virialization overdensity*",
#             # title="EdS virialization overdenisty*",
            
#             xlabel="$z_c$", ylabel="$\zeta_{vir}^*$",
#             # xscale="log",
#             func_to_compare=virial_overdensity_star,
#             legend=("$\Lambda CDM$","177.653"),
#             # legend=("$EdS$","177.653"),
#             # ylim=(150,345),
#             # ylim=(177.55,177.7),
#             ylim=(100.55,350),
#             save=save,
#             dotted=True,
#             name="matter_overdensity_at_virialization_star_LCDM")
# pl.show()

# # mypl.save_run_parameters()

# # """Sounds to notify that the computation has completed."""
# # winsound.Beep(440, 1000)
# # winsound.Beep(880,1000)
# # winsound.Beep(1760,1000)


