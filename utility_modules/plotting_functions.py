# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:15:11 2022

@author: sebas
"""


import matplotlib.pyplot as pl
from datetime import datetime
import numpy as np

import sys
sys.path.append("../data_modules")
import simulation_parameters as params


defoult_save_plot_dir="C:/Users/sebas/OneDrive/Desktop/Thesis/Programs/Plots"

    
def plot(f,xlabel="$x$",ylabel="$y$",title="Function plot",
         legend=("First","Second"),func_to_compare=None,
         save=False,name=None,xscale="linear",yscale="linear",
         xlim=None,ylim=None, dotted=False):
    x=f[0]
    y=f[1]
    fig1 = pl.figure(1,figsize=(10,6))
    pl.plot(x,y,"b")
    if xlim!=None:
        pl.xlim(xlim)
    if ylim!=None:
        pl.ylim(ylim)
    if func_to_compare!=None:
        pl.plot(x,func_to_compare(x),"r")
        
    pl.title(title,fontsize=16)
    pl.xlabel(xlabel,fontsize=12)
    pl.ylabel(ylabel,fontsize=12)
    pl.legend(legend,
          shadow=True, loc="best", handlelength=1.5, fontsize=12)
    if save==True:
        now = datetime.now()
        if name!=None:
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            pl.savefig(defoult_save_plot_dir+name+current_time,dpi=512)
        else:
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            pl.savefig(defoult_save_plot_dir+current_time,dpi=512)
            
    pl.draw() 
    return None

def m_plot(f,
         xlabel="$x$",ylabel="$y$",
         title="Function plot",
         legend=("First","Second"),
         func_to_compare=None,
         save=False,save_plot_dir=defoult_save_plot_dir,name=None,
         xscale="linear",yscale="linear",
         xlim=None,ylim=None, 
         dotted=False,
         x_ticks=[],x_ticklables=[],y_ticks=[],y_ticklables=[]):
    
    colors="bgcmykw"
    linestyle_str =[ 'dashed','dotted','-', '--', '-.', ':', ' ', '', 'solid', 'dashdot'] 
    
    is_singleplot=False
    iterations=len(f)
    try:
      f[0][0][0]
    except:
      is_singleplot=True
      iterations=1
        
    for i in range(iterations):
        if is_singleplot:
            x=f[0]
            y=f[1]
        else:
            x=f[i][0]
            y=f[i][1]
        fig1 = pl.figure(1,figsize=(10,6))
        pl.xscale(xscale)
        pl.yscale(yscale)
        def dots(dotted):
            if dotted:
                return "o"
            else:
                return ""
        pl.plot(x,y,colors[i%7]+dots(dotted),linestyle=linestyle_str[i%10])
    if xlim!=None:
        pl.xlim(xlim)
    if ylim!=None:
        pl.ylim(ylim)
    if func_to_compare!=None:
        function=np.vectorize(func_to_compare)
        pl.plot(x,function(x),"r")
    if len(x_ticks)!=0:
        x_ticks=list(x_ticks)
        pl.xticks(x_ticks,x_ticklables)
    if len(y_ticks)!=0:
        y_ticks=list(y_ticks)
        pl.yticks(y_ticks,y_ticklables)
        
    pl.title(title,fontsize=16)
    pl.xlabel(xlabel,fontsize=12)
    pl.ylabel(ylabel,fontsize=12)
    if legend!=None:
        pl.legend(legend,shadow=True, loc="best", handlelength=1.5, fontsize=12)
    if save==True:
        now = datetime.now()
        if name!=None:
            current_time = now.strftime("%Y_%m_%d")
            pl.savefig(save_plot_dir+"/"+name+"__"+current_time,dpi=200)
        else:
            current_time = now.strftime("%Y_%m_%d")
            pl.savefig(save_plot_dir+"/"+"__"+current_time,dpi=200)
            
    pl.draw() 
    pl.show()
    return None

"""Function that the run parameters in a txt file"""
def save_run_parameters(path):
    with open(path+'/simulation_parameters.txt', 'w') as outfile:
        outfile.write("cosmological_parameters:\n")
        for key, value in params.cosmological_parameters.items():
            outfile.write(f"{key}={value}\n")
        outfile.write("\n")
        
        outfile.write("dark_energy_parameters:\n")
        for key, value in params.dark_energy_parameters.items():
            outfile.write(f"{key}={value}\n")
        outfile.write("\n")
        
        outfile.write("integration_parameters:\n")
        for key, value in params.integration_parameters.items():
            outfile.write(f"{key}={value:.1e}\n")
        outfile.write("\n")
    return None

# def g(x):
#     return x**2
# def h(x):
#     return x
# x=np.linspace(0,5,10)
# m_plot([x,g(x)],dotted=True)
# pl.show()
# m_plot([[x,g(x)],[x,h(x)]],dotted=True)
# pl.show()
