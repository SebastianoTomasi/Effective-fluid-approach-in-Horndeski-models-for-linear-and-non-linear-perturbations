# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:48 2023

@author: sebas
"""
import re
import sys
import os
sys.path.append("../data_modules")
import simulation_parameters as params

#%%


format_string='{:.9e}'
header_delimiter="*********************************\n"
data_delimiter=  "#################################\n"




def insert_header(file,var_par=None,var_par_values=None):
    file.write("cosmological_parameters:\n")
    for key, value in params.cosmological_parameters.items():
        file.write(f"{key}={value}\n")
    file.write("\n")
    
    file.write("dark_energy_parameters:\n")
    
    var_par_str = ','.join(map(str, var_par_values))
    for key, value in params.dark_energy_parameters.items():
        if key==var_par:
            file.write(f"{key}={var_par_str}\n")
        else:
            file.write(f"{key}={value}\n")
    file.write("\n")
    
    file.write("integration_parameters:\n")
    for key, value in params.integration_parameters.items():
        file.write(f"{key}={value:.1e}\n")
    file.write("\n")
    file.write(header_delimiter)

def save_to_txt_twocolumns(data,path):
    if len(data)!=2:
        raise Exception("Data must has two columns: data=[[list_1],[list_2]]")
    
    lenght=len(data[0])
    if lenght!=len(data[1]):
        raise Exception("The dimension of the two columns must be equal!")
        
    with open(path+".txt","w") as file:
        insert_header(file)
        for i in range(lenght):
            x=format_string.format(data[0][i])
            y=format_string.format(data[1][i])
            file.write(x+"\t"+y+"\n")



def import_from_txt_twocolumns(path):
    line="line"
    start_reading=False
    data=[[],[]]
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            if line==header_delimiter:
                start_reading=True
            elif line=="":
                return data
            elif start_reading:
                line=line.split()
                data[0].append(float(line[0]))
                data[1].append(float(line[1]))



def save_to_txt_multicolumn(data,path,var_par=None,var_par_values=None):
    with open(path+".txt","w") as file:
        insert_header(file,var_par,var_par_values)
        for i in range(len(data)):
            length=len(data[i][0])
            for j in range(length+1):
                if j==length:
                    file.write(data_delimiter)
                else:
                    x=format_string.format(data[i][0][j])
                    y=format_string.format(data[i][1][j])
                    file.write(x+"\t"+y+"\n")

def import_from_txt_multicolumn(path):
    start_reading=False          
    line="line"
    data=[]
    sub_data=[[],[]]
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            if start_reading:
                if line=="":
                    return data
                elif line==data_delimiter:
                    data.append(sub_data)
                    sub_data=[[],[]]
                else:
                    line=line.split()
                    sub_data[0].append(float(line[0]))
                    sub_data[1].append(float(line[1]))
            if line==header_delimiter:
                start_reading=True

def import_varied_parameter_values(path,var_par):
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            if var_par in line:
                return eval(line[len(var_par)+1:-1])
            
            
            
"""Import the header to save it in a separated txt in the plots folder."""
def import_header(path):
    header=[]
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            header.append(line)
            if line==header_delimiter:
                return header
            
def save_header(header,path):
    with open(path+".txt","w") as file:
        for line in header:
            file.write(line)


"""Create folder"""
def create_dir(path,name):
    paths = os.path.join(path, name) 
    print(paths)
    try:
        os.mkdir(paths)
        print(f"Directory '{name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{name}' already exists.")
    except Exception as e:
        print(f"Error creating directory: {e}")
        
"""Generates a structered tree of folders to store the data and the plots."""
def generate_data_folders():
    name_specifiers_0=["nonlinear_perturbations","linear_perturbations"]
    name_specifiers_1=["plots","unperturbed_de","perturbed_de","LCDM","EDS"]
    name_specifiers_2=["de_eos_"+str(i) for i in range(1,7)]
    name_specifiers_3=['w_i','w_f','trans_steepness','trans_z']
    
    for spec_0 in name_specifiers_0:
        myie.create_dir("../data/", spec_0)
        for spec_1 in name_specifiers_1:
            if spec_1=="LCDM":
                myie.create_dir("../data/"+spec_0, "LCDM")
            elif spec_1=="EDS":
                myie.create_dir("../data/"+spec_0, "EDS")
            elif spec_1=="unperturbed_de" or spec_1=="perturbed_de":
                myie.create_dir("../data/"+spec_0, spec_1)
                for spec_2 in name_specifiers_2:
                    myie.create_dir("../data/"+spec_0+"/"+spec_1, spec_2)
                    myie.create_dir("../data/"+spec_0+"/"+spec_1+"/"+spec_2, "plots")
                    for spec_3 in name_specifiers_3:
                        myie.create_dir("../data/"+spec_0+"/"+spec_1+"/"+spec_2, spec_3)
                        myie.create_dir("../data/"+spec_0+"/"+spec_1+"/"+spec_2+"/"+"plots", spec_3)
    
# file=open("test.txt","w")
# insert_header(file,"w_i",[1,2,3])
# file.close()

# test_header=import_header("test")
# save_header(test_header,"saved_header")

# test_data=[[1,2,3,4,5,6,7,8,9,10],[1,4,9,16,25,36,49,64,81,100]]
# test_name="test"
# save_to_txt_twocolumns(test_data,test_name)
# import_twocolumn_test=import_from_txt_twocolumns(test_name)

# multicol_test_data=[test_data,test_data,test_data,test_data]
# save_to_txt_multicolumn(multicol_test_data,test_name)

# import_multicolumn_test=import_from_txt_multicolumn(test_name)
