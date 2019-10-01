# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:34:35 2019

@author: vwa13369
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy as sp
from scipy import constants
from scipy.signal import find_peaks
import math
from tkinter import filedialog
from tkinter import *
import time
from numba import jit

#Define global variables
#tracking region limits (from CST model, ymax is the detector co-ordinates, the z and x limits are the inside faces of the monitor body)
detector_xmax=120
detector_xmin=-120
detector_y=116
detector_zmax=374
detector_zmin=354


##DOUBLE CHECK THESE SETTINGS WHEN DIMENSIONS ARE CHANGED, THEY DO NOT UPDATE IN THE CHECK_BOUNDARIES FUNCTION EVEN IF THEY ARE CHANGED IN THE MAIN PROGRAM LOOP
tracking_xmax=263
tracking_xmin=-263
tracking_ymax=116
tracking_ymin=-116
tracking_zmax=462.5
tracking_zmin=66.5

#assign physical constants to variable names
c = sp.constants.c

#generate lists arrays to hold the particle data. These are converted to numpy arrays after they have been populated, because it is quicker to append items to lists than nnumpy arrays
particles=[] #an array to store all the particle objects in during tracking
destroyed_particles=[] #an array to store particle objects that are removed from the simulation, usually because they have moved outside of the simulation region
final_timesteps=[] #an array to view all the final timesteps calculated for particles - for use in debugging

def get_fast_amplifier_profile(filepath=None, sample_rate=(1/60e6)):
     '''
     Reads MCPM and SCPM Data taken using Fast Amplifiers in the EPB1 Ionisation profile monitor, and returns three items:
          1) Raw data from each channeltron
          2) Integrated signal from each channeltron
          3) The specified filepath of the channeltron data
     '''
     #MAIN PROGRAM
     #Import and Analyse the Electric Field From CST
     if filepath == None:
          print("Please choose a fast amplifier data file from the IPM.\nThe data should be stored in a .csv file.\n")
          filepath=open_file_dialogue(message="Select Fast Amplifier .csv file") #let the user choose a filepath graphically
     else: filepath=filepath
     rawdata=pd.read_csv(filepath,skiprows=[0,1])
     print(rawdata.head())
     ctron_array=rawdata.to_numpy()
     no_samples=ctron_array.shape[0]
     no_ctrons=ctron_array.shape[1]
     total_time=no_samples*sample_rate
     total_time_us=total_time*1e6 #convert to microseconds to make it more readable
     print("The time interval contained in the fast amplifier data is: "+str(total_time_us)+"us.")
     integrated_data=[]
     for i in range(1,no_ctrons):
          #go through each channeltron and integrate the profile signal
          tempdata=ctron_array[:,i] #on each iteration of the loop, this contains the signals measured ver time by a single channeltron
          integrated_data.append(np.sum(tempdata))
     #temporary finish
     data=ctron_array
     print(integrated_data)
     return(data,integrated_data,filepath)
     
     
def read_harp_data(filepath,no_wires=24, wire_spacing_mm=6):
     filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\Harp Profiles 21July2019 Shift\\EPM26A2.dat'
     rawdata=pd.read_csv(filepath,header=None,skiprows=[0,9,10,11],delim_whitespace=True)
     hor_harp=rawdata[0:4]
     ver_harp=rawdata[4:9]
     hor_profile=hor_harp.values.flatten()
     ver_profile=ver_harp.values.flatten()
     harp_limit=(((no_wires*wire_spacing_mm)/2)-(wire_spacing_mm/2))
     harp_spacing=np.linspace(-harp_limit,harp_limit,num=no_wires)
     return(hor_profile,ver_profile,harp_spacing)

def open_file_dialogue(message='Select CST EField File'):
     root=Tk()
     root.filename=filedialog.askopenfilename(title=message)
     filepath=root.filename
     root.withdraw()
     return(filepath)
     
def check_axis_format(axis_name):
     if axis_name !='x' and axis_name !='y' and axis_name !='z':          
          print('\n*********************ERROR*******************************************************')
          print('*CST model axes specified incorrectly, please use only "x", "y" or "z" as inputs*')
          print('*********************ERROR*******************************************************\n')
          sys.exit()
     return()
     
def reformat_axes(field_data,model_axes):
     print('Reformatting field data...')
     if model_axes==['x','y','z']: print('field reformatting not required')
     elif model_axes==['x','z','y']:
          field_data=field_data[['x','z','y','Ex','Ez','Ey']]
          print('Input data axes reformatted: x -> x, y -> z, z -> y')
     elif model_axes==['y','x','z']:
          field_data=field_data[['y','x','z','Ey','Ex','Ez']]
          print('Input data axes reformatted: x -> y, y -> x, z -> z')
     elif model_axes==['y','z','x']:
          field_data=field_data[['y','z','x','Ey','Ez','Ex']]
          print('Input data axes reformatted: x -> z, y -> x, z -> y')
     elif model_axes==['z','x','y']:
          field_data=field_data[['z','x','y','Ez','Ex','Ey']]
          print('Input data axes reformatted: x -> y, y -> z, z -> x')
     elif model_axes==['z','y','x']:
          field_data=field_data[['z','y','x','Ez','Ey','Ex']]
          print('Input data axes reformatted: x -> z, y -> y, z -> x')
     print('-----------------------------------------------------------------')
     return(field_data)
     
def import_CST_EField(filepath=None,nrows=None,model_horizontal_axis='x',model_vertical_axis='y',model_longitudinal_axis='z', save_memory='True'):
     
     if filepath==None: filepath=open_file_dialogue() #if a filepath has not been specified, let the user choose a filepath graphically
     #Check format of input, and quit if format is incorrect
     check_axis_format(model_horizontal_axis)
     check_axis_format(model_vertical_axis)
     check_axis_format(model_longitudinal_axis)
     
     #Assume the CST 2018 EField Export Format - 6 columns of data: X(mm), Y(mm), Z(mm),Ex(V/mm),Ey(V/mm),Ez(V/mm)
     print('Electric field data will be imported from: '+filepath)
     print('\nThe column format output by CST is assumed to be: |x(mm)|y(mm)|z(mm)|Ex(V/m)|Ey(V/m)|Ez(V/m)|')
     print('\nLoading field data...') #this can take quite a few minutes
     try:
          field_data=pd.read_csv(filepath,header=None,nrows=nrows,skiprows=[0,1], delim_whitespace=True)
     except FileNotFoundError:
          print('\n*********************ERROR****************************')
          print("*The specified electric field file could not be found*")
          print('*********************ERROR****************************\n')
          sys.exit()
     
     #assign headers to the dataframe
     field_data.columns=['x','y','z','Ex','Ey','Ez']
     
     #multiply all the EField values by 1000 to convert them from V/m to V/mm
     convert_EField_units(field_data)
     print('Import of electric field data is complete!') 
     
     #reformat the field data if necessary
     model_axes=[model_horizontal_axis,model_vertical_axis,model_longitudinal_axis]
     field_data=reformat_axes(field_data,model_axes)
     print("Please note that for tracking to work accurately, the detectors should normal to the y axis of the CST model")
     field_data.columns=['x','y','z','Ex','Ey','Ez']
     
     #reduce the memory size of the stored file by downcasting all data to floats with fewer bytes where possible
     if save_memory: field_data=field_data.apply(pd.to_numeric,downcast='float')
     return(field_data)
 
def convert_EField_units(field_data):
     #converts efield data output in V/m from CST into V/mm to match the position data
     field_data['Ex']=field_data['Ex']/1000
     field_data['Ey']=field_data['Ey']/1000
     field_data['Ez']=field_data['Ez']/1000     
     print("EField data converted from V/m to V/mm")
     return()
def analyse_field_map(field_data,printout=False):
     print('Analysing imported field data...')
     nrows=field_data.shape[0]+1 #counts the number of rows of data. Have to add 1 because the 1st row counts as 0 in the ".shape[0]" command (e.g. a data frame with 4 rows of data would output a shape[0] of 3)
     step_size=field_data['z'][1]-field_data['z'][0]
     min_x=field_data['x'].min()
     max_x=field_data['x'].max()
     min_y=field_data['y'].min()
     max_y=field_data['y'].max()
     min_z=field_data['z'].min()
     max_z=field_data['z'].max()
     x_size=max_x-min_x
     y_size=max_y-min_y
     z_size=max_z-min_z
     if printout==True: print('The imported field map has the following properties:\n - No. data rows = '+str(nrows)+'\n - Step Size = '+str(step_size)+' mm\n - X Size = '+str(x_size)+' mm\n - Y Size = '+str(y_size)+' mm\n - Z Size = '+str(z_size)+' mm')
     print('-----------------------------------------------------------------')
     return(nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z)

def DrawMonitor(plotname):
        ctron_height=8
        plotname.plot([detzmin,detzmin,detzmax,detzmax,detzmin],[detxmin,detxmax,detxmax,detxmin,detxmin],[dety,dety,dety,dety,dety],c='black')
        plotname.plot([detzmin,detzmin+8,detzmin+8,detzmin+8,detzmin+8,detzmin],[detxmin,detxmin,detxmin,detxmax,detxmax,detxmax],[dety,dety+ctron_height+6,dety+ctron_height*8,dety+ctron_height*8,dety+ctron_height,dety],c='black')
        plotname.plot([detzmax,detzmax-8,detzmax-8,detzmax-8,detzmax-8,detzmax],[detxmin,detxmin,detxmin,detxmax,detxmax,detxmax],[dety,dety+ctron_height+6,dety+ctron_height*8,dety+ctron_height*8,dety+ctron_height,dety],c='black')
        plotname.plot([detzmax-8,detzmin+8],[detxmin,detxmin],[dety+ctron_height*8,dety+ctron_height*8],c='black')
        plotname.plot([detzmax-8,detzmin+8],[detxmax,detxmax],[dety+ctron_height*8,dety+ctron_height*8],c='black')
        
        #Draw the Earth Plate with its slot
        plotname.plot([309.5,418.5],[212.5,212.5],[dety-6,dety-6],c='black')
        plotname.plot([309.5,418.5],[-212.5,-212.5],[dety-6,dety-6],c='black')
        
        plotname.plot([detzmin,detzmin,detzmax,detzmax,detzmin],[detxmin,detxmax,detxmax,detxmin,detxmin],[dety-6,dety-6,dety-6,dety-6,dety-6],c='black')
        plotname.plot([309.5,309.5],[-212.5,212.5],[dety-6,dety-6],c='black')
        plotname.plot([418.5,418.5],[-212.5,212.5],[dety-6,dety-6],c='black')
        
        plotname.plot([309.5,309.5],[-212.5,-212.5],[dety-6,dety+ctron_height*8],c='black')
        plotname.plot([418.5,418.5],[-212.5,-212.5],[dety-6,dety+ctron_height*8],c='black')
        plotname.plot([309.5,309.5],[212.5,212.5],[dety-6,dety+ctron_height*8],c='black')
        plotname.plot([418.5,418.5],[212.5,212.5],[dety-6,dety+ctron_height*8],c='black')

class Particle:
     def __init__(self,x,y,z,species,ID,vx=0,vy=0,vz=0, lifetime=0, creation_time=0):
          global particles

          #attach values to the particle's properties
          self.x=x
          self.y=y
          self.z=z
          if lifetime==0:
               self.initial_x=x
               self.initial_y=y
               self.initial_z=z
          self.previous_x=x
          self.previous_y=y
          self.previous_z=z
          self.vx=vx
          self.vy=vy
          self.vz=vz
          self.creation_time=creation_time
          self.ID=ID
          self.species=species
          if species=='proton':
               self.mass=1.6726219e-27
               self.charge=1.60217662e-19
          elif species=='electron':
               self.mass=9.10928e-31
               self.charge=-1.60217662e-19
          elif species=='antiproton':
               self.mass=1.6726219e-27
               self.charge=-1.60217662e-19
          elif species=='positron':
               self.mass=9.10928e-31
               self.charge=1.60217662e-19
          elif species=='oxygen':
               self.mass=1.6726219e-27*16
               self.charge=1.60217662e-19
          elif species=='N2':
               self.mass=1.6726219e-27*28
               self.charge=1.60217662e-19
          elif species=='N+':
               self.mass=1.6726219e-27*14
               self.charge=1.60217662e-19
          elif species=='H20+':
               self.mass=1.6726219e-27*18
               self.charge=1.60217662e-19
          elif species=='OH+':
               self.mass=1.6726219e-27*17
               self.charge=1.60217662e-19
          elif species=='H2':
               self.mass=1.6726219e-27*2
               self.charge=1.60217662e-19
          elif species=='CO':
               self.mass=1.6726219e-27*32
               self.charge=1.60217662e-19
          elif species=='test': #create an ultra-light test particle to test tracking relativity calculations
               self.mass=9e-32
               self.charge=-1.60217662e-19
          else:
               print('\n*********************ERROR****************************')
               print("* Particle species incorrectly defined during creation *")
               print('*********************ERROR****************************\n')
               sys.exit()
          self.lifetime=0
          self.plot_colour=np.random.choice(range(256),size=3)/256 #generate a colour to plot the particle with on any trajectory graphs. This keeps the colour consistent even if each timestep of movement is plotted with different plot commands
          #check if the particle has been created inside the CST model or not
          if check_tracking_boundaries(x=self.x,y=self.y,z=self.z): 
               particles.append(self) #only create the particle if it's co-ordinates are within the simulation region
          else: 
               print("Particle could not be generated - requested location ("+str(self.x)+","+str(self.y)+","+str(self.z)+") falls outside simulation region")
     
     def move(self,timestep=1e-9,plot=False,final_track=False):
          running_times['particle move start']=time.time()
          if final_track==False: #if the particle is doing its final timestep, then these values have already been assigned and do not need to be assigned again
               #update the previous position values to the current position values
               self.previous_x=self.x
               self.previous_y=self.y
               self.previous_z=self.z
               self.previous_vx=self.vx
               self.previous_vy=self.vy
               self.previous_vz=self.vz
          #running_times['field boundary check start']=time.time()
          field_row_number=lookup_field_value(self.previous_x,self.previous_y,self.previous_z,efield)
          #running_times['field boundary check finish']+=(time.time()-running_times['field boundary check start'])
          #calculate relativistic gamma and use this to increase the particle mass to account for relativistic effects
          relativistic_mass=self.mass*(calculate_gamma(self))
          
          #look up field values one each - this operation is time consuming and therefore its best not to look each value up twice
          #Ex=efield['Ex'].values[int(field_row_number)]
          #Ey=efield['Ey'].values[int(field_row_number)]
          #Ez=efield['Ez'].values[int(field_row_number)]
          
          Ex=numpy_efield[int(field_row_number)][0]
          Ey=numpy_efield[int(field_row_number)][1]
          Ez=numpy_efield[int(field_row_number)][2]
          
          #Move the particle through one timestep calculating new positions and velocities using the Lorentz force applied by the EField at the particle's location
          #calculate new positions and new velocities after timestep


          #calculation called in a seperate, purely mathmatical function, optimised with the jit compiler to improve code speed.
          self.x,self.vx=calculate_new_position_and_velocity(self.previous_x,self.previous_vx,timestep,self.charge,relativistic_mass,Ex)
          self.y,self.vy=calculate_new_position_and_velocity(self.previous_y,self.previous_vy,timestep,self.charge,relativistic_mass,Ey)
          self.z,self.vz=calculate_new_position_and_velocity(self.previous_z,self.previous_vz,timestep,self.charge,relativistic_mass,Ez)
          
          #Print position information for debugging
          #print("Ey = "+str(efield['Ey'][field_row_number])+", Previous Y = "+str(self.previous_y)+", Previous Vy = "+str(self.previous_vy)+", New Y_Position = "+str(self.y)+", New Y Velocity = "+str(self.vy)+", Timstep used = "+str(timestep))
          
          if final_track==False: #if final_track is true, then the particle is doing it's final movement over a precise timestep, so this check is not needed.
               #if the particle moves outside of the simulation region then remove it from the particles object list it
               if check_tracking_boundaries(x=self.x,y=self.y,z=self.z)==False:
                    self.perform_final_movement(field_row_number,plot=plot)
               #otherwise, increase the particle lifetime by 1 timestep
               else:
                    self.lifetime=self.lifetime+timestep             
                    #plot the particle movement to the existing plot window if requested by the user
                    if plot==True:
#                         plt.plot([self.previous_z,self.z],[self.previous_y,self.y], color=self.plot_colour, linewidth=0.5)
                         #print("\n\nID="+str(self.ID)+", timestep_counter (i)="+str(i)+", Positions = ("+str(self.x)+","+str(self.y)+","+str(self.z)+").")
                         #print(trajectories[self.ID,i,:])
                         trajectories[self.ID,int(simulation_time/input_timestep),:]=[self.x,self.y,self.z] #i is the timestep number that the simulation is at, present in the loop in main that calls the main particle function
                         #print(trajectories[self.ID,i,:])
                    running_times['particle move finished']+=(time.time()-running_times['particle move start'])
          #If this is the final timestep, update particle lifetime, plot movement if requested, then destroy the particle         
          if final_track==True:
               self.lifetime=self.lifetime+timestep
               if plot==True: trajectories[self.ID,int(simulation_time/input_timestep),:]=[self.x,self.y,self.z] #i is the timestep number that the simulation is at, present in the loop in main that calls the main particle function
               #print("PARTICLE DESTROYED - Lifetime = "+str(self.lifetime)+" s")
               #destroy the particle
               running_times['particle move finished']+=(time.time()-running_times['particle move start'])
               self.destroy()
       
     def perform_final_movement(self,field_row_number,plot): #use the particle's velocity, and the previous and new positions to precisely calculate the time taken to reach the tracking region boundary.
          #print("FINAL MOVEMENT INITIALISED")
          #calculate relativistic gamma and use this to increase the particle mass to account for relativistic effects
          relativistic_mass=self.mass*(calculate_gamma(self))        
          
          #import pdb; pdb.set_trace()
          #identify which plane the particle went outside the tracking region on. Then calculate the time taken by solving the quadratic formula twice (once with a plus sign and once with a - sign). When solving the formula, replace the new position with the position of the tracking region boundary, so that the time taken to reach it is calculated accurately 
          if self.y > tracking_ymax:
               Ey=numpy_efield[int(field_row_number)][1]
               #calculate roots of the quadratic which calculates the final timestep (t_final = (-prev_vy +- sqrt(prev_vy**2-4(q*Ey/2*mass)(y-prev_y)))/(q*Ey/mass))
               a=(self.charge*Ey*1e6)/(2*relativistic_mass) #multiplied by 1e6 to give units of mm/s^2
               b=self.previous_vy
               c=self.previous_y-tracking_ymax
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c])) #np.roots returns the two roots of the quadratic equation which is being solved to calculate the timestep required to reach the simulation boundary
          elif self.y < tracking_ymin:
               Ey=numpy_efield[int(field_row_number)][1]
               a=(self.charge*Ey*1e6)/(2*relativistic_mass)
               b=self.previous_vy
               c=self.previous_y-tracking_ymin
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.x > tracking_xmax:
               Ex=numpy_efield[int(field_row_number)][0]
               a=(self.charge*Ex*1e6)/(2*relativistic_mass)
               b=self.previous_vx
               c=self.previous_x-tracking_xmax
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.x < tracking_xmin:
               Ex=numpy_efield[int(field_row_number)][0]
               a=(self.charge*Ex*1e6)/(2*relativistic_mass)
               b=self.previous_vx
               c=self.previous_x-tracking_xmin
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.z > tracking_zmax:
               Ez=numpy_efield[int(field_row_number)][2]
               a=(self.charge*Ez*1e6)/(2*relativistic_mass)
               b=self.previous_vz
               c=self.previous_z-tracking_zmax
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.z < tracking_zmin:
               Ez=numpy_efield[int(field_row_number)][2]
               a=(self.charge*Ez*1e6)/(2*relativistic_mass)
               b=self.previous_vz
               c=self.previous_z-tracking_zmin
               
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          else:
               print("\n******************ERROR*********************")
               print("* Final movement method called incorrectly *")
               print("*   (No boundaries exceeded by particle)   *")
               print("********************************************")
               sys.exit()
          
          #Select the correct solution from the quadratic: 
          #If both solutions are positive, use the smaller one (the other will be larger than a normal timestep would have been). Additionally, if the equation only had one root, this will be selected here as well
          if final_timestep1 >= 0 and final_timestep2 >= 0: final_timestep=np.min([final_timestep1,final_timestep2]) 
          #if one solution is negative, use the positive one (checked with an exclusive or condition below).
          elif (final_timestep1 < 0 and final_timestep2 >= 0) or (final_timestep1 >= 0 and final_timestep2 < 0): final_timestep=np.max([final_timestep1,final_timestep2])
          #if both solutions are negative, return an error
          elif final_timestep1 < 0 and final_timestep2 < 0:
               print("\n********************ERROR**********************")
               print("*  Final movement timestep calculation error  *")
               print("*     (Both roots return negative times)      *")
               print("***********************************************")
               sys.exit()
                   
          final_timesteps.append(final_timestep)
          #if final_timestep > 1e-9: print("Timestep too large...previous_y = "+str(self.previous_y)+", y = "+str(self.y)+", previous_vy = "+str(self.previous_vy)+", vy = "+str(self.y)+", Ey = "+str(Ey))
          self.move(timestep=final_timestep,final_track=True, plot=plot)

     def destroy(self):
          global particles
          #log the position as the particle's final position
          self.final_x=self.x
          self.final_y=self.y
          self.final_z=self.z
          #remove the particle from the particles array
          particles=np.delete(particles,np.argwhere(particles==self)) #np.argwhere returns the index number of the particles array that contains the current particle. np.delete then deletes the particle at the index specified. This whole line is equivalent to particles.remove(self) if a python list was used, but this would be slower than a numpy array
          #add the particles to a seperate list that stores the removed particles, in case extra analysis on these is required
          destroyed_particles.append(self)

@jit
def calculate_new_position_and_velocity(previous_x,previous_vx,timestep,charge,mass,efield_component):
     '''calculates the new position and velocity of a particle in one axis, as a result of the lorentx force applied on a charged particle by an electric field. \n\n The function requires inputs of the particles's: starting position, starting velocity, timestep size, charge, mass, and electric field present at the particle's location.'''
     x=previous_x+(previous_vx*timestep)+(((charge*timestep*timestep*efield_component)/(2*mass))*1e6)
     vx=previous_vx+(((charge*timestep*efield_component)/(mass))*1e6)
     return(x,vx)

@jit
def calculate_relativistic_beta(vx,vy,vz):
     '''takes a 3D velocity vector in mm/s, converts to m/s, calculates the magnitude of the velocity, then calculates relativistic beta from this.\n\n This function has been seperated and decorated with @jit to increase performance.'''
     #return(math.sqrt(math.pow((vx/1000),2)+math.pow((vy/1000),2)+math.pow((vz/1000),2))) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
     beta=(((((vx/1000)**2)+((vy/1000)**2)+((vz/1000)**2))**0.5)/c) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
     return(beta)
     
@jit
def calculate_gamma_from_beta(beta):  
     '''takes an input relativistic beta value and returns the equivalent relativistic gamma value.\n\n This function has been separated and decorated with @jit to increase performance.'''
     relativistic_gamma=1/(math.sqrt(1-math.pow(beta,2)))
     return(relativistic_gamma)

def calculate_gamma(self):
     #running_times['calculate gamma start']=time.time()
     '''Returns the relativistic gamma of a particle based on it's input velocities in each cartesian plane.\n\n vx,vy and vz should be specified in mm/s'''
     
     relativistic_beta=calculate_relativistic_beta(self.previous_vx,self.previous_vy,self.previous_vz)
     
     #velocity_magnitude=math.sqrt(math.pow((self.previous_vx/1000),2)+math.pow((self.previous_vy/1000),2)+math.pow((self.previous_vz/1000),2)) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
     #relativistic_beta=velocity_magnitude/c
     
     #print("\nCalculate_gamma has been called. Beta value = "+str(relativistic_beta)+"\n-->vy = "+str(self.vy)+", vz = "+str(self.vz)+", vx = "+str(self.vx)+"\n-->previous_vy = "+str(self.previous_vy)+" previous_vz = "+str(self.previous_vz)+"previous_vx = "+str(self.previous_vx)+"\n--> velocity magnitude = "+str(velocity_magnitude))
     if relativistic_beta >= 1:
          #scale the particle's velocities to set the overall magnitude equal to 99.9% the speed of light, to prevent beta >1 and gamma = infinity!
          self.previous_vx = self.previous_vx/(1.001*relativistic_beta)
          self.previous_vy = self.previous_vy/(1.001*relativistic_beta)
          self.previous_vz = self.previous_vz/(1.001*relativistic_beta)
          #recalculate particle velocity and beta as a check
          velocity_magnitude=math.sqrt(math.pow((self.previous_vx/1000),2)+math.pow((self.previous_vy/1000),2)+math.pow((self.previous_vz/1000),2))#calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
          relativistic_beta=velocity_magnitude/c 
          print("Recalculated Particle Beta = "+str(relativistic_beta))
          print("**WARNING*** A particle tried to exceed the speed of light. It's velocity has been reduced to 0.999c, to stop the universe from imploding! \n-->Consider using a smaller timestep value")
     
     relativistic_gamma=calculate_gamma_from_beta(relativistic_beta)
     #relativistic_gamma=1/(math.sqrt(1-math.pow(relativistic_beta,2)))
          
     #running_times['calculate gamma finished']+=(time.time()-running_times['calculate gamma start'])
     return(relativistic_gamma)

def normalise_list(data):
     '''Takes a python list as an input, converts it to a numpy array and scales it so that it's maximum value is 1'''
     data=np.asarray(data)
     data=data/data.max()
     return(data)
     
def normalise_profile_area(data):
     '''Takes a beam profile and normalises the area under the curve to 1 '''
     normalised_profile=data/np.sum(data)
     return(normalised_profile)

def check_field_boundaries(x,y,z,field_data):
     #REMOVED TO SPEED UP CODE - rely on values calculated when analyse field was last called in the main code -->
     #field_step_size=field_data['z'][1]-field_data['z'][0]
     #if x > (field_data['x'].max()+field_step_size/2) or y > (field_data['y'].max()+field_step_size/2) or z > (field_data['z'].max()+field_step_size/2) or x < (field_data['x'].min()-field_step_size/2) or y < (field_data['y'].min()-field_step_size/2) or z < (field_data['z'].min()-field_step_size/2):
     #if set_xmax==0
     if x > field_max_x or y > field_max_y or z > field_max_z or x < field_min_x or y < field_min_y or z < field_min_z:
     #     running_times['field boundary check finished']+=(time.time()-running_times['field boundary check start'])
          return(False) #return false if the particle is outside the simulation region
     else:
     #     running_times['field boundary check finished']+=(time.time()-running_times['field boundary check start'])
          return(True) #return true if the particle is still inside the simulation region

def check_tracking_boundaries(x,y,z):
     if x > tracking_xmax or y > tracking_ymax or z > tracking_zmax or x < tracking_xmin or y < tracking_ymin or z < tracking_zmin:
          return(False)
     else:
          return(True)

def lookup_field_value(x,y,z,field_data): #check whether a specified location falls inside the imported field region or not
     #calculate properties from the field map
     
     #IMPORTANT - calculating these properties every time the function is called slows the code down significantly, so rely on the values saved when the analyse field map function is called in the main code
     #step_size=field_data['z'][1]-field_data['z'][0]
     #x_size=field_data['x'].max()-field_data['x'].min()
     #y_size=field_data['y'].max()-field_data['y'].min()
     #z_size=field_data['z'].max()-field_data['z'].min()
     
     #check that the field value is inside the region covered by the field data
     if check_field_boundaries(x,y,z,field_data)==False: #if the requested location falls outside the range of the imported data, exit the program  
          print('\n*********************ERROR****************************')
          print("* EField location requested which is outside of field*\n")
          print("*  data range - check imported field dimensions and  *\n")
          print("*                 program settings                   *")
          print('*********************ERROR****************************\n')
          sys.exit()
     else:     
          #find nearest value to the requested position
          #call seperate function which has had @jit applied to speed it up loads
          element_num=calculate_row_number(x,y,z)
          #REMOVED TO SPEED UP CODE, rely on field properties calculated by analyse field function in main code
          # element_num=((np.round(((x-field_data['x'].min())/step_size)))*(1+(y_size/2))*(1+(z_size/2)))+((np.round(((y-field_data['y'].min())/step_size)))*(1+(z_size/2)))+(np.round(((z-field_data['z'].min())/step_size)))        
          return(element_num)

@jit
def calculate_row_number(x,y,z):
     '''Takes an input x,y,z co-ordinate and returns the row number in the electric field file which is closest to that location. \n\n This function has been separated and decorated with @jit to increase performance.'''
     #instead of rounding each number with np.round, np.int is used instead because it is much faster. This doesnt round, but instead just ignores everything after the decimal point. By adding 0.5 to each numebr before applying np.int, the result is the same as it would be if np.round had been used.
     element_num=((np.int(((x-min_x)/step_size)+0.5))*(1+(y_size/step_size))*(1+(z_size/step_size)))+((np.int(((y-min_y)/step_size)+0.5))*(1+(z_size/step_size)))+(np.int(((z-min_z)/step_size)+0.5))
     return(element_num)


def analyse_profile(profile_x,profile_y,min_peak_height=0):
     '''Returns the 95% width of the input profile and the location of the peak.
     
     If multiple peaks are detected in the profile (e.g. due to noise), increase the peak threshold value to remove these.
     
     The position and width values and dimensions are calculated from the input profile_x array (see example below)
     
     E.G. For a horizontal profile measurement with an ISIS IPM, profile_x should contain the positions of the centre of each channeltron, and 
     profile_y should contain the measured beam profile, i.e. the particles measured by each channeltron.'''
     spacing=profile_x[1]-profile_x[0]
     lower_boundary=np.sum(profile_y)*0.025
     upper_boundary=np.sum(profile_y)*0.975
     cumsum=0 #cumulative sum
     i=0
     for data in profile_y:
          cumsum=cumsum+data
          if cumsum >= lower_boundary: break
          i=i+1
     lower_location=i*spacing
     cumsum=np.sum(profile_y)
     i=np.size(profile_y)
     for data in reversed(profile_y):
          cumsum=cumsum-data
          if cumsum <= upper_boundary: break
          i=i-1
     upper_location=i*spacing
     width95=upper_location-lower_location
     
     #find profile centre
     midpoint=np.sum(profile_y)*0.5
     i=0
     cumsum=0
     for data in profile_y:
          cumsum=cumsum+data
          if cumsum >= midpoint: break
          i=i+1
     centre_position=np.min(profile_x)+(i*spacing) #returns the centre position of the profile peak
     return(width95,centre_position)
     
def calculate_percent_width(profile_x,profile_y,min_peak_height=0,percentage=95, hhw=False):
     '''Returns the specified% width of the input profile, the half height width (if requested) and the location of the peak.
     
     If multiple peaks are detected in the profile (e.g. due to noise), increase the peak threshold value to remove these.
     
     The position and width values and dimensions are calculated from the input profile_x array (see example below)
     
     E.G. For a horizontal profile measurement with an ISIS IPM, profile_x should contain the positions of the centre of each channeltron, and 
     profile_y should contain the measured beam profile, i.e. the particles measured by each channeltron.'''
     
     spacing=profile_x[1]-profile_x[0]
     lower_boundary=np.sum(profile_y)*((100-percentage)/200)
     upper_boundary=np.sum(profile_y)*(1-((100-percentage)/200))
     cumsum=0 #cumulative sum
     i=0
     for data in profile_y:
          cumsum=cumsum+data
          if cumsum >= lower_boundary: break
          i=i+1
     lower_location=i*spacing
     cumsum=np.sum(profile_y)
     i=np.size(profile_y)
     for data in reversed(profile_y):
          cumsum=cumsum-data
          if cumsum <= upper_boundary: break
          i=i-1
     upper_location=i*spacing
     width=upper_location-lower_location
     
     #find profile centre
     midpoint=np.sum(profile_y)*0.5
     i=0
     cumsum=0
     for data in profile_y:
          cumsum=cumsum+data
          if cumsum >= midpoint: break
          i=i+1
     centre_position=np.min(profile_x)+(i*spacing) #returns the centre position of the profile peak
     
     if hhw==True:
          i=0
          half_height=0.5*np.max(profile_y)
          for data in profile_y:
               if data >= half_height: break
               i=i+1
          lower_location=i*spacing
          i=np.size(profile_y)
          for data in reversed(profile_y):
               if data >= half_height: break
               i=i-1
          upper_location=i*spacing
          hhw=upper_location-lower_location
          return(width,centre_position,hhw)
     else:
          return(width,centre_position)
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
#######################################################################################################################################################################
#MAIN PROGRAM
print('**********************************************************************')
print("*               ISIS IPM PARTICLE TRACKING CODE             CCW v3.0 *")
print('**********************************************************************')

#STOP THE PROGRAM PLOTTING THE SET OF MANUALLY INPUT IBIC PLOTS AND HARP DATA AT AFTER TRACKING
ibicplots=False

#Import and Analyse the Electric Field From CST
plt.close('all') #close anyt plots still open from previous runs of the code
print("Please choose CST electric field files...\n The first file chosen should contain the IPM and particle beam, the second file should contain the IPM with no beam.\nIf only one file is needed, select the same file twice.\n**IMPORTANT - Both fields must be the same size, cover the same co-ordinates and have the same resolution.**\n-------------------------------------\n")
filepath_withbeam=open_file_dialogue(message="Select CST EField File 1 - With Beam Inside the IPM") #let the user choose a filepath graphically
filepath_nobeam=open_file_dialogue(message="Select CST EField File 2 - With No Beam In The Model") #ask
if not filepath_withbeam or not filepath_nobeam: #check that both filepaths were picked (an empty string will trigger this if statement)
     print("**************ERROR******************")
     print("*        Filepath(s) Missing        *")
     print("*Please specify 2 valid EField files*")
     print("*************************************")
     sys.exit()

select_harp=True
if select_harp:
     harp_filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\Harp Profiles 21July2019 Shift\\EPM26A2.dat'     
     if harp_filepath==None: 
          select_harp=False
          print('No harp monitor data loaded')
     else:
          hor_harp,ver_harp,harp_wire_positions=read_harp_data(harp_filepath)
          #interpolate the data to accurately calculate widths and centre positions
          interpolated_harp_wire_positions=np.linspace(-69,69,1000)
          interpolated_hor_harp_profile=np.interp(interpolated_harp_wire_positions,harp_wire_positions,hor_harp)
          interpolated_ver_harp_profile=np.interp(interpolated_harp_wire_positions,harp_wire_positions,ver_harp)
          #analyse the profiles for 95% widths and centre positions in each plane
          hor_harp_width,hor_harp_centre=analyse_profile(interpolated_harp_wire_positions,interpolated_hor_harp_profile)
          ver_harp_width,ver_harp_centre=analyse_profile(interpolated_harp_wire_positions,interpolated_ver_harp_profile)
          print('Harp profile loaded from: '+harp_filepath)
          print('Beam X Centre = '+str(hor_harp_centre)+'mm, Beam Y Centre = '+str(ver_harp_centre)+'mm, \nHorizontal 95% Width = '+str(hor_harp_width)+'mm, Vertical 95% Width = '+str(ver_harp_width)+'mm\n\n')
     
#filepath='C:\\Users\\vwa13369\\Desktop\\AccPhys 2016\\2019_Field_Maps\\-15kV_-1400VBias_2_27e13ppp_radx_54_9_rady_41_5_xoff_0_86_yoff_-2_9_CFOFF.txt'
running_times={'start time':time.time(),'field boundary check start':0,'field boundary check finish':0,'calculate gamma start':0,'calculate gamma finished':0,'particle move start':0,'particle move finished':0} #Log the time to calculate the execution time of the code


#IMPORT EFIELD, ANALYSE AND CONVERT TO NUMPY----------------------------------------------------------------------------------------------------------------------
efield=import_CST_EField(filepath_withbeam,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
#analyse the field map to obtain properties needed for data lookup in the field
nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z=analyse_field_map(efield,printout=True)
#calculate field boundaries - doing this here speeds up the runtime of functions that use these values in if statements
field_max_x=max_x+step_size/2
field_min_x=min_x-step_size/2
field_max_y=max_y+step_size/2
field_min_y=min_y-step_size/2
field_max_z=max_z+step_size/2
field_min_z=min_z-step_size/2

#check that requested trackingboundaries are not larger than the imported Efield size
if tracking_xmax > field_max_x: tracking_xmax=field_max_x
if tracking_xmin < field_min_x: tracking_xmin=field_min_x
if tracking_ymax > field_max_y: tracking_ymax=field_max_y
if tracking_ymin < field_min_y: tracking_ymin=field_min_y
if tracking_zmax > field_max_z: tracking_zmax=field_max_z
if tracking_zmin < field_min_z: tracking_zmin=field_min_z

print("Imported Data Sample:")
print(efield.head())

#Load the electric field data from the pandas dataframe into numpy arrays to increase performance
numpy_efield=efield[['Ex','Ey','Ez']].to_numpy()
print("Electric field values loaded into seperate numpy array for fast access.")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#generate an ion distribution
particle_num=20000 #the number of particles that should be generated inside the monitor
tracking_steps=4000
input_timestep=1e-9
bunch_length=100e-9
bunch_spacing=225e-9
number_of_bunches=2
time_resolution=8e-9 #time resolution in seconds
plot_detected_only=True

store_trajectories=True
if store_trajectories:
     trajectories=np.empty(((particle_num,tracking_steps,3)))
     trajectories[:]=np.nan
     print("Particle trajectory array created with dimensions (particle number,number of timesteps,position data) = "+str(np.shape(trajectories)))
     print("***WARNING - CALCULATING PARTICLE TRAJECTORIES CAN BE MEMORY INTENSIVE.***\n***CONSIDER USING FEWER PARTICLES OR TIMESTEPS IF MEMORY IS A CONCERN***")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#GENERATE PARTICLE DISTIBUTION
beam_xrad=25/0.95 #/0.95 to scale the radius up from a 95% width to a 100% width
beam_yrad=26/0.95
beam_xpos=1
beam_ypos=6
beam_length_mm=500

#if harp data has been chosen, match the beam properties to the harp data
if select_harp==True:
     beam_xrad=(hor_harp_width/2)/0.95
     beam_yrad=(ver_harp_width/2)/0.95
     beam_xpos=hor_harp_centre
     beam_ypos=ver_harp_centre

#if disctribution is not specified, a uniform distribution will be generated
#beam_distribution='gaussian' 
beam_distribution='beta'

detector_z_centre=detector_zmin+((detector_zmax-detector_zmin)/2)
beam_zmin=detector_z_centre-(beam_length_mm/2)
beam_zmax=detector_z_centre+(beam_length_mm/2)

#make sure that the veam will not be generated outside of the simulation region
if beam_zmax > tracking_zmax: beam_zmax=tracking_zmax
if beam_zmin < tracking_zmin: beam_zmin=tracking_zmin

print('-----------------------------------------------------------------')

#GENERATE PARTICLE DISTRIBUTION------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Generating beam between: z_min="+str(beam_zmin)+"mm, and beam z_max="+str(beam_zmax)+"mm.")
print("\Beam Properties:\n - No. Bunches = "+str(number_of_bunches)+"\n - Bunch length = "+str(bunch_length*1e9)+"ns\n - Inter-bunch spacing = "+str(bunch_spacing*1e9)+"ns \n Generating...\n")
running_times['start of beam generation']=time.time()
particle_counter=1
particlers=[]
particlethetas=[]
for j in range(0,number_of_bunches):
     # to calculate bunch start time, in seconds in the loop below, use: bunch_start_time=(j*(bunch_spacing+bunch_length))*1e9
     for i in range(0,int(np.round(particle_num/number_of_bunches))):
          #create a particle. --> Calculate the x and y positions as random numbers generated inside gaussian (normal) distributions for each axis. The distribution mean (loc) values are set to the beam offset positions, and the standard deviations (scale) are set to half of the beam width in CST (so that the 95% widths correspond to 2 signma, which is correct for a gaussian). Size is the number of points that the function generates
          particle_creation_time=((j*(bunch_spacing+bunch_length)*1e9)+np.random.randint(low=0,high=(bunch_length*1e9)+1))/1e9 #pick a random creation time in seconds (but in integer values of ns) for each particle that fits inside the bunch length
          if beam_distribution=='gaussian':
               particle_r=(abs((np.random.normal(loc=0,scale=0.5,size=None)))**0.5)*beam_xrad
               particle_theta=(np.random.rand()*360)
               particlers.append(particle_r)
               particle_x=(particle_r*math.cos(math.radians(particle_theta)))+beam_xpos
               particle_y=(particle_r*math.sin(math.radians(particle_theta))*beam_yrad/beam_xrad)+beam_ypos
#               particle_x=(np.random.normal(loc=beam_xpos,scale=0.5*beam_xrad,size=1)[0])
#               particle_y=(np.random.normal(loc=beam_ypos,scale=0.5*beam_yrad,size=1)[0])
          elif beam_distribution=='beta':
               alpha=1
               beta=1
               #generate random magnitudes and angles in polar co-ordinate values using the x radius, this will be converted into a circular beam with a beta distribution, then stretched along the y axis to complete the beam generation
               particle_r=((abs((np.random.beta(alpha,beta))-0.5)*2)**0.5)*beam_xrad #generate a random magnitude using the beta distribution. The random number needs to be square rooted to ensure that distribution is not weighted too much towards the inside of the beam when the polar co-ordinates are converted back into cartesian co-ordinates (for explanation see:http://www.anderswallin.net/2009/05/uniform-random-points-in-a-circle-using-polar-coordinates/)
               #WEDNESDAY 31st JULY
               #need to calculate the "inverse of the cumulative distribution for the beta function...for a uniform distribution it is just sqrt(r)
               particlers.append(particle_r)
               particle_theta=(np.random.rand()*360) #generate a random angle between 0-369
               #convert from polar to cartesian co-ordinates
               particle_x=particle_r*math.cos(math.radians(particle_theta))+beam_xpos #convert degrees value into radians, so that the math.cos function works
               particle_y=(particle_r*math.sin(math.radians(particle_theta))*beam_yrad/beam_xrad)+beam_ypos
               #the beam now has a beta distribution but is circular, with radius = x_rad
               #scale all the y-co-ordinates by y_rad/x_rad to generate the final beam shape
               particle_y=particle_y*beam_yrad/beam_xrad
          else: #generate a uniform beam by default (follows same procedure as above, but without the comments. Polar distribution generated then converted to cartesian, to prevent the distribution shape being a square)
               beam_distribution='uniform'
               particle_r=((np.random.rand()**0.5)*beam_xrad)
               particle_theta=(np.random.rand()*360)
               particlers.append(particle_r)
               particlethetas.append(particle_theta)
               particle_x=(particle_r*math.cos(math.radians(particle_theta))+beam_xpos)
               particle_y=((particle_r*math.sin(math.radians(particle_theta))*beam_yrad/beam_xrad)+beam_ypos)
          particle_z=np.random.uniform(low=beam_zmin,high=beam_zmax)
          #if particle_counter%3==0: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='oxygen',ID=particle_counter-1)
          #elif particle_counter%3==1: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='N2',ID=particle_counter-1)
          dice=(np.random.rand())*2
          if dice <= 0.62: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='H20+',ID=particle_counter-1)
          elif dice > 0.62 and dice <= 0.78: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='OH+',ID=particle_counter-1)
          elif dice > 0.78 and dice <= 0.98: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='proton',ID=particle_counter-1)
          elif dice > 0.98 and dice <= 1: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='oxygen',ID=particle_counter-1)
          elif dice > 1 and dice <= 1.2: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='H2',ID=particle_counter-1)
          elif dice > 1.2 and dice <= 1.6: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='N+',ID=particle_counter-1)
          else: Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='N2',ID=particle_counter-1)
          particle_counter=particle_counter+1
print("Particle beam generated, using a "+beam_distribution+" probability distribution, containing "+str(len(particles))+" particles.")
particles=np.array(particles)
print("Particle list converted to numpy array for faster tracking.")
#xpositions=[]
#ypositions=[]
#for particle in particles:
#    xpositions.append(particle.x)
#    ypositions.append(particle.y)
#plt.figure()
#plt.scatter(xpositions,ypositions)
#plt.figure()
#plt.hist(xpositions,bins=100)
#plt.show()

running_times['finished beam generation']=time.time()
#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#TRACK PARTICLES THROUGH ELECTRIC FIELD

simulation_time=0
#calculate field transition times from bunch structure
#caution, only works for timesteps that are picosecond or larger
transition_times=[]
print("\n Calculating field transition times frombunch structure...")
#calculate field transition times and save them...values will be SAVED IN NANOSECONDS to reduce the danger of floating point arithmetic errors
for bunch_num in range(0,number_of_bunches):
     if bunch_num==0: 
          transition_times.append(bunch_length*1e9)
          print(bunch_length)
     else: #the rounding and the *1e9 and /1e9 operations are to deal with floating point arithmetic errors that creep in here due to the small timestep values
          transition_times.append(np.around(((bunch_length*(bunch_num))+(bunch_spacing*(bunch_num)))*1e9,decimals=6))
          transition_times.append(np.around(((bunch_length*(bunch_num+1))+(bunch_spacing*(bunch_num)))*1e9,decimals=6))
          print(np.around(((bunch_length*(bunch_num))+(bunch_spacing*(bunch_num)))*1e9,decimals=6)/1e9)
          print(np.around(((bunch_length*(bunch_num+1))+(bunch_spacing*(bunch_num)))*1e9,decimals=6)/1e9)
for time_value in transition_times:
     #check that all of the calculated field transition times are divisible by the simulation timestep, so that the code will be able to stop the simulation and swap fields whenever needed
     if (time_value)%(input_timestep*1e9)!=0: #*1e9 to remove floating poiont arithmetic errors
          print("Transition time = "+str(time_value)+"ns, Simulation timestep = "+str(input_timestep*1e9)+"ns.")
          print("****************************************")
          print("* ERROR - Timestep and bunch structure *")
          print("* are not divisible. Beam field cannot *")
          print("* be swapped on/off at correct times.  *")
          print("*    Please change the specified       *")
          print("*  timestep and re-run the program.    *")
          print("****************************************")
          sys.exit()
transition_times.append(np.round(tracking_steps*input_timestep*1e9,decimals=6)+1)#append a final value to the transition times that will never be reached, to prevent indexing errors in the code later on

#PARTICLE TRACKING------------------------------------------------------------------------------------------------------------------------------------------------------------------------
running_times['start tracking']=time.time()
print("Tracking "+str(particle_num)+" particles through "+str(tracking_steps)+" timesteps.\nPlease wait...")
print("0 ns tracked.",end=""  ) #end="" stops the output moving to a new line after printing this message (used to create a tidy progress readout in the output console)
transition_count=0
for i in range (0,tracking_steps):
    count=0
    for particle in reversed(particles): #iterate through the particles list in reversed order, so that if any particle objects are deleted from the particles list, the iteration onto the next element is not affected by the change in the list's overall size (when iterating from start to finish, if a particle is deleted, the element numbers for the remaining particles are reduced by 1, so they iterator misses out the next particle in the list)
        if particle.creation_time <= simulation_time:
             particle.move(timestep=input_timestep, plot=True) #move all particles unless they have not yet been generated
             count=count+1
    simulation_time=simulation_time+input_timestep
    #print("Simulation Time = "+str(simulation_time))
    print("\r"+str(i+1)+" ns tracked. There are "+str(count)+" particles being tracked.", end="")#re-write to the console with an updated progress message. "\r" prints from the start of the current line on the console, to overwrite the previous time printout
    if (np.round(simulation_time*1e9,decimals=6)==transition_times[transition_count]):
         transition_count=transition_count+1
         del efield, numpy_efield #delete the existing electric fields stored in memory to reduce overall memory usage
         if transition_count%2==1: #if the transition count is odd, a bunch is leaving the monitor (simulation always begins with a bunch in the monitor)
              filepath=filepath_withbeam
              print("\nThe particle bunch is leaving the monitor - changing E_Field...\n")
         else:
              filepath=filepath_nobeam
              print("\nA particle bunch is entering the monitor - changing E_Field...\n")
         efield=import_CST_EField(filepath,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
         numpy_efield=efield[['Ex','Ey','Ez']].to_numpy()         
print("\nThere are "+str(len(particles))+" particles remaining in the simulation region.")
running_times['finished tracking']=time.time()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ANALYSE RESULTS####################################################################################################################

#generate an array of particles which reached the detector
detected_particles=np.asarray([[particle.final_x,particle.final_y,particle.final_z,particle.initial_x,particle.initial_y,particle.initial_z,particle.creation_time,particle.lifetime] for particle in destroyed_particles if particle.final_y==116 and particle.final_x>=detector_xmin and particle.final_x<=detector_xmax and particle.final_z>=detector_zmin and particle.final_z<=detector_zmax])
if np.size(detected_particles) == 0 :
     print ("***There were no particles that reached the detectors.\n --> Consider increasing the number or length of timesteps, or double check the position of the detector")
else:
     print("The number of particles reaching the detectors is: "+str(np.size(detected_particles[:,0])))
#generate an array containing ALL particle data for ease of plotting
for particle in particles:
     particle.final_x=particle.x
     particle.final_y=particle.y
     particle.final_z=particle.z
all_particles=np.append(particles,destroyed_particles)
#generate an array of each particle's initial position
initial_positions=np.asarray([[particle.initial_x,particle.initial_y,particle.initial_z] for particle in all_particles])
#generate an array of each particle's final position
final_positions=np.asarray([[particle.x,particle.y,particle.z] for particle in all_particles])
#generate an array of particle time properties
particle_time_structure=np.asarray([[particle.creation_time,particle.lifetime] for particle in all_particles])

particle_tracks=[]
untracked_particles=0
#generate an array of arrays, each containing a single particles trajectory data for 3D plotting (this is necessary as each particle trajectory is different lengths, once the NaN values are removed each row will be different lengths so needs to be in separate arrays)
for i in range(0,np.size(trajectories[:,0,0])): #selects each particle from the trajectories array
     #INDEXING: for trajectories[i,j,k]: i represents particle number, j is the time in the simulation, k is an [x,y,z] array containing position data
     x_positions=trajectories[i,:,0][~np.isnan(trajectories[i,:,0])] #get every x position for he particle that is not a NaN value (~ is the numpy not operator)
     y_positions=trajectories[i,:,1][~np.isnan(trajectories[i,:,1])]
     z_positions=trajectories[i,:,2][~np.isnan(trajectories[i,:,2])]
     #if particles were generated outside of the simulation region, they wont have any trajectory information, so np.size will return 0. If this is the case, skip and move on to the next particle
     if np.size(x_positions)!=0:
          if y_positions[-1]==116 and x_positions[-1]>=detector_xmin and x_positions[-1]<=detector_xmax and z_positions[-1]>=detector_zmin and z_positions[-1]<=detector_zmax: 
               detected=True
          else: detected=False
          particle_tracks.append([x_positions,y_positions,z_positions,detected])
     else: untracked_particles=untracked_particles+1

particle_num=particle_num-untracked_particles


#PLOT DATA##########################################################################################################################

#%%
     
#Plot the initial and final particle distributions in the transverse plane
plt.figure(figsize=(15,10))
plt.subplot(2,4,1)
if not plot_detected_only:
     plt.scatter(initial_positions[:,0],initial_positions[:,1],s=1, color='orange')
     plt.scatter(final_positions[:,0],final_positions[:,1],s=1, color='C0')
else:
     if np.size(detected_particles)>0:
          plt.scatter(detected_particles[:,3],detected_particles[:,4],s=1, color='orange')
          plt.scatter(detected_particles[:,0],detected_particles[:,1],s=1, color='C0')
plt.title('Initial Particle Distribution:\n Transverse Plane')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis([-150,150,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

#plot the initial and final particle distributions in the longitudinal plane
plt.subplot(2,4,2)
if not plot_detected_only:
     plt.scatter(initial_positions[:,2],initial_positions[:,1], s=1, color='orange')
     plt.scatter(final_positions[:,2],final_positions[:,1], s=1, color='C0')

else:
     if np.size(detected_particles)>0:
          plt.scatter(initial_positions[:,2],initial_positions[:,1],s=1, color='lightgray')
          plt.scatter(detected_particles[:,5],detected_particles[:,4], s=1, color='orange')
          plt.scatter(detected_particles[:,2],detected_particles[:,1], s=1, color='C0')
plt.title('Initial/Final Particle Distribution\n Longitudinal Plane')
plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.axis([min_z,max_z,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

#Plot rough particle tracks using initial and final positions
plt.subplot(2,4,3)
if not plot_detected_only:
     #Rough 2-point trajectories
     #plt.plot(([initial_positions[:,0],final_positions[:,0]]),([initial_positions[:,1],final_positions[:,1]]), '--', linewidth=0.25)
     #full trajectories (slower to process)
     for i in range(0,particle_num):
          plt.plot(particle_tracks[i][0],particle_tracks[i][1],'--',linewidth=0.5)
else:
     if np.size(detected_particles)!=0:
          #Rough 2-point trajectories
#          plt.plot(([detected_particles[:,3],detected_particles[:,0]]),([detected_particles[:,4],detected_particles[:,1]]), '--', linewidth=0.25)
          #Full accurate trajectory Plot (slower to process)
          for i in range(0,particle_num):
               if not particle_tracks[i][3]: #check if the particle reached the detector or not
                    junk=1
#                    if np.random.rand() >= 0.9: #plot 10% of the tracks for undetected particles
#                         plt.plot(particle_tracks[i][0],particle_tracks[i][1],'--',linewidth=0.5,color='lightgray')
               else: #plot all detected particle trajectories
                    plt.plot(particle_tracks[i][0],particle_tracks[i][1],linewidth=0.5)
plt.title('Detected Particle Trajectories:\n Horizontal Plane')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis([-150,150,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

plt.subplot(2,4,4)
if not plot_detected_only:
     #Rough 2-point trajectories
#     plt.plot(([initial_positions[:,2],final_positions[:,2]]),([initial_positions[:,1],final_positions[:,1]]), '--', linewidth=0.25)
     #full trajectories (slower to process)
     for i in range(0,particle_num):
          plt.plot(particle_tracks[i][2],particle_tracks[i][1],'--',linewidth=0.5)
else:
     if np.size(detected_particles)>0:
          #Rough 2-point trajectories
          #plt.plot(([detected_particles[:,5],detected_particles[:,2]]),([detected_particles[:,4],detected_particles[:,1]]), '--', linewidth=0.25)
     #full trajectories (slower to process)
          for i in range(0,particle_num):
               if not particle_tracks[i][3]: #check if particle reached the detector or not
                    junk=1
#                    if np.random.rand() >= 0.9: #plot 10% of the tracks for undetected particles
#                         plt.plot(particle_tracks[i][2],particle_tracks[i][1],'--',linewidth=0.2,color='lightgray')
               else: #and plot all the trajectories of detected particles
                    plt.plot(particle_tracks[i][2],particle_tracks[i][1],linewidth=0.5)                    
          
plt.title('Detected Particle Trajectories:\n Longitudinal Plane')
plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.axis([min_z,max_z,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

#plot time distribution of detected particles
bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))
plt.subplot(2,4,5)
plt.hist(particle_time_structure[:,0]*1e9, bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)), edgecolor='black', color='orange')
plt.title("Time Structure of Particles Created\n Resolution = "+str(time_resolution*1e9)+"ns")
plt.xlabel('Time Elapsed(ns)')
plt.ylabel('Particles Detected')   

bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))
plt.subplot(2,4,6)
plt.title("Time Structure of Detected Particles\n Resolution = "+str(time_resolution*1e9)+"ns")
plt.xlabel('Time Elapsed(ns)')
plt.ylabel('Particles Detected')     
if np.size(detected_particles) > 0:
     plt.hist((detected_particles[:,6]+detected_particles[:,7])*1e9, bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)), edgecolor='black')
     
plt.subplot(2,4,7)
plt.hist(initial_positions[:,0],bins=40, range=(-120,120), edgecolor='black', color='orange')
plt.title("Initial Beam Profile")

plt.subplot(2,4,8)
plt.title("Beam Profile At Channeltrons")
if np.size(detected_particles) > 0:
     plt.hist(detected_particles[:,0],bins=40, range=(-120,120), edgecolor='black')

#set the layout up so there is space for information text below the plots
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.figtext(0.05,0.09,'Simulation Details',weight='bold')
plt.figtext(0.05,0.07,'Electric Field Information: Imported Step Size = '+str(step_size)+' mm, X Max = '+str(max_x)+' mm, X Min = '+str(min_x)+", Y Max = "+str(max_y)+' mm, Y Min = '+str(min_y)+', Z Max = '+str(max_z)+' mm, Z Min = '+str(min_z)+'mm')
plt.figtext(0.05,0.05,'Type of particle tracked = '+particle.species)
plt.figtext(0.05,0.03,'Electric Field Filepath 1: '+filepath_withbeam)
plt.figtext(0.05,0.01,'Electric Field Filepath 2: '+filepath_nobeam)

if ibicplots==False: 
     print('Tracking Simulation Complete - Please run specific plotting cells manually as required.')
     sys.exit()
#plot particle trajectories
#
#plt.figure()
#plt.title("Particle Trajectories within the IPM")
#ax=plt.axes(projection="3d")
#ax.set_xlabel('z (mm)')
#ax.set_ylabel('x (mm)')
#ax.set_zlabel('y (mm)')
#for i in range(0,particle_num):
#     if particle_tracks[i][3]: #check if particle reached the detector or not  
#          ax.plot3D(particle_tracks[i][2],particle_tracks[i][0],particle_tracks[i][1])
     #plt.plot(particle_tracks[i][0],particle_tracks[i][1],particle_tracks[i][2])

#%%
#---------------------------------------------------------------------------------------------------------------------------------
#plot a comparison of profiles
#generate histograms, each containing 2 arrays. Arrays at [0] contains profile data, arrays at [1] contain the position of each profile data point (e.g. channeltron position, harp wire position)
ideal_profile_data=np.histogram(initial_positions[:,0],bins=40, range=(-120,120))
ideal_profile=normalise_profile_area(ideal_profile_data[0])
profile_data=np.histogram(detected_particles[:,0],bins=40, range=(-120,120))
channeltron_positions=profile_data[1]-3
channeltron_positions=np.delete(channeltron_positions,0)
detected_profile=normalise_profile_area(profile_data[0])


#interpolate the profiles with 1000 points to make analysis more accurate (e.g. 95% width calculation)
interpolated_channeltron_positions=np.linspace(-120,120,1000)
interpolated_ideal_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,ideal_profile)
interpolated_detected_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,detected_profile)

#normalise the harp profile and create and interpolated version for plotting comparisons
hor_harp=normalise_profile_area(hor_harp)
interpolated_harp_wire_positions=np.linspace(-69,69,1000)
interpolated_hor_harp_profile=np.interp(interpolated_harp_wire_positions,harp_wire_positions,hor_harp)


#calculate 95% widths and peak centres
width_percent=95
ideal_95w,ideal_centre=calculate_percent_width(interpolated_channeltron_positions,interpolated_ideal_profile,percentage=width_percent)
detected_95w,detected_centre=calculate_percent_width(interpolated_channeltron_positions,interpolated_detected_profile,percentage=width_percent)
harp_95w,harp_centre=calculate_percent_width(interpolated_harp_wire_positions,interpolated_hor_harp_profile,percentage=width_percent)


###################IBIC 2019 MANUAL DATA INPUT ################################################################
#do the same for the MCPM data daken during machine physics measurements

#Get a profile measured using the fast amplifiers on the ISIS EPB MCPM
ctron_time_data,MCPM_integrated_profile,MCPM_filepath=get_fast_amplifier_profile() #integrated data: index 0 contains single channel data, which should be ignored. Indexes 1-40 contain MCPM data
print("MCPM Profile Data: "+str(MCPM_integrated_profile))
backup_MCPM_integrated_profile=MCPM_integrated_profile
#MCPM_integrated_profile=[0.023,0.051,-0.020,0.120,0.003,0.036,0.048,0.309,0.246,0.749,2.062,1.284,6.159,7.387,9.593,10.068,12.813,10.268,12.927,11.854,12.905,11.220,13.474,15.672,9.889,9.427,5.919,6.155,2.333,1.390,0.774,0.035,0.073,0.072,-0.353,-0.039,0.035,0.013,0.596,-0.004]
MCPM_profile=normalise_profile_area(MCPM_integrated_profile)
interpolated_MCPM_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,MCPM_profile)
MCPM_95w,MCPM_centre,MCPM_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_MCPM_profile,percentage=width_percent,hhw=True)
#IBIC2019 Measurements Manual Input of each time peak
#first small peak in measurement, with bias set high to make it visible
#INTEGRATION WINDOW = 4.4-4.7us
#calibrated
peak1_profile=[0.095,0.046,-0.023,-0.026,1.438,0.606,1.819,0.781,1.120,0.328,0.893,0.183,1.493,1.846,2.324,2.215,2.014,1.863,3.734,2.407,2.625,2.090,2.756,2.289,2.072,2.196,1.408,1.382,0.934,0.561,0.529,0.876,0.232,0.189,0.975,0.163,1.095,-0.047,-0.839,0.082]
#raw
#peak1_profile=[0.355,0.280,0.496,0.062,0.376,0.754,0.332,0.667,0.959,1.206,1.234,0.145,1.391,1.843,1.387,0.849,0.443,0.601,0.826,1.667,2.661,0.438,1.622,0.673,1.733,0.773,0.527,0.706,1.460,0.312,1.185,0.649,0.326,0.789,0.264,0.511,0.100,0.144,0.100,0.425]

peak1_profile=np.asarray(peak1_profile)
peak1_profile=normalise_profile_area(peak1_profile)
interpolated_peak1_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak1_profile)
peak1_95w,peak1_centre,peak1_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak1_profile,percentage=width_percent,hhw=True)

#first small peak in measurement, with bias set high to make it visible
#INTEGRATION WINDOW = 4.7-5us
#peak2_profile=[0.105,0.143,-0.038,-0.030,2.597,0.972,3.867,1.625,1.923,0.625,1.740,0.598,3.170,3.856,4.897,4.923,4.945,4.337,6.347,5.532,5.348,5.551,6.116,6.346,4.018,4.475,3.369,2.806,1.711,1.218,1.095,1.378,0.441,0.330,2.054,0.203,1.635,-0.128,-1.024,0.051]
#ALTERNATE INTEGRATION WINDOW 4.75-5
#calibrated
peak2_profile=[0.010,0.007,-0.003,-0.004,0.110,0.040,0.090,0.071,0.062,0.020,0.055,0.041,0.086,0.128,0.174,0.152,0.098,0.077,0.158,0.234,0.253,0.147,0.241,0.203,0.199,0.091,0.123,0.117,0.077,0.035,0.035,0.058,0.010,0.017,0.041,0.009,0.023,-0.001,0.010,-0.001]
#raw
#peak2_profile=[0.040,0.012,0.023,0.157,0.033,0.049,0.011,0.165,0.057,0.081,0.068,0.081,0.100,0.108,0.074,0.086,0.028,0.020,0.047,0.177,0.197,0.011,0.148,0.079,0.164,0.052,0.043,0.090,0.124,0.028,0.061,0.114,0.016,0.099,0.024,0.084,-0.002,0.009,0.010,0.081]

peak2_profile=np.asarray(peak2_profile)
peak2_profile=normalise_profile_area(peak2_profile)
interpolated_peak2_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak2_profile)
peak2_95w,peak2_centre,peak2_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak2_profile,percentage=width_percent, hhw=True)

#first small peak in measurement, with bias set high to make it visible
#INTEGRATION WINDOW = us
#calibrated
peak3_profile=[-0.000,0.004,-0.000,0.048,0.043,0.013,-0.073,-0.219,0.059,0.095,0.436,-0.023,0.751,0.955,1.149,1.002,0.879,0.907,1.186,1.258,1.544,0.785,1.643,1.357,1.408,0.839,0.897,0.771,0.511,0.130,0.099,0.023,0.002,0.002,0.015,0.004,-0.106,-0.005,-0.037,0.002]
#raw
#peak3_profile=[-0.001,-0.004,-0.000,-0.174,-0.005,0.005,-0.011,-0.074,0.069,0.331,0.443,0.063,0.841,0.963,0.624,0.439,0.185,0.272,0.338,0.858,1.480,0.158,0.991,0.334,1.236,0.307,0.331,0.370,0.828,0.067,0.272,0.029,0.007,0.016,0.013,0.023,0.001,0.021,0.030,0.021]

peak3_profile=np.asarray(peak3_profile)
peak3_profile=normalise_profile_area(peak3_profile)
interpolated_peak3_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak3_profile)
peak3_95w,peak3_centre,peak3_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak3_profile,percentage=width_percent,hhw=True)

#first small peak in measurement, with bias set high to make it visible
#INTEGRATION WINDOW = 5.3-5.7us
#calibrated
peak4_profile=[0.008,0.003,-0.000,0.020,0.030,0.012,0.006,-0.081,0.002,0.169,1.249,0.485,3.019,4.125,5.803,5.596,4.748,4.571,6.740,6.235,6.213,5.094,7.089,6.649,6.141,3.946,4.107,3.491,1.671,0.509,0.045,-0.053,-0.009,0.001,-0.053,-0.003,0.036,0.009,0.066,-0.001]
#raw
#peak4_profile=[-0.012,0.015,0.027,0.205,0.009,0.013,0.005,-0.024,-0.026,0.309,0.830,0.144,2.001,2.445,1.648,1.150,0.655,0.924,1.113,2.300,3.887,0.563,2.691,0.923,3.528,0.868,0.928,0.964,1.560,0.122,0.057,-0.031,-0.003,0.006,-0.008,0.004,-0.004,-0.004,0.007,0.024]

peak4_profile=np.asarray(peak4_profile)
peak4_profile=normalise_profile_area(peak4_profile)
interpolated_peak4_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak4_profile)
peak4_95w,peak4_centre,peak4_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak4_profile,percentage=width_percent, hhw=True)

#first small peak in measurement, with bias set high to make it visible
#INTEGRATION WINDOW = 5.7-6us
#calibrated
peak5_profile=[-0.003,0.002,-0.002,-0.043,-0.024,-0.002,0.089,0.150,0.006,0.007,0.021,0.125,0.890,1.902,3.141,3.415,3.406,3.410,4.606,4.143,4.050,3.955,4.225,4.600,3.520,2.111,1.546,0.876,0.107,-0.003,0.006,-0.012,-0.001,0.004,0.003,-0.001,-0.082,-0.001,0.003,-0.002]
#raw
#peak5_profile=[0.028,0.012,0.002,0.226,0.022,0.021,0.008,0.158,0.005,0.032,0.018,0.075,0.554,1.096,0.893,0.810,0.463,0.710,0.793,1.631,2.561,0.455,1.710,0.680,2.076,0.481,0.349,0.215,0.057,-0.009,0.018,-0.037,0.007,0.009,0.011,-0.031,0.006,0.020,0.006,-0.009]

peak5_profile=np.asarray(peak5_profile)
peak5_profile=normalise_profile_area(peak5_profile)
interpolated_peak5_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak5_profile)
peak5_95w,peak5_centre,peak5_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak5_profile,percentage=width_percent, hhw=True)

#calibrated
peaks1_2_profile=[0.188,0.133,-0.058,-0.043,2.247,1.138,3.337,1.243,1.877,0.592,1.891,0.310,2.479,3.357,4.591,4.103,3.501,3.206,5.131,4.534,4.880,3.992,5.166,4.687,3.576,3.480,2.369,2.644,1.490,0.916,0.904,1.252,0.392,0.349,2.094,0.215,1.941,-0.099,-1.481,0.107]
peaks1_2_profile=np.asarray(peaks1_2_profile)
peaks1_2_profile=normalise_profile_area(peaks1_2_profile)
interpolated_peaks1_2_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peaks1_2_profile)
peaks1_2_95w,peaks1_2_centre,peak1_2_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peaks1_2_profile,percentage=width_percent, hhw=True)
##################################################################################################################


plt.figure()
plt.title("Comparison of Profiles\n (Area Under Profiles Normalised)")
plt.xlabel('x (mm)')
plt.ylabel('Detected Particles (arbitrary units)')
plt.plot(interpolated_channeltron_positions, interpolated_detected_profile)
plt.plot(interpolated_harp_wire_positions, interpolated_hor_harp_profile,'--',color='red')
plt.plot(interpolated_channeltron_positions, interpolated_ideal_profile,'--')
plt.plot(interpolated_channeltron_positions, interpolated_MCPM_profile)
plt.legend(['Simulated IPM\nMeasurement','EPM 26A Harp \nMonitor Profile','Generated Beam\nDistribution\nIn Simulation','Measured Profile\nwith EPB1 MCPM'], loc='upper right')
plt.figtext(0.05,0.16,'Electric Field Information: Imported Step Size = '+str(step_size)+' mm, X Max = '+str(max_x)+' mm, X Min = '+str(min_x)+", Y Max = "+str(max_y)+' mm, Y Min = '+str(min_y)+', Z Max = '+str(max_z)+' mm, Z Min = '+str(min_z)+'mm')
plt.figtext(0.05,0.14,'Actual Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(ideal_95w))+"mm")
plt.figtext(0.05,0.12,'Simulated Measurement: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(detected_95w))+"mm")
plt.figtext(0.05,0.10,'Harp Beam: '+str(width_percent)+'% Width'+str("{:10.2f}".format(harp_95w))+"mm")
plt.figtext(0.05,0.08,'MCPM Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(MCPM_95w))+"mm")
plt.figtext(0.05,0.06,'Electric Field 1 Filepath: '+filepath_withbeam)
plt.figtext(0.05,0.04,'Electric Field 2 Filepath: '+filepath_nobeam)
plt.figtext(0.05,0.02,'MCPM Data Filepath: '+MCPM_filepath)
plt.subplots_adjust(bottom=0.27)
plt.show()

plt.figure()
plt.title("Comparison of Simulated and Measured Profiles\n (Area Under Profiles Normalised)")
plt.xlabel('x (mm)')
plt.ylabel('Detected Particles (arbitrary units)')
plt.plot(interpolated_channeltron_positions, interpolated_detected_profile)
plt.plot(interpolated_channeltron_positions, interpolated_MCPM_profile)
#plt.plot(interpolated_channeltron_positions, interpolated_peak1_profile)
plt.legend(['Simulated IPM\nMeasurement','Measured Profile\nwith EPB1 MCPM'], loc='upper right')
plt.figtext(0.05,0.18,'Electric Field Information: Imported Step Size = '+str(step_size)+' mm, X Max = '+str(max_x)+' mm, X Min = '+str(min_x)+", Y Max = "+str(max_y)+' mm, Y Min = '+str(min_y)+', Z Max = '+str(max_z)+' mm, Z Min = '+str(min_z)+'mm')
plt.figtext(0.05,0.16,'Actual Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(ideal_95w))+"mm")
plt.figtext(0.05,0.14,'Simulated Measurement: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(detected_95w))+"mm")
plt.figtext(0.05,0.12,'MCPM Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(MCPM_95w))+"mm")
#plt.figtext(0.05,0.10,'Peak #1 Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(peak1_95w))+"mm")
plt.figtext(0.05,0.08,'Type of particle tracked = '+particle.species)
plt.figtext(0.05,0.06,'Electric Field 1 Filepath: '+filepath_withbeam)
plt.figtext(0.05,0.04,'Electric Field 2 Filepath: '+filepath_nobeam)
plt.figtext(0.05,0.02,'MCPM Data Filepath: '+MCPM_filepath)
plt.subplots_adjust(bottom=0.27)
plt.show()

plt.figure()
plt.title("Comparison of Measured Profiles From Each Individual Time Peak\n (Area Under Profiles Normalised)")
plt.xlabel('x (mm)')
plt.ylabel('Detected Particles (arbitrary units)')
plt.plot(interpolated_channeltron_positions, interpolated_detected_profile)
plt.plot(interpolated_channeltron_positions, interpolated_MCPM_profile)
plt.plot(interpolated_channeltron_positions, interpolated_peak1_profile,'--')
plt.plot(interpolated_channeltron_positions, interpolated_peak2_profile,'--')
plt.plot(interpolated_channeltron_positions, interpolated_peak3_profile,'--')
plt.plot(interpolated_channeltron_positions, interpolated_peak4_profile,'--')
plt.plot(interpolated_channeltron_positions, interpolated_peak5_profile,'--')
#plt.plot(interpolated_channeltron_positions, interpolated_peaks1_2_profile,'--')
plt.legend(['Simulated IPM\nMeasurement','Measured Profile\nwith EPB1 MCPM','Measured Profile\nPeak #1 Only','Measured Profile\nPeak #2 Only','Measured Profile\nPeak #3 Only','Measured Profile\nPeak #4 Only','Measured Profile\nPeak #5 Only'], loc='upper right')
plt.figtext(0.05,0.2,'Electric Field Information: Imported Step Size = '+str(step_size)+' mm, X Max = '+str(max_x)+' mm, X Min = '+str(min_x)+", Y Max = "+str(max_y)+' mm, Y Min = '+str(min_y)+', Z Max = '+str(max_z)+' mm, Z Min = '+str(min_z)+'mm')
plt.figtext(0.05,0.18,'Actual Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(ideal_95w))+"mm")
plt.figtext(0.05,0.16,'Simulated Measurement: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(detected_95w))+"mm")
plt.figtext(0.05,0.14,'MCPM Beam: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(MCPM_95w))+"mm")
plt.figtext(0.05,0.11,'Peak #1: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(peak1_95w))+"mm"+', Peak #2: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(peak2_95w))+"mm"+', Peak #3: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(peak3_95w))+"mm"+', Peak #4: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(peak4_95w))+"mm"+', Peak #5: '+str(width_percent)+'% Width = '+str("{:10.2f}".format(peak5_95w))+"mm")
plt.figtext(0.05,0.09,'Peak #1: FWHM = '+str("{:10.2f}".format(peak1_hhw))+"mm"+', Peak #2 FWHM = '+str("{:10.2f}".format(peak2_hhw))+"mm"+', Peak #3 FWHM = '+str("{:10.2f}".format(peak3_hhw))+"mm"+', Peak #4 FWHM = '+str("{:10.2f}".format(peak4_hhw))+"mm"+', Peak #5 FWHM = '+str("{:10.2f}".format(peak5_hhw))+"mm")
plt.figtext(0.05,0.07,'Type of particle tracked = '+particle.species)
plt.figtext(0.05,0.05,'Electric Field 1 Filepath: '+filepath_withbeam)
plt.figtext(0.05,0.03,'Electric Field 2 Filepath: '+filepath_nobeam)
plt.figtext(0.05,0.01,'MCPM Data Filepath: '+MCPM_filepath)
plt.subplots_adjust(bottom=0.27)
plt.show()

#%%

#drift field sweep of time data
ctron_time_data_3kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_150021.csv')
ctron_time_data_5kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_145927.csv') 
ctron_time_data_10kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_145829.csv') 
ctron_time_data_15kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_145800.csv') 
ctron_time_data_20kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_145631.csv') 
ctron_time_data_25kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_145542.csv') 
ctron_time_data_30kV,dummy1,dummy2=get_fast_amplifier_profile()#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_145523.csv')

time_axis_size_ns=(ctron_time_data[:,0].size*(1/60e6))*1e9

ctron_num=24
start_time=4250 #start time for the plot. Beam pickup in the monitor starts at about 4520
ctron_time_axis=np.linspace(8,time_axis_size_ns-8,ctron_time_data[:,0].size)
ctron_time_axis=ctron_time_axis-start_time
plt.figure()
plt.title('Measured Channeltron Signal vs. Time')
plt.xlabel('Time (ns)')
plt.ylabel('Measured Voltage (V)')
plt.plot(ctron_time_axis,ctron_time_data_3kV[:,ctron_num])
#plt.plot(ctron_time_axis,ctron_time_data_5kV[:,ctron_num])
plt.plot(ctron_time_axis,ctron_time_data_10kV[:,ctron_num])
#plt.plot(ctron_time_axis,ctron_time_data_15kV[:,ctron_num])
#plt.plot(ctron_time_axis,ctron_time_data_20kV[:,ctron_num])
plt.plot(ctron_time_axis,ctron_time_data_25kV[:,ctron_num])
#plt.plot(ctron_time_axis,ctron_time_data_30kV[:,ctron_num])
plt.xlim(0,4000)
#plt.legend(['3kV Drift Field','5kV Drift Field','10kV Drift Field','15kV Drift Field','20kV Drift Field','25kV Drift Field','30kV Drift Field'])
plt.legend(['3kV Drift Field','10kV Drift Field','25kV Drift Field'])
plt.figtext(0.05,0.06,'Beam Intensity = 2.27e13 ppp in EPB1, MCPM Bias = -1.33 kV')
plt.figtext(0.05,0.04,'Channeltron plotted = '+str(ctron_num))
plt.figtext(0.05,0.02,'Plot start time = '+str(start_time)+'ns')
plt.show()

plt.figure()
plt.title('Measured Channeltron Signal vs. Time', fontsize=20)
plt.xlabel('Time (ns)', fontsize=16)
plt.ylabel('Measured Voltage (V)', fontsize=16)
plt.plot(ctron_time_axis,ctron_time_data_3kV[:,ctron_num], linewidth=3)
plt.plot(ctron_time_axis,ctron_time_data_10kV[:,ctron_num], linewidth=3)
#plt.plot(ctron_time_axis,ctron_time_data_20kV[:,ctron_num])
plt.plot(ctron_time_axis,ctron_time_data_30kV[:,ctron_num], linewidth=3)
plt.xlim(0,4000)
plt.legend(['3kV Drift Field','10kV Drift Field','30kV Drift Field'], fontsize=16)
plt.figtext(0.05,0.06,'Filename = '+dummy2)
plt.figtext(0.05,0.045,'Beam Intensity = 2.62e13 ppp in EPB1, MCPM Bias = -1.5 kV')
plt.figtext(0.05,0.03,'Channeltron plotted = '+str(ctron_num))
plt.figtext(0.05,0.015,'Plot start time = '+str(start_time)+'ns')
plt.subplots_adjust(top=0.95,bottom=0.2,left=0.053,right=0.980)
plt.show()


running_times['program complete']=time.time()
running_times['TOTAL PARTICLE.MOVE TIME']=running_times['particle move finished'] #cumulative total from each function call
running_times['TOTAL CALCULATE GAMMA TIME']=running_times['calculate gamma finished'] #cumulative total from each function call
running_times['TOTAL FIELD BOUNDARY CHECK TIME']=running_times['field boundary check finish'] #cumulative total from each function call
running_times['PARTICLE TRACKING TOTAL TIME']=running_times['finished tracking']-running_times['start tracking']
running_times['BEAM GENERATION TOTAL TIME']=running_times['finished beam generation']-running_times['start of beam generation']
running_times['TOTAL PROGRAM TIME']=running_times['program complete']-running_times['start time']
print('*************************  SUMMARY  **********************************')
print("PROGRAM TIMING SUMMARY:")
print(running_times)
print("\n TOTAL TIME IN PARTICLE.MOVE = "+str(running_times['TOTAL PARTICLE.MOVE TIME'])+"\n TOTAL TRACKING TIME = "+str(running_times['PARTICLE TRACKING TOTAL TIME'])+"\n TOTAL EXECUTION TIME = "+str(running_times['TOTAL PROGRAM TIME']))
print('***************************  END  ************************************')

#######################################################################################################################################################################


#%%
#Manual Plot of peak detection times vs drift field for IBIC paper
drift_field_values=[3,5,10,15,20,25,30] #in kV
p1_times=[482,441,347,308,258,241,225] #in ns
p2_times=[924,790,692-40,641-40,590-40,574-40,558-40]
p3_times=[1561,1322,1042,919,825,725,693]
p4_times=[2335,1945,1510,1305,1175,1042,960]
p5_times=[3193,2576,1956,1673,1492,1376,1292]

plt.figure()
plt.rcParams["font.family"]="Century"
#plt.title('Average Peak Detection Times vs. Drift Field Strength', fontsize=20)
plt.xlabel('Drift Field Potential (kV)',fontsize=16)
plt.ylabel('Average Detection Time (ns)',fontsize=16)
plt.plot(drift_field_values,p1_times, marker="D")
plt.plot(drift_field_values,p2_times, marker="D")
plt.plot(drift_field_values,p3_times, marker="D")
plt.plot(drift_field_values,p4_times, marker="D")
plt.plot(drift_field_values,p5_times, marker="D")
plt.legend(['Peak #1', 'Peak #2','Peak #3','Peak #4','Peak #5'], fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()            


#%%
plt.figure()
plt.xlabel('Time (ns)', fontsize=16)
plt.ylabel('Measured Voltage (V)', fontsize=16)
plt.plot(ctron_time_axis,ctron_time_data_25kV[:,ctron_num], color='C5')
#plt.plot(ctron_time_axis,ctron_time_data_10kV[:,ctron_num], linestyle='dashed')
#plt.plot(ctron_time_axis,ctron_time_data_25kV[:,ctron_num])
plt.xlim(0,4000)
plt.legend(['25 kV Drift Field','10 kV Drift Field','25 kV Drift Field'], fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


#%%
#PLOT TIME STRUCTURE AGAIN BUT AS LINE GRAPH
bin_num=int(np.ceil((tracking_steps*input_timestep)/(2*time_resolution)))
bin_num=int(bin_num/1)
plt.figure()
plt.title("Time Structure of Detected Particles\n Resolution = "+str(2*time_resolution*1e9)+"ns")
plt.xlabel('Time Elapsed(ns)')
plt.ylabel('Particles Detected')     
if np.size(detected_particles) > 0:
     test=plt.hist((detected_particles[:,6]+detected_particles[:,7])*1e9, bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)), edgecolor='black')
times=test[1]+((test[1][1]-test[1][0])/2)
times=np.delete(times,-1)
creation_times=plt.hist(particle_time_structure[:,0]*1e9, bins=int(bin_num), range=(0,(tracking_steps*input_timestep*1e9)), edgecolor='black', color='orange')
plt.figure()
plt.xlabel('Time (ns)', fontsize=18)
plt.ylabel('Total Ions Detected\n(Arbitrary units)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(times,test[0]*25)
plt.plot(times,creation_times[0]*2.5, linestyle='dashed')
plt.title('Simulated IPM Measurement Time Structure\n',fontsize=20)
plt.legend(['Total No. Ions Detected','No. Ions Generated by Beam'], fontsize=16)
plt.figtext(0.05,0.03,'Electric Field Filepath 1: '+filepath_withbeam)
plt.figtext(0.05,0.01,'Electric Field Filepath 2: '+filepath_nobeam)
plt.ylim(0,10000)
plt.xlim(0,4000)
plt.show()

#%%
#Plot integrated profile and fudge calibration factors as it was calibrated while saturating...
#'C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Best_3D_Plot_shows_all_Things\\EPB1_MCPM_020919_151748.csv'
ctron_time_data_3d,dummy1,dummy2=get_fast_amplifier_profile('C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Best_3D_Plot_shows_all_Things\\EPB1_MCPM_020919_151748.csv')#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_150021.csv')
integrated_data=[]
for i in range(1,41):
     integrated_data.append(np.sum(ctron_time_data_3d[:,i]))
measured_cal=[0.046,0.517,0.434,0.796,0.187,0.968,0.232,1.467,0.827,0.801,3.147,0.943,1.742,2.144,2.901,0.951,2.638,1.326,4.458,0.893,1.21,2.295,2.878,3.252,1.966,1.349,0.816,0/812,2.237,0.663,0.32,0.302,0.237,0.582,0.122,0.155,0.227,-0.02,0.358,0.863]
measured_cal=np.asarray(measured_cal)
#manual_cal=[1,1,1,1,1,1,1,1,1,1,1.1,0.17,0.95,0.8,0.9,1,1.25,1.05,1,0.88,0.65,1.15,0.9,1.1,0.77,1,1.1,1,1,1,1,1,1,1,1,1,1,1,1,1]
manual_cal=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9,0.9,1,1,1,1,0.9,1,1,1.4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
manual_cal=np.asarray(manual_cal)
averaged=moving_average(integrated_data,5)
averaged=np.insert(averaged,0,0)
averaged=np.insert(averaged,0,0)
averaged=np.insert(averaged,-1,0)
averaged=np.insert(averaged,-1,0)
plt.figure()
plt.plot(integrated_data*manual_cal)
plt.plot(averaged)
plt.legend(['Manual Calibration File','Rolling Average of Raw Data'])
plt.show()
#%%
#PLOT 3D Graph of the data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

ctron_time_data_3d,dummy1,dummy2=get_fast_amplifier_profile('C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Best_3D_Plot_shows_all_Things\\EPB1_MCPM_020919_151748.csv')#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_150021.csv')#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_150021.csv')
start_time=4e-6#3000e-9
end_time=7e-6#6000e-9
start_sample=int(start_time/(1/60e6))
end_sample=int(end_time/(1/60e6))
num_samples=end_sample-start_sample
for i in range(1,41):
     ctron_time_data_3d[:,i]=ctron_time_data_3d[:,i]*manual_cal[i-1]

x = channeltron_positions
#y = np.linspace(0,1e-5,num=len(ctron_time_data_3d[:,1]))
y = np.linspace(start_time,end_time,num=num_samples)
y=(y*1e9)-4000
X, Y = np.meshgrid(x, y)
print(X.shape)
print(Y.shape)

z = ctron_time_data_3d[start_sample:end_sample,1:41]
Z = z.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#ax.view_init(elev=90, azim=0)
#ax.view_init(elev=0, azim=0)
#ax.set_ylim([0, 10])
plt.ylim(0,3000)
ax.set_ylabel('Time (ns)', fontsize=14)
#ax.set_xlabel('Horizontal Position (mm)', fontsize=14)
ax.set_zlabel('Measured Signal Strength (Arb.)', fontsize=14)
plt.show()

#%%
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

ctron_time_data_3d,dummy1,dummy2=get_fast_amplifier_profile()#'C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Best_3D_Plot_shows_all_Things\\EPB1_MCPM_020919_151748.csv')#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_150021.csv')#(filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\Drift_Field_Sweep_Setup_Settings\\EPB1_MCPM_020919_150021.csv')
#manual_cal=[1,1,1,1,1,1,1,1,1,1,1,0.6,1,1,1,1,0.85,0.9,0.9,1.5,1.8,0.7,1.4,1,1.9,1,1,1,1,0.75,1,1,1,1,1,1,1,1,1,1]
manual_cal=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.8,0.8,1,1.4,1,0.8,1,0.9,1,0.9,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
for i in range(1,41):
     ctron_time_data_3d[:,i]=ctron_time_data_3d[:,i]*manual_cal[i-1]
     
start_time=4.2e-6#3000e-9
plot_length=1e-6
end_time=start_time+plot_length#6000e-9
start_sample=int(start_time/(1/60e6))
end_sample=int(end_time/(1/60e6))
num_samples=end_sample-start_sample

np_mcpm_data=np.asarray(ctron_time_data_3d[:,1:41])
averaged_data = []
avg_num=5
for i in range(0, 600):
    averaged_data.append(moving_average(np_mcpm_data[i,:], avg_num))

averaged_data = np.asarray(averaged_data)
print(averaged_data.shape)
#x = channeltron_positions
x = (np.linspace(((avg_num-1)/2),(40-(avg_num-1)/2),40-(avg_num-1))*6)-120

#y = np.linspace(0,1e-5,num=len(ctron_time_data_3d[:,1]))
y = np.linspace(start_time,end_time,num=num_samples)
y=(y*1e9)-(start_time*1e9)
X, Y = np.meshgrid(x, y)
print(X.shape)
print(Y.shape)

z = averaged_data[start_sample:end_sample,:]
Z = z.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#ax.view_init(elev=90, azim=0)
#ax.view_init(elev=0, azim=0)
#ax.set_ylim([0, 10])
plt.ylim(0,plot_length*1e9)
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
ax.set_ylabel('Time (ns)', fontsize=14)
#ax.set_xlabel('Horizontal Position (mm)', fontsize=14)
ax.set_zlabel('Measured Signal Strength (Arb.)', fontsize=14)
ax.set_xticks([])
plt.show()

