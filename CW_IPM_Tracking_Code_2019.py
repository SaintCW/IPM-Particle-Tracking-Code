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

def open_file_dialogue():
     root=Tk()
     root.filename=filedialog.askopenfilename(title="Select CST EField File")
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
          #calculate new positions after one timestep
#          self.x=self.previous_x+(self.previous_vx*timestep)+(((self.charge*timestep*timestep*Ex)/(2*relativistic_mass))*1e6)
#          self.y=self.previous_y+(self.previous_vy*timestep)+(((self.charge*timestep*timestep*Ey)/(2*relativistic_mass))*1e6)
#          self.z=self.previous_z+(self.previous_vz*timestep)+(((self.charge*timestep*timestep*Ez)/(2*relativistic_mass))*1e6)
         
          
          #calculate new velocities after timestep
#          self.vx=self.previous_vx+(((self.charge*timestep*Ex)/(relativistic_mass))*1e6)
#          self.vy=self.previous_vy+(((self.charge*timestep*Ey)/(relativistic_mass))*1e6)
#          self.vz=self.previous_vz+(((self.charge*timestep*Ez)/(relativistic_mass))*1e6)

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
     '''Takes a python list as an input, converts it to a numpy array and normalises it'''
     data=np.asarray(data)
     data=data/data.max()
     return(data)

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

def check_tracking_boundaries(x,y,z,set_xmax=tracking_xmax ,set_xmin=tracking_xmin, set_ymax=tracking_ymax, set_ymin=tracking_ymin, set_zmax=tracking_zmax,set_zmin=tracking_zmin):
     if x > set_xmax or y > set_ymax or z > set_zmax or x < set_xmin or y < set_ymin or z < set_zmin:
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
     element_num=((np.int(((x-min_x)/step_size)+0.5))*(1+(y_size/2))*(1+(z_size/2)))+((np.int(((y-min_y)/step_size)+0.5))*(1+(z_size/2)))+(np.int(((z-min_z)/step_size)+0.5))
     return(element_num)


#######################################################################################################################################################################
#MAIN PROGRAM
print('**********************************************************************')
print("*               ISIS IPM PARTICLE TRACKING CODE             CCW v3.0 *")
print('**********************************************************************')
#Import and Analyse the Electric Field From CST
plt.close('all') #close anyt plots still open from previous runs of the code

filepath=open_file_dialogue() #let the user choose a filepath graphically
#filepath='C:\\Users\\vwa13369\\Desktop\\AccPhys 2016\\2019_Field_Maps\\-15kV_-1400VBias_2_27e13ppp_radx_54_9_rady_41_5_xoff_0_86_yoff_-2_9_CFOFF.txt'
running_times={'start time':time.time(),'field boundary check start':0,'field boundary check finish':0,'calculate gamma start':0,'calculate gamma finished':0,'particle move start':0,'particle move finished':0} #Log the time to calculate the execution time of the code

efield=import_CST_EField(filepath,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
#analyse the field map to obtain properties needed for data lookup in the field
nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z=analyse_field_map(efield,printout=True)
#calculate field boundaries - doing this here speeds up the runtime of functions that use these values in if statements
field_max_x=max_x+step_size/2
field_min_x=min_x-step_size/2
field_max_y=max_y+step_size/2
field_min_y=min_y-step_size/2
field_max_z=max_z+step_size/2
field_min_z=min_z-step_size/2

print("Imported Data Sample:")
print(efield.head())

#Load the electric field data from the pandas dataframe into numpy arrays to increase performance
numpy_efield=efield[['Ex','Ey','Ez']].to_numpy()
print("Electric field values loaded into seperate numpy array for fast access.")

#generate an ion distribution
particle_num=50000 #the number of particles that should be generated inside the monitor
tracking_steps=2000
input_timestep=1e-9
bunch_length=100e-9
bunch_spacing=225e-9
number_of_bunches=2
time_resolution=16e-9 #time resolution in seconds
plot_detected_only=True

store_trajectories=True
if store_trajectories:
     trajectories=np.empty(((particle_num,tracking_steps,3)))
     trajectories[:]=np.nan
     print("Particle trajectory array created with dimensions (particle number,number of timesteps,position data) = "+str(np.shape(trajectories)))
     print("***WARNING - CALCULATING PARTICLE TRAJECTORIES CAN BE MEMORY INTENSIVE.***\n***CONSIDER USING FEWER PARTICLES OR TIMESTEPS IF MEMORY IS A CONCERN***")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#GENERATE PARTICLE DISTIBUTION
beam_xrad=54.9
beam_yrad=41.5
beam_xpos=0.86
beam_ypos=-2.9
beam_length_mm=1000

detector_z_centre=detector_zmin+((detector_zmax-detector_zmin)/2)
beam_zmin=detector_z_centre-(beam_length_mm/2)
beam_zmax=detector_z_centre+(beam_length_mm/2)
print("Generating beam between: z_min="+str(beam_zmin)+"mm, and beam z_max="+str(beam_zmax)+"mm")
#make sure that the veam will not be generated outside of the simulation region
if beam_zmax > tracking_zmax: beam_zmax=tracking_zmax
if beam_zmin < tracking_zmin: beam_zmin=tracking_zmin
print('-----------------------------------------------------------------')
print("\nGenerating beam with "+str(number_of_bunches)+" bunches, with a bunch length of "+str(bunch_length*1e9)+"ns, and inter-bunch spacing of "+str(bunch_spacing*1e9)+"ns...\n")
running_times['start of beam generation']=time.time()
particle_counter=1
for j in range(0,number_of_bunches):
     # to calculate bunch start time, in seconds in the loop below, use: bunch_start_time=(j*(bunch_spacing+bunch_length))*1e9
     for i in range(0,int(np.round(particle_num/number_of_bunches))):
          #create a particle. --> Calculate the x and y positions as random numbers generated inside gaussian (normal) distributions for each axis. The distribution mean (loc) values are set to the beam offset positions, and the standard deviations (scale) are set to half of the beam width in CST (so that the 95% widths correspond to 2 signma, which is correct for a gaussian). Size is the number of points that the function generates
          particle_creation_time=((j*(bunch_spacing+bunch_length)*1e9)+np.random.randint(low=0,high=(bunch_length*1e9)+1))/1e9 #pick a random creation time in seconds (but in integer values of ns) for each particle that fits inside the bunch length
          particle_x=(np.random.normal(loc=beam_xpos,scale=0.5*beam_xrad,size=1)[0])
          particle_y=(np.random.normal(loc=beam_ypos,scale=0.5*beam_yrad,size=1)[0])
          particle_z=np.random.uniform(low=beam_zmin,high=beam_zmax)
          Particle(x=particle_x,y=particle_y,z=particle_z,creation_time=particle_creation_time, species='proton',ID=particle_counter-1)
          particle_counter=particle_counter+1
print("Particle beam generated, containing "+str(len(particles))+" particles.")
particles=np.array(particles)
print("Particle list converted to numpy array for faster tracking.")
running_times['finished beam generation']=time.time()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#TRACK PARTICLES THROUGH ELECTRIC FIELD
simulation_time=0
running_times['start tracking']=time.time()
print("Tracking "+str(particle_num)+" particles through "+str(tracking_steps)+" timesteps.\nPlease wait...")
print("0 ns tracked.",end=""  ) #end="" stops the output moving to a new line after printing this message (used to create a tidy progress readout in the output console)
for i in range (0,tracking_steps):
    count=0
    for particle in reversed(particles): #iterate through the particles list in reversed order, so that if any particle objects are deleted from the particles list, the iteration onto the next element is not affected by the change in the list's overall size (when iterating from start to finish, if a particle is deleted, the element numbers for the remaining particles are reduced by 1, so they iterator misses out the next particle in the list)
        if particle.creation_time <= simulation_time:
             particle.move(timestep=input_timestep, plot=True) #move all particles unless they have not yet been generated
             count=count+1
    simulation_time=simulation_time+input_timestep
    #print("Simulation Time = "+str(simulation_time))
    print("\r"+str(i+1)+" ns tracked. There are "+str(count)+" particles being tracked.", end="")#re-write to the console with an updated progress message. "\r" prints from the start of the current line on the console, to overwrite the previous time printout
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
#generate an array of arrays, each containing a single particles trajectory data for 3D plotting (this is necessary as each particle trajectory is different lengths, once the NaN values are removed each row will be different lengths so needs to be in separate arrays)
for i in range(0,np.size(trajectories[:,0,0])): #selects each particle from the trajectories array
     #INDEXING: for trajectories[i,j,k]: i represents particle number, j is the time in the simulation, k is an [x,y,z] array containing position data
     x_positions=trajectories[i,:,0][~np.isnan(trajectories[i,:,0])] #get every x position for he particle that is not a NaN value (~ is the numpy not operator)
     y_positions=trajectories[i,:,1][~np.isnan(trajectories[i,:,1])]
     z_positions=trajectories[i,:,2][~np.isnan(trajectories[i,:,2])]
     if y_positions[-1]==116 and x_positions[-1]>=detector_xmin and x_positions[-1]<=detector_xmax and z_positions[-1]>=detector_zmin and z_positions[-1]<=detector_zmax: 
          detected=True
     else: detected=False
     particle_tracks.append([x_positions,y_positions,z_positions,detected])



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

plt.tight_layout()

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