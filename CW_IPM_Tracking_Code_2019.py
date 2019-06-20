# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:34:35 2019

@author: vwa13369
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tkinter import filedialog
from tkinter import *

#Define global variables
#tracking region limits (from CST model, ymax is the detector co-ordinates, the z and x limits are the inside faces of the monitor body)
tracking_xmax=263
tracking_xmin=-263
tracking_ymax=116
tracking_ymin=-116
tracking_zmax=462.5
tracking_zmin=66.5


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
     def __init__(self,x,y,z,species,vx=0,vy=0,vz=0, lifetime=0):
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
          if species=='proton':
               self.mass=1.6726219e-27
               self.charge=-1.60217662e-19
          elif species=='electron':
               self.mass=9.10928e-31
               self.charge=-1.60217662e-19
          else:
               print('\n*********************ERROR****************************')
               print("* Particle species incorrectly defined during creation *")
               print('*********************ERROR****************************\n')
               sys.exit()
          self.lifetime=0
          self.plot_colour=np.random.choice(range(256),size=3)/256 #generate a colour to plot the particle with on any trajectory graphs. This keeps the colour consistent even if each timestep of movement is plotted with different plot commands
          #check if the particle has been created inside the CST model or not
          if check_tracking_boundaries(x=self.x,y=self.y,z=self.z): particles.append(self) #only create the particle if it's co-ordinates are within the simulation region
          else: print("Particle could not be generated - requested location falls outside simulation region")
     
     def move(self,timestep=1e-9,plot=False,final_track=False):
          if final_track==False: #if the particle is doing its final timestep, then these values have already been assigned and do not need to be assigned again
               #update the previous position values to the current position values
               self.previous_x=self.x
               self.previous_y=self.y
               self.previous_z=self.z
               self.previous_vx=self.vx
               self.previous_vy=self.vy
               self.previous_vz=self.vz
          field_row_number=lookup_field_value(self.previous_x,self.previous_y,self.previous_z,efield)
          
          #calculate relativistic gamma and use this to increase the particle mass to account for relativistic effects
          velocity_magnitude=(np.sqrt((self.previous_vx/1000)**2+(self.previous_vy/1000)**2+(self.previous_vz/1000)**2)) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
          relativistic_beta=velocity_magnitude/3e8
          relativistic_gamma=1/(np.sqrt(1-relativistic_beta**2))
          relativistic_mass=self.mass*relativistic_gamma
          
          #Move the particle through one timestep calculating new positions and velocities using the Lorentz force applied by the EField at the particle's location
          #calculate new positions after one timestep
          self.x=self.previous_x+(self.previous_vx*timestep)+(((self.charge*timestep*timestep*efield['Ex'][field_row_number])/(2*relativistic_mass))*1e6)
          self.y=self.previous_y+(self.previous_vy*timestep)+(((self.charge*timestep*timestep*efield['Ey'][field_row_number])/(2*relativistic_mass))*1e6)
          self.z=self.previous_z+(self.previous_vz*timestep)+(((self.charge*timestep*timestep*efield['Ez'][field_row_number])/(2*relativistic_mass))*1e6)
          
          #calculate new velocities after timestep
          self.vx=self.previous_vx+(((self.charge*timestep*efield['Ex'][field_row_number])/(relativistic_mass))*1e6)
          self.vy=self.previous_vy+(((self.charge*timestep*efield['Ey'][field_row_number])/(relativistic_mass))*1e6)
          self.vz=self.previous_vz+(((self.charge*timestep*efield['Ez'][field_row_number])/(relativistic_mass))*1e6)
          
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
                    if plot==True:plt.plot([self.previous_x,self.x],[self.previous_y,self.y], color=self.plot_colour)
          #If this is the final timestep, update particle lifetime, plot movement if requested, then destroy the particle
          if final_track==True:
               self.lifetime=self.lifetime+timestep
               if plot==True:plt.plot([self.previous_x,self.x],[self.previous_y,self.y], color=self.plot_colour)
               self.destroy()
               
     def perform_final_movement(self,field_row_number,plot): #use the particle's velocity, and the previous and new positions to precisely calculate the time taken to reach the tracking region boundary.
          #calculate relativistic gamma and use this to increase the particle mass to account for relativistic effects - the relativistic-adjusted mass has not been passed to this function so it must be recalculated
          velocity_magnitude=(np.sqrt((self.previous_vx/1000)**2+(self.previous_vy/1000)**2+(self.previous_vz/1000)**2)) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
          relativistic_beta=velocity_magnitude/3e8
          relativistic_gamma=1/(np.sqrt(1-relativistic_beta**2))
          relativistic_mass=self.mass*relativistic_gamma
          
          
          #identify which plane the particle went outside the tracking region on. Then calculate the time taken by solving the quadratic formula twice (once with a plus sign and once with a - sign). When solving the formula, replace the new position with the position of the tracking region boundary, so that the time taken to reach it is calculated accurately 
          if self.y > tracking_ymax:
               Ey=efield['Ey'][field_row_number]
               #calculate roots of the quadratic which calculates the final timestep (t_final = (-prev_vy +- sqrt(prev_vy**2-4(q*Ey/2*mass)(y-prev_y)))/(q*Ey/mass))
               a=(self.charge*Ey*1e6)/(2*relativistic_mass) #multiplied by 1e6 to give units of mm/s^2
               b=self.previous_vy
               c=self.previous_y-tracking_ymax
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))#np.root returns the two roots of the quadratic equation which is being solved to calculate the timestep required to reach the simulation boundary
          elif self.y < tracking_ymin:
               Ey=efield['Ey'][field_row_number]
               a=(self.charge*Ey*1e6)/(2*relativistic_mass)
               b=self.previous_vy
               c=self.previous_y-tracking_ymin
               final_timestep1,final_timestep2=np.roots([a,b,c])
               print("Final timestep roots are: "+str(final_timestep1)+"ns, or "+str(final_timestep2)+"ns.")
          elif self.x > tracking_xmax:
               Ex=efield['Ex'][field_row_number]
               a=(self.charge*Ex*1e6)/(2*relativistic_mass)
               b=self.previous_vx
               c=self.previous_x-tracking_xmax
               final_timestep1,final_timestep2=np.roots([a,b,c])
          elif self.x < tracking_xmin:
               Ex=efield['Ex'][field_row_number]
               a=(self.charge*Ex*1e6)/(2*relativistic_mass)
               b=self.previous_vx
               c=self.previous_x-tracking_xmin
               final_timestep1,final_timestep2=np.roots([a,b,c])
          elif self.z > tracking_zmax:
               Ez=efield['Ez'][field_row_number]
               a=(self.charge*Ez*1e6)/(2*relativistic_mass)
               b=self.previous_vz
               c=self.previous_z-tracking_zmax
               final_timestep1,final_timestep2=np.roots([a,b,c])
          elif self.z < tracking_zmin:
               Ez=efield['Ez'][field_row_number]
               a=(self.charge*Ez*1e6)/(2*relativistic_mass)
               b=self.previous_vz
               c=self.previous_z-tracking_zmin
               final_timestep1,final_timestep2=np.roots([a,b,c])
          else:
               print("******************ERROR*********************")
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
               print("********************ERROR**********************")
               print("*  Final movement timestep calculation error  *")
               print("*     (Both roots return negative times)      *")
               print("***********************************************")
               sys.exit()
          
          
          final_timesteps.append(final_timestep)
          #if final_timestep > 1e-9: print("Timestep too large...previous_y = "+str(self.previous_y)+", y = "+str(self.y)+", previous_vy = "+str(self.previous_vy)+", vy = "+str(self.y)+", Ey = "+str(Ey))
          self.move(timestep=final_timestep,final_track=True, plot=plot)

     def destroy(self):
          #log the position as the particle's final position
          self.final_x=self.x
          self.final_y=self.y
          self.final_z=self.z
          #remove the particle from the particles list
          particles.remove(self)
          #add the particles to a seperate list that stores the removed particles, in case extra analysis on these is required
          destroyed_particles.append(self)

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
     if x > (max_x+step_size/2) or y > (max_y+step_size/2) or z > (max_z+step_size/2) or x < (min_x-step_size/2) or y < (min_y-step_size/2) or z < (min_z-step_size/2):
          return(False) #return false if the particle is outside the simulation region
     else:
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
     if check_field_boundaries(x,y,z,field_data)==False:  #if the requested location falls outside the range of the imported data, exit the program      
          print('\n*********************ERROR****************************')
          print("* EField location requested which is outside of field*\n")
          print("*  data range - check imported field dimensions and  *\n")
          print("*                 program settings                   *")
          print('*********************ERROR****************************\n')
          sys.exit()
     else:     
          #find nearest value to the requested position
          element_num=((np.round(((x-min_x)/step_size)))*(1+(y_size/2))*(1+(z_size/2)))+((np.round(((y-min_y)/step_size)))*(1+(z_size/2)))+(np.round(((z-min_z)/step_size)))
          #REMOVED TO SPEED UP CODE, rely on field properties calculated by analyse field function in main code
          # element_num=((np.round(((x-field_data['x'].min())/step_size)))*(1+(y_size/2))*(1+(z_size/2)))+((np.round(((y-field_data['y'].min())/step_size)))*(1+(z_size/2)))+(np.round(((z-field_data['z'].min())/step_size)))
          
          return(element_num)


#######################################################################################################################################################################
#MAIN PROGRAM
print('**********************************************************************')
print("*               ISIS IPM PARTICLE TRACKING CODE             CCW v3.0 *")
print('**********************************************************************')

#Import and Analyse the Electric Field From CST
plt.close('all') #close anyt plots still open from previous runs of the code

filepath=open_file_dialogue() #let the user choose a filepath graphically
efield=import_CST_EField(filepath,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
#analyse the field map to obtain properties needed for data lookup in the field
nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z=analyse_field_map(efield,printout=True)

print("Imported Data Sample:")
print(efield.head())

#generate an ion distribution
particles=[] #an array to store all the particle objects in during tracking
destroyed_particles=[] #an array to store particle objects that are removed from the simulation, usually because they have moved outside of the simulation region
final_timesteps=[] #an array to view all the final timesteps calculated for particles - for use in debugging
particle_num=1000 #the number of particles that should be generated inside the monitor
tracking_steps=500

#create particles
beam_xrad=54.9
beam_yrad=41.5
beam_xpos=0.86
beam_ypos=-2.9
for i in range (0,particle_num):
     #create a particle. --> Calculate the x and y positions as random numbers generated inside gaussian (normal) distributions for each axis. The distribution mean (loc) values are set to the beam offset positions, and the standard deviations (scale) are set to half of the beam width in CST (so that the 95% widths correspond to 2 signma, which is correct for a gaussian). Size is the number of points that the function generates
     Particle(x=(np.random.normal(loc=beam_xpos,scale=0.5*beam_xrad,size=1)[0]),y=(np.random.normal(loc=beam_ypos,scale=0.5*beam_yrad,size=1)[0]),z=364,species='proton')
print("There are "+str(len(particles))+" particles generated in the initial distribution.")

print("Tracking "+str(particle_num)+" particles through "+str(tracking_steps)+" timesteps.\nPlease wait...")
print("0 ns tracked.",end=""  ) #end="" stops the output moving to a new line after printing this message (used to create a tidy progress readout in the output console)
for i in range (0,tracking_steps): 
    for particle in particles:
         particle.move(plot=False)
    print("\r"+str(i+1)+" ns tracked.", end="")#re-write to the console with an updated progress message. "\r" prints from the start of the current line on the console, to overwrite the previous time printout
print("\nThere are "+str(len(particles))+" particles remaining in the simulation region.")


#ANALYSE RESULTS####################################################################################################################
#generate an array of particle initial positions for plotting
#Only use particles that have completed tracking and therefore been moved to destroyed particles
#initial_positions[:,0] gives all the initial x positions. referencing with [:,1] or [:,2] would give y and z positions respectively
initial_positions=np.array([0,0,0]) #initialise the array with a dummy set of 0's to make sure it has the right shape for stacking
for particle in particles:
     particle_positions=np.array([particle.initial_x,particle.initial_y,particle.initial_z])
     initial_positions=np.vstack((initial_positions,particle_positions))
for particle in destroyed_particles: 
     particle_positions=np.array([particle.initial_x,particle.initial_y,particle.initial_z])
     initial_positions=np.vstack((initial_positions,particle_positions))
initial_positions=np.delete(initial_positions,0,0) #remove the dummy row from the top of the array

#generate an array of particle final positions for plotting
final_positions=np.array([0,0,0]) #initialise the array with a dummy set of 0's to make sure it has the right shape for stacking
for particle in particles: 
     particle_positions=np.array([particle.x,particle.y,particle.z])
     final_positions=np.vstack((final_positions,particle_positions))
for particle in destroyed_particles: 
     particle_positions=np.array([particle.final_x,particle.final_y,particle.final_z])
     final_positions=np.vstack((final_positions,particle_positions))
final_positions=np.delete(final_positions,0,0) #remove the dummy row from the top of the array

if np.size(destroyed_particles) > 0:
     detected_positions=np.array([0,0,0])
     for particle in destroyed_particles: 
          if particle.final_y == tracking_ymax:
               particle_positions=np.array([particle.final_x,particle.final_y,particle.final_z])
               detected_positions=np.vstack((detected_positions,particle_positions))
     detected_positions=np.delete(detected_positions,0,0) #remove the dummy row from the top of the array
if np.size(detected_positions) == 0 : print ("***There were no particles that reached the detectors.\n --> Consider increasing the number or length of timesteps, or double check the position of the detector")
else:print("The number of particles reaching the detectors is: "+str(np.size(detected_positions[:,0])))
#PLOT DATA##########################################################################################################################

#Plot the initial particle distribution
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.scatter(initial_positions[:,0],initial_positions[:,1],s=1)
plt.title('Initial Particle Distribution (2D)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis([-150,150,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

#plot final positions
plt.subplot(2,3,2)
plt.scatter(final_positions[:,0],final_positions[:,1], s=1)
plt.title('Final Particle Distribution (2D)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis([-150,150,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

#Plot rough particle tracks using initial and final positions
plt.subplot(2,3,3)
plt.plot(([initial_positions[:,0],final_positions[:,0]]),([initial_positions[:,1],final_positions[:,1]]), '--', linewidth=0.25)
plt.title('Particle Trajectories (2D)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis([-150,150,-150,150]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range


plt.subplot(2,3,5)
plt.hist(initial_positions[:,0],bins=40, range=(-120,120), edgecolor='black')
plt.title("Initial Beam Profile (!TEST!)")

if np.size(destroyed_particles) > 0:
     plt.subplot(2,3,6)
     plt.hist(detected_positions[:,0],bins=40, range=(-120,120), edgecolor='black')
     plt.title("Beam Profile At Channeltrons (!TEST!)")

plt.tight_layout()
plt.show()

"""
#test particle acceleration
testy=Particle(x=0,y=0,z=100,species='electron',vx=0,vy=0,vz=0,lifetime=0)
#print("Particle Created: Mass = "+str(testy.mass)+", Charge = "+str(testy.charge))
test_efield=-333.333 #20kV over a 60mm gap is 333.333V/mm
test_positions=[]
velocities=[]
betas=[]
total_energies=[]
kinetic_energies=[]
timestep=1e-12

for  i in range (0,15000):
     testy.previous_x=testy.x
     testy.previous_y=testy.y
     testy.previous_z=testy.z
     testy.previous_vx=testy.vx
     testy.previous_vy=testy.vy
     testy.previous_vz=testy.vz
     
     relativistic_beta=(np.sqrt((testy.previous_vx/1000)**2+(testy.previous_vy/1000)**2+(testy.previous_vz/1000)**2))/3e8 #calculate the magnitude of the 3D previous velocity vector and divide by 3e11 (speed of light in mm/s)
     if relativistic_beta > 1 : 
          relativistic_beta=0.99999999
          print("Warning, beta tried to exceed 1. \n --> Beta was reduced to 0.99999999 to stop the universe breaking")
     relativistic_gamma=1/(np.sqrt(1-relativistic_beta**2))
     mass=testy.mass*relativistic_gamma     
     total_energy=(relativistic_gamma*testy.mass*3e8*3e8)*6.242e12
     betas.append(relativistic_beta)
     total_energies.append(total_energy)
     kinetic_energies.append(total_energy-((testy.mass*3e8*3e8)*6.242e12))
     #print("Previous_Vx = "+str(testy.previous_vx)+"mm/s, Beta = "+str(relativistic_beta)+", Gamma = "+str(relativistic_gamma)+", Mass = "+str(mass)+"kg")
     #Move the particle through one timestep calculating new positions and velocities using the Lorentz force applied by the EField at the particle's location
     #calculate new positions after one timestep
     testy.x=testy.previous_x+(testy.previous_vx*timestep)+(((testy.charge*timestep*timestep*test_efield)/(2*mass))*1e6) #*10e6 to convert the calculation of acceleration into mm/s^2 (using an input Efield given in V/mm)
     testy.y=testy.previous_y+(testy.previous_vy*timestep)+(((testy.charge*timestep*timestep*0)/(2*mass))*1e6)
     testy.z=testy.previous_z+(testy.previous_vz*timestep)+(((testy.charge*timestep*timestep*0)/(2*mass))*1e6)
     #calculate new velocities after timestep
     testy.vx=testy.previous_vx+(((testy.charge*timestep*test_efield)/(mass))*1e6)
     testy.vy=testy.previous_vy+(((testy.charge*timestep*0)/(mass))*1e6)
     testy.vz=testy.previous_vz+(((testy.charge*timestep*0)/(mass))*1e6)
     test_positions.append(testy.x)
     velocities.append(testy.vx/1000)#store velocities as m/s for plotting, not mm.s
     #print("Charge = "+str(particle.charge)+", Mass = "+str(particle.mass)+", X Velocity = "+str(particle.vx))
plt.figure()
plt.plot(total_energies,velocities)
plt.plot(kinetic_energies,betas)
plt.title("Velocity vs. Proton Energy")
plt.xlabel("Energies [MeV]")
plt.ylabel("Relativistic Beta")
plt.legend(["Total Energy","Kinetic Energy"])

plt.figure()
plt.plot(velocities)
plt.title("Velocity (m/s) vs. Time (ps)")
#normalise all plots
test_positions=normalise_list(test_positions)
total_energies=normalise_list(total_energies)
kinetic_energies=normalise_list(kinetic_energies)
velocities=normalise_list(velocities)

plt.show()
"""

print('***************************  END  ************************************')
#######################################################################################################################################################################