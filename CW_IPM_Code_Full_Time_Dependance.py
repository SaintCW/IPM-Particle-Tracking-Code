# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:34:35 2019

@author: vwa13369
"""
#tracking region limits (from CST model, ymax is the detector co-ordinates, the z and x limits are the inside faces of the monitor body)
#detector_xmax=120
#detector_xmin=-120
#detector_y=116
#detector_zmax=374
#detector_zmin=354
##DOUBLE CHECK THESE SETTINGS WHEN DIMENSIONS ARE CHANGED, THEY DO NOT UPDATE IN THE CHECK_BOUNDARIES FUNCTION EVEN IF THEY ARE CHANGED IN THE MAIN PROGRAM LOOP
#tracking_xmax=263
#tracking_xmin=-263
#tracking_ymax=116
#tracking_ymin=-116
#tracking_zmax=462.5
#tracking_zmin=66.5

#import standard libraries that may be used
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

##################################################################################################################################################
          #####################################################
          #    @@@ @@@ @@@  @@@      @@@ @@@ @@@ @   @ @@@    #
          #    @   @ @ @  @ @       @    @    @  @   @ @  @   #
          #    @   @ @ @  @ @@       @@  @@   @  @   @ @@@    #
          #    @   @ @ @  @ @          @ @    @  @   @ @      #
          #    @@@ @@@ @@@  @@@     @@@  @@@  @   @@@  @      #
          #####################################################

#Define global variables
custom_timestructure=True #Set to false for faster, semi-time dependant space charge field (switched on/off for each bunch) or set to true to specify an accurate bunch time structure and scale the space charge field for each timestep
#tracking region limits (from CST model, ymax is the detector co-ordinates, the z and x limits are the inside faces of the monitor body)
detector_xmax=120
detector_xmin=-120
detector_y=116
detector_zmax=374.5 #374
detector_zmin=354.5 #354

beam_length_mm=900

particle_num=5000
tracking_steps=5000
##DOUBLE CHECK THESE SETTINGS WHEN DIMENSIONS ARE CHANGED, THEY DO NOT UPDATE IN THE CHECK_BOUNDARIES FUNCTION EVEN IF THEY ARE CHANGED IN THE MAIN PROGRAM LOOP
tracking_xmax=263
tracking_xmin=-263
tracking_ymax=110 #116  ...110 is the co-ordinate of the earth plate with a slot in it. 116 is the coordinate of the channeltrons themselves. Use 100 if you have defined a second area inside the earth plate slot to account for the angular cutoff of the earth plate
tracking_ymin=-110 #-116
tracking_zmax=detector_zmax+100
tracking_zmin=66.5
##EXTRA TRACKING BOUNDARIES FOR THE AREA INSIDE THE CHANNELTRON EARTH PLATE SLOT AND THE CHANNELTRONS THEMSELVES
earth_enclosure_slot_region=True
tracking_earthplate_xmax=detector_xmax
tracking_earthplate_xmin=detector_xmin
tracking_earthplate_ymax=detector_y
tracking_earthplate_ymin=110
tracking_earthplate_zmax=detector_zmax
tracking_earthplate_zmin=detector_zmin

#Drawings lines for the IPM to be visualised on a graph
#transverse cross seciton of monitor at detectors
transverse_cross_section_dwg_xarray=[tracking_xmin,tracking_xmin,tracking_earthplate_xmin,tracking_earthplate_xmin,tracking_earthplate_xmax,tracking_earthplate_xmax,tracking_xmax,tracking_xmax,tracking_xmin]
transverse_cross_section_dwg_yarray=[tracking_ymin,tracking_ymax,tracking_earthplate_ymin,tracking_earthplate_ymax,tracking_earthplate_ymax,tracking_earthplate_ymin,tracking_ymax,tracking_ymin,tracking_ymin]
#longitudinal cross section of monitor down monitor centre
longitudinal_cross_section_dwg_xarray=[tracking_zmin,tracking_zmin,tracking_earthplate_zmin,tracking_earthplate_zmin,tracking_earthplate_zmax,tracking_earthplate_zmax,tracking_zmax,tracking_zmax,tracking_zmin]
longitudinal_cross_section_dwg_yarray=[tracking_ymin,tracking_ymax,tracking_earthplate_ymin,tracking_earthplate_ymax,tracking_earthplate_ymax,tracking_earthplate_ymin,tracking_ymax,tracking_ymin,tracking_ymin]
#assign physical constants to variable names
c = sp.constants.c
#generate lists arrays to hold the particle data. These are converted to numpy arrays after they have been populated, because it is quicker to append items to lists than nnumpy arrays
particles=[] #an array to store all the particle objects in during tracking
destroyed_particles=[] #an array to store particle objects that are removed from the simulation, usually because they have moved outside of the simulation region
final_timesteps=[] #an array to view all the final timesteps calculated for particles - for use in debugging
##########################################################################################################################################################


def get_fast_amplifier_profile(filepath=None, sample_rate=(1/60e6)):
     '''
     Reads MCPM and SCPM Data taken using Fast Amplifiers in the EPB1 Ionisation profile monitor, and returns three items:
          1) Raw data from each channeltron
          2) Integrated signal from each channeltron
          3) The specified filepath of the channeltron data
     '''
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
     '''
     This function reads in raw data from an ISIS EPB Harp monitor, and converts the data file into three lists:
     \n     1) The horizontal profile data
     \n     2) The vertical profile data
     \n     3) A list containing the position of each wire, as offsets from the beam centre, calculated from the input wire spacing and number of wires variables.
     '''
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
     '''
     Opens a file dialogue box for the user to pick a file graphically, rather than having to hardcode filepaths.
     '''
     root=Tk()
     root.filename=filedialog.askopenfilename(title=message)
     filepath=root.filename
     root.withdraw()
     return(filepath)
     
def check_axis_format(axis_name):
     '''
     A function that is used within the import_CST_EField function.\n
     This function checks that the axis specified when the importing function is called for import of a CST 3D field file is specified in the correct format...i.e. with an "x" and "y" and a "z" variable
     '''
     if axis_name !='x' and axis_name !='y' and axis_name !='z':          
          print('\n*********************ERROR*******************************************************')
          print('*CST model axes specified incorrectly, please use only "x", "y" or "z" as inputs*')
          print('*********************ERROR*******************************************************\n')
          sys.exit()
     return()
     
def reformat_axes(field_data,model_axes):
     '''
     \nUsed within the import_CST_EField function. This function takes a 3D field data filefrom CST as one input, and an array specifying how the axes are oriented in the CST model. 
     \n
     \nThe model axes input is a 3-element array which specifies which cartesian axis represents each direction in the CST model, with the following format: "model_axes=[model_horizontal_axis,model_vertical_axis,model_longitudinal_axis]"
     \n
     \nThis function reformats the axes is requested, as different CST models will have different orientations. All imported files need to have their axes swithched around so that they are formatted as the IPM particle tracking code expects. This formatting is described below.
     \n
     \nThe tracking code assumes that:
     \n     1) The z axis represents the beam direction,
     \n     2) The x axis represents the transverse horizontal axis,
     \n     3) The y axis represents the transverse vertical direction.
     '''
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
     """
     Purpose
     -------
     This function is used to import a 3D Electric Field file from CST, which should initially be exported from CST as an ASCII text file.
     
     The filepath can be specified as an input to the function, or left as the default "None", in which case, a file dialogue will open for the user to graphically select the file.
     
     Returns
     -----
     The function returns a pandas dataframe containing the specified 3D field data.
     
     Other Inputs
     -----    
     The model axes in CST need to be specified as inputs to this function, to allow the code to adjust these if necessary to fit with the convention in particle accelerator components, that the z axis represents the beam direction, the y axis representing the transverse vertical dimension, and the x axis representing the transverse horizontal dimension.
     
     If the save_memory argument is specified as True, then the function will reduce the data types used where possible, to reduce the amount of memory occupied by the field, which can often be quite large for a 3D field file.
     For example, if a numerical column of fata are stoored as doubles, but are small enough, they will be changed to be stored as single floats instead.
     """
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
     '''Converts the units of the efield data output in V/m from CST into V/mm to match the position data calculated by the tracking code'''
     field_data['Ex']=field_data['Ex']/1000
     field_data['Ey']=field_data['Ey']/1000
     field_data['Ez']=field_data['Ez']/1000     
     print("EField data converted from V/m to V/mm")
     return()
     
def analyse_field_map(field_data,printout=False):
     '''
     Purpose
     ------
     This function takes an imported CST E Field array as an input and analyses the data, extracting the number of rows, the step size between data points, and the maximum and minimum position in each of the three co-ordinates.
     
     This data is used in various other sections of the particle tracking code when looking up the electric field at specified 3D co-ordinates, and making sure particles are generated and tracked inside the field map's volume.
     
     The analysed field parameters can be printed by setting the printout argument to True.
     
     Returns
     ------
     The function returns the following field map properties in the following order:

     - number of rows
     - step size (distance in mm between each point in the field map)
     - x axis size
     - y axis size
     - z axis size
     - minimum x value
     - maximum x value
     - minimum y value
     - maximum y value
     - minimum z value
     - maximum z value
     '''
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
     '''
     Draws an outline of the ISIS IPM Channeltrons and earth plate into a 3D graph, for adding context to particle trajectory/position plots.
     
     The plotname variable specifies which 3D plot should be drawn on.
     '''
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
     '''
     ISIS IPM Tracker: Particle Class
     -----
     The Particle class is used in the ISIS IPM particle tracking code to store information for a single particle.
     
     The class keeps a record of a particle's 3D position values, 3D velocity values, charge and mass, as well as the amount of time the particle has existed and what time it was created in the simulation.
     
     Species Argument
     -----
     Any of the following particle species can be generated, each with a different charge and mass value. This is done by passing one of the strings below to species argument:
          
          1) proton
          2) electron
          3) antiproton
          4) positron
          5) oxygen
          6) N2
          7) N+
          8) H20+
          9) OH+
          10) H2
          11) CO
          12) test
     
     N.B. the "test" particle is an ultra-light particle with 10% the mass of an electron and a charge of -1.6e-19, used to test relativistic tracking conditions and make sure particles to not accelerate past the speed of light
     
     Particle Methods
     -----
     The particle class comes with a set of built in functions:
          
          1) __init__
               Initialises the particle, sets its mass and charge valus (based on the species argument). Checks that the particle is within the current simulation volume, using the check_boundaries function, to make sure particles are not created in space where there is no field data or a long way away from the simulation region of interest. All created particles are stored in a "Particles" list for processing during the simulation.
          2) move
               Looks up the electric field values nearest to the particles position and calculated the particles movement and velocity changes over one timestep, using the Lorentz Force and kinematic equations of motion.
          3) perform_final_movement
               If the particle moves outside of the simulation region of interest during one movement step, the final movement is recalculated over a smaller timestep, so that the particle's final position lies exactly on the border of the simulation region. This is useful when, for example, selecting all particles which reach the IPM detectors, as the co-ordinates of the detector surface can be used as a filter when looking at each particle's final position.
          4) destroy
               Remove a particle from the Particles list, so that it is no longer considered in other calculations.
     Related Functions
     -----
     Several other functions related to particle movement used to be contained in the Particle class, but have since been removed and instead are seperate functions, compiled using the @jit decorator, to speed up mathematical operations during particle tracking. These are called from within the particle move and perform_final_movement functions, to perform the bulk of the calculations during tracking, and include:
     
          - calculate_new_position_and_velocity
          - calculate_relativistic_beta
          - calculate_gamma_from_beta
          - calculate_gamma
     
     '''
     def __init__(self,x,y,z,species,ID,vx=0,vy=0,vz=0, lifetime=0, creation_time=0):
          #global particles

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
          field_row_number=lookup_field_value(self.previous_x,self.previous_y,self.previous_z,combined_efield)
          #running_times['field boundary check finish']+=(time.time()-running_times['field boundary check start'])
          #calculate relativistic gamma and use this to increase the particle mass to account for relativistic effects
          relativistic_mass=self.mass*(calculate_gamma(self))
          
          #look up field values one each - this operation is time consuming and therefore its best not to look each value up twice
          #Ex=efield['Ex'].values[int(field_row_number)]
          #Ey=efield['Ey'].values[int(field_row_number)]
          #Ez=efield['Ez'].values[int(field_row_number)]
          
          Ex=combined_efield[int(field_row_number)][0]
          Ey=combined_efield[int(field_row_number)][1]
          Ez=combined_efield[int(field_row_number)][2]
          
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
          #Store tracking limits in seperate variables, to allow tracking region to be selected below
          max_xlimit, max_ylimit,max_zlimit=tracking_xmax,tracking_ymax,tracking_zmax
          min_xlimit, min_ylimit, min_zlimit=tracking_xmin,tracking_ymin,tracking_zmin
         
          #check if the particle is in the earth enclosure or not. If it is, change the tracking boundaries to the earth enclosure boundaries
          #if self.x < tracking_earthplate_xmax and self.x > tracking_earthplate_xmin and self.y < tracking_earthplate_ymax and self.y > tracking_earthplate_ymin and self.z < tracking_earthplate_zmax and self.z > tracking_earthplate_zmin:
          if self.previous_y > tracking_ymax: #if this is true, then the particle is being tracked through the earth enclosure region, not the main region
          #If the above statement is true, the particle is in the earth enclosure, so the tracking limits will be swapped for the final timestep calculation
               #print("Particle (ID="+str(particle.ID)+" is in the earthed enclosure for its final timestep")
               max_xlimit, max_ylimit,max_zlimit=tracking_earthplate_xmax,tracking_earthplate_ymax,tracking_earthplate_zmax
               min_xlimit, min_ylimit, min_zlimit=tracking_earthplate_xmin,tracking_earthplate_ymin,tracking_earthplate_zmin
          #identify which plane the particle went outside the tracking region on. Then calculate the time taken by solving the quadratic formula twice (once with a plus sign and once with a - sign). When solving the formula, replace the new position with the position of the tracking region boundary, so that the time taken to reach it is calculated accurately 
          if self.y > max_ylimit:
               Ey=combined_efield[int(field_row_number)][1]
               #calculate roots of the quadratic which calculates the final timestep (t_final = (-prev_vy +- sqrt(prev_vy**2-4(q*Ey/2*mass)(y-prev_y)))/(q*Ey/mass))
               a=(self.charge*Ey*1e6)/(2*relativistic_mass) #multiplied by 1e6 to give units of mm/s^2
               b=self.previous_vy
               c=self.previous_y-max_ylimit
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c])) #np.roots returns the two roots of the quadratic equation which is being solved to calculate the timestep required to reach the simulation boundary
          elif self.y < min_ylimit:
               Ey=combined_efield[int(field_row_number)][1]
               a=(self.charge*Ey*1e6)/(2*relativistic_mass)
               b=self.previous_vy
               c=self.previous_y-min_ylimit
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.x > max_xlimit:
               Ex=combined_efield[int(field_row_number)][0]
               a=(self.charge*Ex*1e6)/(2*relativistic_mass)
               b=self.previous_vx
               c=self.previous_x-max_xlimit
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.x < min_xlimit:
               Ex=combined_efield[int(field_row_number)][0]
               a=(self.charge*Ex*1e6)/(2*relativistic_mass)
               b=self.previous_vx
               c=self.previous_x-min_xlimit
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.z > max_zlimit:
               Ez=combined_efield[int(field_row_number)][2]
               a=(self.charge*Ez*1e6)/(2*relativistic_mass)
               b=self.previous_vz
               c=self.previous_z-max_zlimit
               final_timestep1=np.max(np.roots([a,b,c]))
               final_timestep2=np.min(np.roots([a,b,c]))
          elif self.z < min_zlimit:
               Ez=combined_efield[int(field_row_number)][2]
               a=(self.charge*Ez*1e6)/(2*relativistic_mass)
               b=self.previous_vz
               c=self.previous_z-min_zlimit
               
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
               print("\n########## DEBUGGING REPORT #################")
               print("Problematic Particle Co-ordinates: x="+str(self.x)+", y="+str(self.y)+", z="+str(self.z))
               print("Boundary Limits Used: max_xlimit="+str(max_xlimit)+", min_xlimit="+str(min_xlimit)+", max_ylimit="+str(max_ylimit)+", min_ylimit="+str(min_ylimit)+", max_zlimit="+str(max_zlimit)+", min_zlimit="+str(min_zlimit))
               print("Most recently calculated quadratic coefficients: a="+str(a)+", b="+str(b)+", c="+str(c))
               print("Most recently calculated roots = "+str(final_timestep1)+", "+str(final_timestep2))
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
     '''Calculates the new position and velocity of a particle in one axis, as a result of the lorenzx force applied on a charged particle by an electric field.
     
     The function requires inputs of the particles': starting position, starting velocity, timestep size, charge, mass, and electric field present at the particle's location.
     
     This function has been seperated from the Particle class and decorated with @jit to increase performance speed.'''
     x=previous_x+(previous_vx*timestep)+(((charge*timestep*timestep*efield_component)/(2*mass))*1e6)
     vx=previous_vx+(((charge*timestep*efield_component)/(mass))*1e6)
     return(x,vx)

@jit
def calculate_relativistic_beta(vx,vy,vz):
     '''takes a 3D velocity vector in mm/s, converts to m/s, calculates the magnitude of the velocity, then calculates relativistic beta from this.
     
     This function has been seperated from the Particle class and decorated with @jit to increase performance speed.'''
     #return(math.sqrt(math.pow((vx/1000),2)+math.pow((vy/1000),2)+math.pow((vz/1000),2))) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
     beta=(((((vx/1000)**2)+((vy/1000)**2)+((vz/1000)**2))**0.5)/c) #calculate the magnitude of the 3D previous velocity vector. Each quantity is divided by 1000 to convert from mm/s to m/s, to make calculating beta simpler
     return(beta)
     
@jit
def calculate_gamma_from_beta(beta):  
     '''takes an input relativistic beta value and returns the equivalent relativistic gamma value.
     
     This function has been separated and decorated with @jit to increase performance speed.'''
     relativistic_gamma=1/(math.sqrt(1-math.pow(beta,2)))
     return(relativistic_gamma)

def calculate_gamma(self):
     #running_times['calculate gamma start']=time.time()
     '''Returns the relativistic gamma of a particle based on its velocities in each cartesian plane.
     
     vx,vy and vz should be specified in mm/s.
     
     This function has been separated and decorated with @jit to increase performance speed.'''
     
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
     '''
     This function is called from within the lookup_field_value function, to do the mathmatical operation to check is a position (specified by x,y,z arguments) lies within the volume covered by the electric field data loaded into the program.
     
     This has been separated from the main lookup_field_value function  to allow experiments with decorators to speed the logic operation up, and also to allow for easier modification in the future for more complicated geometry checks. 
     
     Warning
     ----
     This function should generally not be called on its own by the user. If a field boundary check is required, the user should instead use the lookup_field_value function.
     '''
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
     '''
     Purpose
     -----
     This function takes a 3D co-ordinate (x,y,x) as input arguments, and checks whether the position lies within the user specified tracking boundaries.
     
     While this function is similar to the lookup_field_value function, it differs as the user-specified tracking region is often smaller than the electric field data file.
     
     User Specified Tracking Boundaries
     -----
     The user should specify tracking boundaries to the variables: tracking_xmax, tracking_xmin, tracking_ymax, tracking_ymin, tracking_zmax and tracking_zmin
     
     These traditionally are used to prevent particles moving through a solid part of the IPM monitor, such as the monitor wall or the IPM detectors (e.g. channeltron or MCP)
     Additionally any regions which do not result in any particles reaching the IPM detectors can be excluded to speed up the simulation and concentrate generated ions in areas that will produce profile data.
     '''
     if x > tracking_xmax or y > tracking_ymax or z > tracking_zmax or x < tracking_xmin or y < tracking_ymin or z < tracking_zmin: #If the particle is outside of any of the 6 boundaries, return False.
          ###FOR THE FUTURE...It would be more scaleable and easier to maintain if each seperate tracking region were stored in a single array (i.e. a 2D array with 6 columns, and one row for each regions boundaries). That way any boundary check can be writeen to cycle through all the regions in the array each time. This would allow for the creation of more complocated tracking regions made of multiple cuboid areas relatively easily.
          if earth_enclosure_slot_region==True: #If this is true, then do a second check to see if the particle has travelled through the earth enclosure slot and into the extra tracking region near the channeltrons
               #check if the particle is in the extra tracking region inside the earth enclosure slot (a feature of the ISIS IPM DESIGNS). If so, return true.
               if x > tracking_earthplate_xmax or y > tracking_earthplate_ymax or z > tracking_earthplate_zmax or x < tracking_earthplate_xmin or y < tracking_earthplate_ymin or z < tracking_earthplate_zmin:
                    return(False)
               else:
                    return(True)
          else:
               return (False)
     else:
          return(True)

def lookup_field_value(x,y,z,field_data): #check whether a specified location falls inside the imported field region or not
     '''
     This function takes a 3D (x,y,z) position co-ordinate as an input, and a field data file, then checks if the location falls within the volume covered in the field file or not.
     
     Functionally, this is most often used to identify positions outside of the field volume to prevent particles being generated there, as motion calculations would crash the code due to the inability to look up electric field values for these.
     '''
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
     '''
     Purpose
     -----
     Takes an input x,y,z co-ordinate and returns the row number in the CST electric field file which is closest to that location.The calculation of row number within the field file is based on the field properties most recently calculated using the analyse_field_map function.
     
     This function has been separated and decorated with @jit to increase performance speed.
     
     Important Pre-Requisite
     -----          
     **If the analyse_field_map function has not been called prior to this function being used, the results may be innacurate or may cause an error if there is no available field data at all.**
     
     While it would seem practical to build a call to analyse field map into this function, this is not possible, as it cannot be run every time the calculate_row_number function is run. analyse_field_map is memory and time intensive, and the calculate_row_number function is run millions of times during any tracking operation. The user must instead take care to make sure whenevr a field file is changed, that the analyse function is subsequently ran.
     '''
     #instead of rounding each number with np.round, np.int is used instead because it is much faster. This doesnt round, but instead just ignores everything after the decimal point. By adding 0.5 to each numebr before applying np.int, the result is the same as it would be if np.round had been used.
     element_num=((np.int(((x-min_x)/step_size)+0.5))*(1+(y_size/step_size))*(1+(z_size/step_size)))+((np.int(((y-min_y)/step_size)+0.5))*(1+(z_size/step_size)))+(np.int(((z-min_z)/step_size)+0.5))
     return(element_num)


def analyse_profile(profile_x,profile_y,min_peak_height=0):
     '''
     Returns the 95% width of the input profile and the location of its mean centre position.
     
     *The min_peak_height input should be ignored as it is not yet implemented in this function, but is a placeholder for future use.*
     
     The position and width values and dimensions are calculated from the input profile_x array (see example below)
     
     Example profile_x and profile_y Input
     -----     
     For a horizontal profile measurement with an ISIS IPM, profile_x should contain the positions of the centre of each channeltron, and 
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
     '''
     Purpose
     -----
     Returns the specified % width of an input profile, the half height width (if requested) and the mean centre location of the distribution.
     
     *The min_peak_height input should be ignored as it is not yet implemented in this function, but is a placeholder for future use.*
     
     The position and width values and dimensions are calculated from the input profile_x array (see example below)
     
     Example Profile Input
     -----
          
     For a horizontal profile measurement with an ISIS IPM, profile_x should contain the positions of the centre of each channeltron, and 
     profile_y should contain the measured beam profile, i.e. the particles measured by each channeltron.
     '''
     
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
     '''
     A moving average function which takes an input numpy array, x, and a number of samples to average over, w, and smooths the data accordingly.
     '''
     return np.convolve(x, np.ones(w), 'valid') / w

####################################################################################################################################################
#MAIN PROGRAM
print('**********************************************************************')
print("*               ISIS IPM PARTICLE TRACKING CODE             CCW v4.1 *")
print('**********************************************************************')

#STOP THE PROGRAM PLOTTING THE SET OF MANUALLY INPUT IBIC PLOTS AND HARP DATA AT AFTER TRACKING
ibicplots=False
select_harp=False

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

custom_time_file_path='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\Time Dependand Space Charge Tests\\EBP1_Approx_Time_Structure_0_602ns.txt'

if select_harp:
     harp_filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\IBIC 2019 Machine Physics Data 2_Sep\\IPM with Harp in\\EPM26A2.dat'#'C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\Harp Profiles 21July2019 Shift\\EPM26A2.dat'     
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
print("***** Loading and Analysing Electric Field Files *****")
print("Loading CST EField with Beam Field Included...")
efield_withbeam=import_CST_EField(filepath_withbeam,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
#analyse the field map to obtain properties needed for data lookup in the field
nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z=analyse_field_map(efield_withbeam,printout=True)

print("Loading CST EField with Beam Removed...")
efield_nobeam=import_CST_EField(filepath_nobeam,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z=analyse_field_map(efield_nobeam,printout=True)

#calculate field boundaries - doing this here speeds up the runtime of functions that use these values in if statements
field_max_x=max_x+step_size/2
field_min_x=min_x-step_size/2
field_max_y=max_y+step_size/2
field_min_y=min_y-step_size/2
field_max_z=max_z+step_size/2
field_min_z=min_z-step_size/2

#check that requested trackingboundaries are not larger than the imported Efield size
if tracking_xmax > field_max_x:
     tracking_xmax=field_max_x
if tracking_xmin < field_min_x: 
     tracking_xmin=field_min_x
if tracking_ymax > field_max_y:
     tracking_ymax=field_max_y
if tracking_ymin < field_min_y:
     tracking_ymin=field_min_y
if tracking_zmax > field_max_z:
     tracking_zmax=field_max_z
if tracking_zmin < field_min_z: 
     tracking_zmin=field_min_z
#repeat the check for the extra, inside earthplate region. If there is no field data then a region cannot be used for tracking!
if tracking_earthplate_xmax > field_max_x:
     tracking_earthplate_xmax=field_max_x
if tracking_earthplate_xmin < field_min_x: 
     tracking_earthplate_xmin=field_min_x
if tracking_earthplate_ymax > field_max_y:
     tracking_earthplate_ymax=field_max_y
if tracking_earthplate_ymin < field_min_y:
     tracking_earthplate_ymin=field_min_y
if tracking_earthplate_zmax > field_max_z:
     tracking_earthplate_zmax=field_max_z  
if tracking_earthplate_zmin < field_min_z: 
     tracking_earthplate_zmin=field_min_z

print("Imported Data Sample:")
print(efield_withbeam.head())

#Load the electric field data from the pandas dataframe into numpy arrays to increase performance
numpy_efield_withbeam=efield_withbeam[['Ex','Ey','Ez']].to_numpy()
numpy_efield_nobeam=efield_nobeam[['Ex','Ey','Ez']].to_numpy()
print("Electric field values loaded into seperate numpy array for fast access.")
#calculate beam-only field
numpy_beam_field=numpy_efield_withbeam-numpy_efield_nobeam
print("Beam space charge field calculated.")
#delete all the copies of the efield which will no longer be used to reduce memory usage
del efield_withbeam,efield_nobeam,numpy_efield_withbeam
#create a combined efield array, which will calculate the combined background efield and spacecharge field from the beam in each timestep
combined_efield=numpy_efield_nobeam
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#%%
#generate an ion distribution
particle_num=particle_num #the number of particles that should be generated inside the monitor
tracking_steps=tracking_steps
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
beam_xrad=(49.85/0.95) #/0.95 to scale the radius up from a 95% width to a 100% width.
beam_yrad=(59.66/0.95)
beam_xpos=-2.95
beam_ypos=18.36
beam_intensity_ppp=2.5e13
#beam_length_mm=500
print("*** GENERATED BEAM INFORMATION *** \n  --> Beam X Radius = "+str(beam_xrad)+" mm, Beam Y Radius = "+str(beam_yrad)+"mm\n  --> Beam X Offset = "+str(beam_xpos)+"mm, Beam Y Offset = "+str(beam_ypos)+"mm\n")

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

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Calculate/Load Beam Time Structure
#GENERATE BEAM TIME STRUCTURE
#test values of scaling the beam field

#REMINDER*********
#beam intensity ppp is input above with the other beam properties
#************
if custom_timestructure==False:
     #Generate structure that simply switches the beam on/off for a faster semi-time dependant simulation
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
     beam_field_scaling_factors=[]
     transition_count=0
     for t in range(0,tracking_steps+1):
          #check if a field transition time has been reached
          if t*1e9*input_timestep > transition_times[transition_count]:
               transition_count=transition_count+1
          #calculate whether the beam is present or not, then scale by 1 or 0 accordingly
          if transition_count%2==0: beam_field_scaling_factors.append(1)
          else: beam_field_scaling_factors.append(0)
     print("Simplified space charge field time structure calculated")
else:
     #LOAD CUSTOM TIME STRUCTURE, SCALE IT AND GENERATE AN ARRAY OF SCALING FACTORS FOR EACH TIMESTEP
     custom_timefile=np.loadtxt(custom_time_file_path, delimiter=',')
     temparray=np.zeros((tracking_steps)-custom_timefile.size)
     #SCALING THE INPUT TIME STRUCTURE ARRAY SO THAT IT REPRESENTS THE NUMBER OF PROTONS WITHIN THE MONITOR AT EACH 1ns TIMESTEP IN THE SIMULATION.
     #The time structure array is first normalised so that the area underneath it is equal to 1 (i.e. there is one protons worth of charge in total present throughout the full time structure)
     normalised_custom_timefile=custom_timefile/custom_timefile.sum()
     #Then this normalised array is multiplied by the beam intensity so that the area underneath the time structure is equal to the total number of protons in the beam, rather than 1 particle
     half_scaled_timestruct=normalised_custom_timefile*(beam_intensity_ppp)
     #finally the time structure is multiplied by 1.76ns/the input timestep. This is because each proton travelling at 0.84c takes 1.76ns to travel through the monitor. If each simulation timestep is longer/shorter than this, then the total number of particles in the monitor needs to be adjusted accordingly. e.g. if the timestep is 1ns, then the space charge field will be created by the full set of protons entering the IPM at this time, and also by some protons which entered the monitor in the previous timestep.
     scaled_timestruct=((half_scaled_timestruct*(1.76e-9/input_timestep))/1e12) #to account for the CST model now using 1e12 protons to generate the beam field, instead of a single proton, due to CST model accuracy limits
     
     #TEST ADDITION - increased beam field effect due to relativistic effects (field line density lorentz contraction)
     beam_beta=0.84     
     beam_gamma=1/(math.sqrt(1-beam_beta**2))
     scaled_timestruct=scaled_timestruct/beam_gamma #increase the strength of the beam field proportionally to the beam's gamme value (0.84 in ISIS EPB1) to account for the increased field density caused by relativistic effects (lorentz contraction of field lines).
     
     
     beam_field_scaling_factors=np.concatenate((scaled_timestruct,temparray),axis=None)
     #dummy oscillating time structure data
     #beam_field_scaling_factors=np.abs(np.sin(np.linspace(0,32*np.pi,tracking_steps+1)))

#print_timestructure     
plt.figure()
plt.plot(np.linspace(0,1e9*(input_timestep*tracking_steps),num=tracking_steps),beam_field_scaling_factors)
plt.title("Beam Space Charge Field Time Structure")
plt.xlabel("Time (ns)")
plt.ylabel("Space Charge Field Scaling Factor")
plt.show()

#%%

#GENERATE PARTICLE DISTRIBUTION------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Generating beam between: z_min="+str(beam_zmin)+"mm, and beam z_max="+str(beam_zmax)+"mm.")
print("\Beam Time Structure Properties:\n - No. Bunches = "+str(number_of_bunches)+"\n - Bunch length = "+str(bunch_length*1e9)+"ns\n - Inter-bunch spacing = "+str(bunch_spacing*1e9)+"ns \n Generating...\n")
particle_counter=1
particlers=[]
particlethetas=[]
for j in range(0,number_of_bunches):
     #create a particle. --> Calculate the x and y positions as random numbers generated inside gaussian (normal) distributions for each axis. The distribution mean (loc) values are set to the beam offset positions, and the standard deviations (scale) are set to half of the beam width in CST (so that the 95% widths correspond to 2 signma, which is correct for a gaussian). Size is the number of points that the function generates
     
     #Calculate Particle Generation Times For Each Ion
     # to calculate bunch start time, in seconds in the loop below, use: bunch_start_time=(j*(bunch_spacing+bunch_length))*1e9
     for i in range(0,int(np.round(particle_num/number_of_bunches))):
          if custom_timestructure==False:          
               #If there is no custom time structure loaded, generate the particles uniformly within each bunch using the semi-time dependant approximation
               particle_creation_time=((j*(bunch_spacing+bunch_length)*1e9)+np.random.randint(low=0,high=(bunch_length*1e9)+1))/1e9 #pick a random creation time in seconds (but in integer values of ns) for each particle that fits inside the bunch length
          else:
               #If there is a custome time structure, particle generation rates should increase and decrease proportionally to the space charge field...i.e. when the beam is more intense within the monitor, the rate of ion generation should increase
               normalised_beam_timestructure=beam_field_scaling_factors/beam_field_scaling_factors.sum() #create a list containing the beam time structure, normalised. Prbability of particle generation will be weighted by this normalised list.
               weighted_times=(np.random.choice(a=tracking_steps,size=particle_num,p=normalised_beam_timestructure))/1e9
               particle_creation_time=weighted_times[particle_counter-1]
          #Caculate generation positions for each particle
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

#%%
#TRACK PARTICLES THROUGH ELECTRIC FIELD

simulation_time=0
#Only calculate field transitions if you are using semi-time dependant tracking, which switches between two separate field maps. If a custom timstructure is entered, you do not need to swap electric field files each time a bunch enters or leaves the IPM
if custom_timestructure==False:
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

print("Tracking "+str(particle_num)+" particles through "+str(tracking_steps)+" timesteps.\nPlease wait...")
#PARTICLE TRACKING------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("0 ns tracked.",end=""  ) #end="" stops the output moving to a new line after printing this message (used to create a tidy progress readout in the output console)
transition_count=0
for i in range (0,tracking_steps):
    count=0
    #Calculate the electric field at this specific simulation time
    combined_efield=numpy_efield_nobeam+(numpy_beam_field*beam_field_scaling_factors[i])
    for particle in reversed(particles): #iterate through the particles list in reversed order, so that if any particle objects are deleted from the particles list, the iteration onto the next element is not affected by the change in the list's overall size (when iterating from start to finish, if a particle is deleted, the element numbers for the remaining particles are reduced by 1, so they iterator misses out the next particle in the list)
        if particle.creation_time <= simulation_time:
             particle.move(timestep=input_timestep, plot=True) #move all particles unless they have not yet been generated
             count=count+1
    simulation_time=simulation_time+input_timestep
    #print("Simulation Time = "+str(simulation_time))
    print("\r"+str(i+1)+" ns tracked. There are "+str(count)+" particles being tracked.", end="")#re-write to the console with an updated progress message. "\r" prints from the start of the current line on the console, to overwrite the previous time printout

#LEGACY CODE - SEMI TIME DEPENDANCE FIELD SWITCHING
#    if (np.round(simulation_time*1e9,decimals=6)==transition_times[transition_count]):
#         transition_count=transition_count+1
#         del efield, numpy_efield #delete the existing electric fields stored in memory to reduce overall memory usage
#         if transition_count%2==1: #if the transition count is odd, a bunch is leaving the monitor (simulation always begins with a bunch in the monitor)
#              filepath=filepath_withbeam
#              print("\nThe particle bunch is leaving the monitor - changing E_Field...\n")
#         else: 
#              filepath=filepath_nobeam
#              print("\nA particle bunch is entering the monitor - changing E_Field...\n")
#         efield=import_CST_EField(filepath,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
#         numpy_efield=efield[['Ex','Ey','Ez']].to_numpy()         
print("\nThere are "+str(len(particles))+" particles remaining in the simulation region.")

#FULL TIME DEPENDANCE PARTICLE TRACKING------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#%%
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ANALYSE RESULTS####################################################################################################################

#generate an array of particles which reached the detector
detected_particles=np.asarray([[particle.final_x,particle.final_y,particle.final_z,particle.initial_x,particle.initial_y,particle.initial_z,particle.creation_time,particle.lifetime] for particle in destroyed_particles if particle.final_y==detector_y and particle.final_x>=detector_xmin and particle.final_x<=detector_xmax and particle.final_z>=detector_zmin and particle.final_z<=detector_zmax])
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
          if y_positions[-1]==detector_y and x_positions[-1]>=detector_xmin and x_positions[-1]<=detector_xmax and z_positions[-1]>=detector_zmin and z_positions[-1]<=detector_zmax: 
               detected=True
          else: detected=False
          particle_tracks.append([x_positions,y_positions,z_positions,detected])
     else: untracked_particles=untracked_particles+1

particle_num=particle_num-untracked_particles

#%%
#PLOT DATA##########################################################################################################################
print("Generating Plots...")
#Print samples of the particle trajectories in both transverse and longitudinal planes
plt.subplot(1,2,1)
for i in range(particle_num-200, particle_num):
    plt.plot(particle_tracks[i][2],particle_tracks[i][1],linewidth=0.5)
plt.plot(longitudinal_cross_section_dwg_xarray,longitudinal_cross_section_dwg_yarray, '--', color='black', linewidth=0.5)
plt.title("Sample Particle Trajectories Viewed in the Z-Y Plane")
plt.xlabel("Z (mm)")
plt.ylabel("Y (mm)")
plt.subplot(1,2,2)
for i in range(particle_num-200, particle_num):
    plt.plot(particle_tracks[i][0],particle_tracks[i][1],linewidth=0.5)
plt.plot(transverse_cross_section_dwg_xarray,transverse_cross_section_dwg_yarray, '--', color='black', linewidth=0.5)
plt.title("Sample Particle Trajectories Viewed in the X-Y Plane")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.show()

#If no particles reached the detectors, display a warning and quit the program
if detected_particles.size==0:
     print('*****************************************************************************************')
     print('* No Particles were detected by the IPM. Examine the sample plot, check the simulation  *')
     print('*    time length and the detector and tracking boundary co-ordinates in this program.   *')
     print('*****************************************************************************************')
     sys.exit()     

#PLOT SIMULATED PROFILE WITH STATS
ideal_profile_data=np.histogram(initial_positions[:,0],bins=40, range=(-120,120))
ideal_profile=normalise_profile_area(ideal_profile_data[0])
profile_data=np.histogram(detected_particles[:,0],bins=40, range=(-120,120))
detected_profile=normalise_profile_area(profile_data[0])
#generated array of interpolated chaneltron positions for a more accurate x-axis
channeltron_positions=profile_data[1]-3
channeltron_positions=np.delete(channeltron_positions,0)
interpolated_channeltron_positions=np.linspace(-120,120,1000)
#Generate an interpolated simulated profile for higher accuracy when calculating stats
interpolated_detected_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,detected_profile)   
interpolated_ideal_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,ideal_profile)
#calculate 95% widths and peak centres
width_percent=95
detected_95w,detected_centre,detected_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_detected_profile,percentage=width_percent,hhw=True)
ideal_95w,ideal_centre=calculate_percent_width(interpolated_channeltron_positions,interpolated_ideal_profile,percentage=width_percent)
#Generate Plot
plt.figure()
plt.plot(interpolated_channeltron_positions,interpolated_detected_profile)
plt.title("Simulated IPM Profile Measurement And Analysis")
plt.xlabel("Position (mm)")
plt.ylabel("Profile Amplitude (normalised)")
plt.figtext(0.05,0.13,"95% Width = "+str("{:10.2f}".format(detected_95w))+" mm")
plt.figtext(0.05,0.10,"Half Height Width = "+str("{:10.2f}".format(detected_hhw))+" mm")
plt.figtext(0.05,0.07,"Profile Centre = "+str("{:10.2f}".format(detected_centre))+" mm")
plt.figtext(0.05,0.04,'Electric Field Filepath 1: '+filepath_withbeam)
plt.figtext(0.05,0.01,'Electric Field Filepath 2: '+filepath_nobeam)
plt.subplots_adjust(top=0.88,bottom=0.22,left=0.125,right=0.9,hspace=0.2,wspace=0.2)
plt.show()


     
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
plt.plot(transverse_cross_section_dwg_xarray,transverse_cross_section_dwg_yarray, '--', color='black', linewidth=0.5)       
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
plt.plot(longitudinal_cross_section_dwg_xarray,longitudinal_cross_section_dwg_yarray, '--', color='black', linewidth=0.5)       
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

plt.plot(transverse_cross_section_dwg_xarray,transverse_cross_section_dwg_yarray, '--', color='black', linewidth=0.5)
plt.title('Detected Particle Trajectories:\n Horizontal Plane')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis([-150,150,-150,detector_y+10]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

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

plt.plot(longitudinal_cross_section_dwg_xarray,longitudinal_cross_section_dwg_yarray, '--', color='black', linewidth=0.5)          
plt.title('Detected Particle Trajectories:\n Longitudinal Plane')
plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.axis([min_z,max_z,-150,detector_y+10]) #the first 2 elements set the lower and upper x axis limits, the next 2 elements set the y axis range

#####
#plot time distribution of detected particles
bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))

axes1=plt.subplot(2,4,5)
axes2=axes1.twinx()
axes1.hist(particle_time_structure[:,0]*1e9, bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)),color='orange',alpha=0.6)#,edgecolor='black')
#ax.title("Time Structure of Particles Created\n Resolution = "+str(time_resolution*1e9)+"ns")
axes1.set_xlabel('Time Elapsed(ns)')
axes1.set_ylabel('Ions Generated By Beam (Orange)')   

bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))
#plt.subplot(2,4,6)
#axes1.cla()
plt.title("Time Structure of Generated (Orange)\n and Detected (Blue) Particles \n(Resolution = "+str(time_resolution*1e9)+"ns)")
axes2.set_ylabel('Ions Detected By Channeltrons (Blue)')     
if np.size(detected_particles) > 0:
     axes2.hist(((detected_particles[:,6]+detected_particles[:,7])*1e9), bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)),alpha=0.6)#, edgecolor='black')
#ax.ylim(0,250)
plt.show()


##################
#plotting the variation of space charge field over time     
timeaxis=np.linspace(0,1e9*(input_timestep*tracking_steps),num=tracking_steps)
#integrate the bunch shape seen by the sum monitor (used to calculate scaling factor array). The integral of this over time is proportional to the number of particles in the beam
normalised_beam_field_scaling_factors=beam_field_scaling_factors/np.sum(beam_field_scaling_factors)
integrated_time_structure=np.trapz(y=normalised_beam_field_scaling_factors,x=timeaxis*input_timestep) #the x axis is multiplied by the input timestep so that the charge contribution from each timestep integral is correctly scaled
#find the constant of proportionality that links the scaling factors to the beam intensity
proportionality_k1=integrated_time_structure/beam_intensity_ppp
#at any point in time, the time structure at that point (which covers 1 ns), divided by the proportionality constant, gives the number of particles entering the monitor in that timestep (1ns).
#each particle stays in the monitor for 1.76ns, due to the beam velocity of 0.84c.
#therefore total particles in monitor at a time t1 is equal to the time structure at t1 divided by the proportionality constant, all multiplied by (1.76ns/timestep). If, for example, the timestep was 3ns, this second term would reduce the overall charge as the protons would only be present for some of this timestep
particles_in_monitor=(beam_field_scaling_factors/proportionality_k1)*(1.76e-9/input_timestep)
#plot beam field time structure     
plt.subplot(2,4,6)
plt.plot(timeaxis,(1e12*beam_field_scaling_factors))
plt.title("Beam Space Charge Field Time Structure\n  ")
plt.xlabel("Time (ns)")
plt.ylabel("Space Charge Field Scaling Factor")

plt.subplot(2,4,7)
plt.hist(initial_positions[:,0],bins=40, range=(-120,120), edgecolor='black', color='orange')
plt.title("Initial Beam Profile\n (95% Width = "+str("{:10.2f}".format(ideal_95w))+" mm)")

plt.subplot(2,4,8)
plt.title("Beam Profile At Channeltrons\n (95% Width = "+str("{:10.2f}".format(detected_95w))+" mm)")
if np.size(detected_particles) > 0:
     plt.hist(detected_particles[:,0],bins=40, range=(-120,120), edgecolor='black')

#set the layout up so there is space for information text below the plots
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.figtext(0.05,0.11,'Simulation Details',weight='bold')
plt.figtext(0.05,0.09,'Electric Field Information: Imported Step Size = '+str(step_size)+' mm, X Max = '+str(max_x)+' mm, X Min = '+str(min_x)+", Y Max = "+str(max_y)+' mm, Y Min = '+str(min_y)+', Z Max = '+str(max_z)+' mm, Z Min = '+str(min_z)+'mm')
plt.figtext(0.05,0.07,'Beam Intensity: '+str(beam_intensity_ppp/1e12)+'e12 ppp')
plt.figtext(0.05,0.05,'Type of particle tracked = H20+, OH+, H+, O+, H2+, N+, N2+')
plt.figtext(0.05,0.03,'Electric Field Filepath 1: '+filepath_withbeam)
plt.figtext(0.05,0.01,'Electric Field Filepath 2: '+filepath_nobeam)

if ibicplots==False: 
     print('*****************************************************************************************')
     print('*Tracking Simulation Complete - Please run specific plotting cells manually as required.*')
     print('*****************************************************************************************')
     sys.exit()

#%%
#PLOT SIMULATED PROFILE WITH STATS
profile_data=np.histogram(detected_particles[:,0],bins=40, range=(-120,120))
detected_profile=normalise_profile_area(profile_data[0])
#generated array of interpolated chaneltron positions for a more accurate x-axis
channeltron_positions=profile_data[1]-3
channeltron_positions=np.delete(channeltron_positions,0)
interpolated_channeltron_positions=np.linspace(-120,120,1000)
#Generate an interpolated simulated profile for higher accuracy when calculating stats
interpolated_detected_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,detected_profile)   
#calculate 95% widths and peak centres
width_percent=95
detected_95w,detected_centre,detected_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_detected_profile,percentage=width_percent,hhw=True)
#Generate Plot
plt.figure()
plt.plot(interpolated_channeltron_positions,interpolated_detected_profile)
plt.title("Simulated IPM Profile Measurement And Analysis")
plt.xlabel("Position (mm)")
plt.ylabel("Profile Amplitude (normalised)")
plt.figtext(0.05,0.13,"95% Width = "+str("{:10.2f}".format(detected_95w))+" mm")
plt.figtext(0.05,0.10,"Half Height Width = "+str("{:10.2f}".format(detected_hhw))+" mm")
plt.figtext(0.05,0.07,"Profile Centre = "+str("{:10.2f}".format(detected_centre))+" mm")
plt.figtext(0.05,0.04,'Electric Field Filepath 1: '+filepath_withbeam)
plt.figtext(0.05,0.01,'Electric Field Filepath 2: '+filepath_nobeam)
plt.subplots_adjust(top=0.88,bottom=0.22,left=0.125,right=0.9,hspace=0.2,wspace=0.2)
plt.show()


#%%
fig,axes1=plt.subplots()
axes2=axes1.twinx()
axes1.hist(particle_time_structure[:,0]*1e9, bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)),color='orange',alpha=0.6)#,edgecolor='black')
#ax.title("Time Structure of Particles Created\n Resolution = "+str(time_resolution*1e9)+"ns")
axes1.set_xlabel('Time Elapsed(ns)')
axes1.set_ylabel('Ions Generated By Beam (Orange)')   

bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))
#plt.subplot(2,4,6)
#axes1.cla()
plt.title("Time Structure of Generated (Orange)\n and Detected (Blue) Particles \n(Resolution = "+str(time_resolution*1e9)+"ns)")
axes2.set_ylabel('Ions Detected By Channeltrons (Blue)')     
if np.size(detected_particles) > 0:
     axes2.hist(((detected_particles[:,6]+detected_particles[:,7])*1e9), bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)),alpha=0.6)#, edgecolor='black')
#ax.ylim(0,250)
plt.show()

#%%
#plot time distribution of detected particles
bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))

plt.subplot(2,4,5)
plt.hist(particle_time_structure[:,0]*1e9, bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)),color='orange',alpha=0.6)#,edgecolor='black')
plt.title("Time Structure of Particles Created\n Resolution = "+str(time_resolution*1e9)+"ns")
plt.xlabel('Time Elapsed(ns)')
plt.ylabel('Particles Detected')   

bin_num=int(np.ceil((tracking_steps*input_timestep)/time_resolution))
#plt.subplot(2,4,6)
plt.title("Time Structure of Generated (Orange)\n and Detected (Blue) Particles \n(Resolution = "+str(time_resolution*1e9)+"ns)")
plt.xlabel('Time Elapsed(ns)')
plt.ylabel('Particles Detected')     
if np.size(detected_particles) > 0:
     plt.hist(((detected_particles[:,6]+detected_particles[:,7])*1e9), bins=bin_num, range=(0,(tracking_steps*input_timestep*1e9)),alpha=0.9)#, edgecolor='black')
plt.ylim(0,250)
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


#ISIS IPM SPECIFIC PLOTTING TO ANALYSE THE TIME STRUCTURE
################### PROCESSING AND PLOTTING MULTIPLE PROFILES ################################################################
#do the same for the MCPM data daken during machine physics measurements

#Get a profile measured using the fast amplifiers on the ISIS EPB MCPM
ctron_time_data,MCPM_integrated_profile,MCPM_filepath=get_fast_amplifier_profile() #integrated data: index 0 contains single channel data, which should be ignored. Indexes 1-40 contain MCPM data

reverse_ctrons=False #reverse order of channeltrons as the fast amplifier code may be outputting chanel 0 as channel 39, channel 1 as channel 38 etc...
if reverse_ctrons:
     MCPM_integrated_profile.reverse() 
     
print("MCPM Profile Data: "+str(MCPM_integrated_profile))
backup_MCPM_integrated_profile=MCPM_integrated_profile
#MCPM_integrated_profile=[0.023,0.051,-0.020,0.120,0.003,0.036,0.048,0.309,0.246,0.749,2.062,1.284,6.159,7.387,9.593,10.068,12.813,10.268,12.927,11.854,12.905,11.220,13.474,15.672,9.889,9.427,5.919,6.155,2.333,1.390,0.774,0.035,0.073,0.072,-0.353,-0.039,0.035,0.013,0.596,-0.004]
MCPM_profile=normalise_profile_area(MCPM_integrated_profile)
interpolated_MCPM_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,MCPM_profile)
MCPM_95w,MCPM_centre,MCPM_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_MCPM_profile,percentage=width_percent,hhw=True)
#IBIC2019 Measurements Manual Input of each time peak
#first small peak in measurement, with bias set high to make it visible
#INTEGRATION WINDOW = 4.4-4.7us
peak1_profile=np.zeros(40)
peak1_profile=np.asarray(peak1_profile)
peak1_profile=normalise_profile_area(peak1_profile)
interpolated_peak1_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak1_profile)
peak1_95w,peak1_centre,peak1_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak1_profile,percentage=width_percent,hhw=True)

#INTEGRATION WINDOW = 4.7-5us
peak2_profile=np.zeros(40)
peak2_profile=np.asarray(peak2_profile)
peak2_profile=normalise_profile_area(peak2_profile)
interpolated_peak2_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak2_profile)
peak2_95w,peak2_centre,peak2_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak2_profile,percentage=width_percent, hhw=True)

#INTEGRATION WINDOW = us
peak3_profile=np.zeros(40)
peak3_profile=np.asarray(peak3_profile)
peak3_profile=normalise_profile_area(peak3_profile)
interpolated_peak3_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak3_profile)
peak3_95w,peak3_centre,peak3_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak3_profile,percentage=width_percent,hhw=True)

#INTEGRATION WINDOW = 5.3-5.7us
peak4_profile=np.zeros(40)
peak4_profile=np.asarray(peak4_profile)
peak4_profile=normalise_profile_area(peak4_profile)
interpolated_peak4_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak4_profile)
peak4_95w,peak4_centre,peak4_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak4_profile,percentage=width_percent, hhw=True)

#INTEGRATION WINDOW = 5.7-6us
peak5_profile=np.zeros(40)
peak5_profile=np.asarray(peak5_profile)
peak5_profile=normalise_profile_area(peak5_profile)
interpolated_peak5_profile=np.interp(interpolated_channeltron_positions,channeltron_positions,peak5_profile)
peak5_95w,peak5_centre,peak5_hhw=calculate_percent_width(interpolated_channeltron_positions,interpolated_peak5_profile,percentage=width_percent, hhw=True)

##################################################################################################################

#Plot comparing the simulated profile, harp profile, measured profile, and profile of generated beam distributiuon in the IPM code
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

#plot ocomparing the simulated IPM profile and the measured IPM profile
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

#Plot of each individual peak profile
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

print('***************************  END  ************************************')

#######################################################################################################################################################################
#%%
#Field Calculation Test Plots
#plt.figure()
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[ten_billion_proton_field.mean(),twenty_billion_proton_field.mean(),thirty_billion_proton_field.mean(),fifty_billion_proton_field.mean(),onehundred_billion_proton_field.mean(),fivehundred_billion_proton_field.mean(),onetrillion_proton_field.mean()])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[ten_billion_proton_field.mean(),2*ten_billion_proton_field.mean(),3*ten_billion_proton_field.mean(),5*ten_billion_proton_field.mean(),10*ten_billion_proton_field.mean(),50*ten_billion_proton_field.mean(),100*ten_billion_proton_field.mean()])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[twenty_billion_proton_field.mean()/2,twenty_billion_proton_field.mean(),twenty_billion_proton_field.mean()*1.5,twenty_billion_proton_field.mean()*2.5,twenty_billion_proton_field.mean()*5,twenty_billion_proton_field.mean()*25,twenty_billion_proton_field.mean()*50])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[thirty_billion_proton_field.mean()/3,thirty_billion_proton_field.mean()*2/3,thirty_billion_proton_field.mean(),thirty_billion_proton_field.mean()*5/3,thirty_billion_proton_field.mean()*10/3,thirty_billion_proton_field.mean()*50/3,thirty_billion_proton_field.mean()*100/3])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[fifty_billion_proton_field.mean()/5,fifty_billion_proton_field.mean()*2/5,fifty_billion_proton_field.mean()*3/5,fifty_billion_proton_field.mean(), fifty_billion_proton_field.mean()*2,fifty_billion_proton_field.mean()*10,fifty_billion_proton_field.mean()*20])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[onehundred_billion_proton_field.mean()/10,onehundred_billion_proton_field.mean()*2/10,onehundred_billion_proton_field.mean()*3/10,onehundred_billion_proton_field.mean()*5/10,onehundred_billion_proton_field.mean(),onehundred_billion_proton_field.mean()*5,onehundred_billion_proton_field.mean()*10])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[fivehundred_billion_proton_field.mean()/50,fivehundred_billion_proton_field.mean()*2/50,fivehundred_billion_proton_field.mean()*3/50,fivehundred_billion_proton_field.mean()*5/50,fivehundred_billion_proton_field.mean()*1/5,fivehundred_billion_proton_field.mean(),fivehundred_billion_proton_field.mean()*2])
#plt.plot([1e10,2e10,3e10,5e10,10e10,50e10,1e12],[onetrillion_proton_field.mean()/100,onetrillion_proton_field.mean()*2/100,onetrillion_proton_field.mean()*3/100,onetrillion_proton_field.mean()*5/100,onetrillion_proton_field.mean()*1/10,onetrillion_proton_field.mean()*1/2,onetrillion_proton_field.mean()])
#plt.legend(['CST Average Fields','Fields Calculated from 1e10 Proton Field','Fields Calculated from 2e10 Proton Field','Fields Calculated from 3e10 Proton Field','Fields Calculated from 5e10 Proton Field','Fields Calculated from 10e10 Proton Field','Fields Calculated from 50e10 Proton Field','Fields Calculated from 1e12 Proton Field'])
#plt.title("Comparison of Beam EFields Calculated with Different Levels of Charge in the Initial CST Model")
#plt.xtitle("Total Charge In IPM")
#plt.ytitle("CST 3D Efield File Average Value")
#plt.show()