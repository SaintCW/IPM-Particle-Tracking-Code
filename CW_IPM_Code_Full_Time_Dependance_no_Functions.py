# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:34:35 2019

@author: vwa13369
"""
#import my ISIS IPM tracking library
import cw_ipm_tracking_library as ipm
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
##########################################################################################################################################################



#MAIN PROGRAM
print('**********************************************************************')
print("*               ISIS IPM PARTICLE TRACKING CODE             CCW v3.0 *")
print('**********************************************************************')

#STOP THE PROGRAM PLOTTING THE SET OF MANUALLY INPUT IBIC PLOTS AND HARP DATA AT AFTER TRACKING
ibicplots=False
select_harp=False

#Import and Analyse the Electric Field From CST
plt.close('all') #close anyt plots still open from previous runs of the code
print("Please choose CST electric field files...\n The first file chosen should contain the IPM and particle beam, the second file should contain the IPM with no beam.\nIf only one file is needed, select the same file twice.\n**IMPORTANT - Both fields must be the same size, cover the same co-ordinates and have the same resolution.**\n-------------------------------------\n")
filepath_withbeam=ipm.open_file_dialogue(message="Select CST EField File 1 - With Beam Inside the IPM") #let the user choose a filepath graphically
filepath_nobeam=ipm.open_file_dialogue(message="Select CST EField File 2 - With No Beam In The Model") #ask
if not filepath_withbeam or not filepath_nobeam: #check that both filepaths were picked (an empty string will trigger this if statement)
     print("**************ERROR******************")
     print("*        Filepath(s) Missing        *")
     print("*Please specify 2 valid EField files*")
     print("*************************************")
     sys.exit()

if select_harp:
     harp_filepath='C:\\Users\\vwa13369\\Desktop\\IPM_Particle_Tracking_Code\\Harp Profiles 21July2019 Shift\\EPM26A2.dat'     
     if harp_filepath==None: 
          select_harp=False
          print('No harp monitor data loaded')
     else:
          hor_harp,ver_harp,harp_wire_positions=ipm.read_harp_data(harp_filepath)
          #interpolate the data to accurately calculate widths and centre positions
          interpolated_harp_wire_positions=np.linspace(-69,69,1000)
          interpolated_hor_harp_profile=np.interp(interpolated_harp_wire_positions,harp_wire_positions,hor_harp)
          interpolated_ver_harp_profile=np.interp(interpolated_harp_wire_positions,harp_wire_positions,ver_harp)
          #analyse the profiles for 95% widths and centre positions in each plane
          hor_harp_width,hor_harp_centre=ipm.analyse_profile(interpolated_harp_wire_positions,interpolated_hor_harp_profile)
          ver_harp_width,ver_harp_centre=ipm.analyse_profile(interpolated_harp_wire_positions,interpolated_ver_harp_profile)
          print('Harp profile loaded from: '+harp_filepath)
          print('Beam X Centre = '+str(hor_harp_centre)+'mm, Beam Y Centre = '+str(ver_harp_centre)+'mm, \nHorizontal 95% Width = '+str(hor_harp_width)+'mm, Vertical 95% Width = '+str(ver_harp_width)+'mm\n\n')
     

#filepath='C:\\Users\\vwa13369\\Desktop\\AccPhys 2016\\2019_Field_Maps\\-15kV_-1400VBias_2_27e13ppp_radx_54_9_rady_41_5_xoff_0_86_yoff_-2_9_CFOFF.txt'
running_times={'start time':time.time(),'field boundary check start':0,'field boundary check finish':0,'calculate gamma start':0,'calculate gamma finished':0,'particle move start':0,'particle move finished':0} #Log the time to calculate the execution time of the code

#IMPORT EFIELD, ANALYSE AND CONVERT TO NUMPY----------------------------------------------------------------------------------------------------------------------
efield=ipm.import_CST_EField(filepath_withbeam,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
#analyse the field map to obtain properties needed for data lookup in the field
nrows,step_size,x_size,y_size,z_size,min_x,max_x,min_y,max_y,min_z,max_z=ipm.analyse_field_map(efield,printout=True)
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

running_times['start tracking']=time.time()
print("Tracking "+str(particle_num)+" particles through "+str(tracking_steps)+" timesteps.\nPlease wait...")
#PARTICLE TRACKING------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
         efield=ipm.import_CST_EField(filepath,nrows=None, model_horizontal_axis='z',model_vertical_axis='y',model_longitudinal_axis='x')
         numpy_efield=efield[['Ex','Ey','Ez']].to_numpy()         
print("\nThere are "+str(len(particles))+" particles remaining in the simulation region.")

#FULL TIME DEPENDANCE PARTICLE TRACKING------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
     print('*****************************************************************************************')
     print('*Tracking Simulation Complete - Please run specific plotting cells manually as required.*')
     print('*****************************************************************************************')
     sys.exit()

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

