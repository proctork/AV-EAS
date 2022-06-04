# -*- coding: utf-8 -*-
"""
2022-06-03

This script calculates the yield and economics accounting for
both the crop and PV returns. It relies upon the hourly rad
csv files from the calc_spatial_hourly_rad_for_DSSAT.py script
and the economic values from the Economics.py script

To run this file for rainfed conditions change the seasonal file 
KPAR2020.SNX. Change line 79 "IRRIG" value from A to N 
Indicating no irrigation

@author: proctork@oregonstate.edu
"""
import subprocess 
import numpy as np
import pandas as pd
import os
import datetime
import shutil
import glob
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib import rcParams

def reg_pickle_open(filename):
    with open(filename, 'rb') as f:
        pFile=pickle.load(f)
    
    return pFile

def reg_pickle_dump (save_dir,filename,data):
  # pickle files
    with open(save_dir+filename+'.p','wb') as f:
        pickle.dump(data,f)
        
def update_srad(updatedSrad_file):
    """
    This function updates the Solar Radiation (MJ/m^2/day) SRAD
    parameter in DSSAT. This is accomplished by reading in the new
    SRAD values and then writing them into the WTH file used by
    DSSAT

    Parameters
    ----------
    updatedSrad_file : str
        name of file which holds new SRAD data, series of SRAD
        values for each day of year

    Returns
    -------
    None.

    """
    wFile='C:\\DSSAT47\\Weather\\ARAO2001.WTH'
    wRead=open(wFile,'r+')
    wWrite=wRead.readlines()
    wRead.close()
    
    newSrad=pd.read_csv(updatedSrad_file,header=None)
    newSrad=np.round(newSrad,1).astype(str)
    
    for i in range(0,365):
        string=wWrite[i+5]
        new_val=newSrad[0][i]
        start_idx=(10-len(new_val))
        
        if start_idx==7:
            new_string=string[0:start_idx]+' '+new_val+string[11:]
        else:
            new_string=string[0:start_idx]+' '+new_val+string[11:]
        #string[]=new_val
        wWrite[i+5]=new_string
    
    newF=open(wFile,'w+')
    newF.writelines(wWrite)
    newF.close()
#%%
# This script runs the DSSAT model
# many of the variables are static names which do not change 
# for example the batch file is a text file which holds information 
# about the management practices and crop of choice 
# the file itself can be altered but the file name will remain the same 

dssat_dir='C:\\DSSAT47' # dir with all dssat file 
dssatexe=dssat_dir+'\\build\\bin\\dscsm047.exe' # name of fortran script that runs dssat model
run_type='N' # seasonal 
batch_file='DSSBatch.v47' # name of batch file 
input_string=dssatexe+' '+run_type+' '+batch_file
output_dir=dssat_dir+'\\Outputs' # folder where outputs are stored 
command_to_setpath='set PATH=%PATH;'+dssat_dir


#%%
# Similair to the batch file, the DSSAT fortran code is set to 
# read rad data from a file named "rad_values.txt" in a specific folder
# the next few lines identify the file of choice in a different folder
# copy it to the folder DSSAT expects, adn renames the file 
# the rad file must be named "rad_values.txt" for the code to work 
        # update name to correct one 

input_rad_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\DSSAT_AVrad_Spatial'
input_rad_destination=r'C:\DSSAT47\hourly_rad_data'
srad_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\DSSAT_AVrad_spatial_day'


#%%
# I should add something to explicitly make sure they are lined up
# right now its based on sorting and naming conventions
fi=glob.glob(input_rad_dir+'\\*')
srad_fi=glob.glob(srad_dir+'\\*')
all_fi=pd.DataFrame(fi,columns=['hourly'])
all_fi['Srad']=srad_fi
#%%
def run_dssat_single(fname,srad_name):
    pheight=fname.split('_')[9]
    pspace=fname.split('_')[11]
    pspace=pspace.split('.txt')[0]
    ff=glob.glob(input_rad_destination+'\\*')
    for f in ff :
        os.remove(f)
    
    shutil.copy(fname,input_rad_destination)
    
    # rename files to be used in this run using standard naming convention 
    # naming convention is "rad_values+{shading fraction}.txt
    fil=glob.glob(input_rad_destination+'\\*')
    for i in range(0,len(fil)):
        zstr=str(i+1)
        os.rename(fil[i],input_rad_destination+'\\rad_values_'+zstr.zfill(3)+'.txt')
    
    # update weather file 
    
    update_srad(srad_name)
    
    
    #start=datetime.datetime.now() # just used for timing 
    subprocess.check_call(command_to_setpath, shell=True) # set the path
    os.chdir(output_dir) # set the directory to store output files 
    
    # the next line is the one that actually runs dssat and will take the longest
    subprocess.check_call(input_string, shell=True) 
    #print(datetime.datetime.now()-start)
     
    # These lines open the batch file in order to get the correct 
    # experiment name in DSSAT, that is required to correctly access
    # the output files 
    batch_file_name=output_dir+'\\'+batch_file
    batch_file_csv=pd.read_csv(batch_file_name,header=2,delim_whitespace=True)
    dssat_experiment_name=batch_file_csv.iloc[2]['!']
    
    # Open the output files and extract yield 
    outputFile=output_dir+'\\'+dssat_experiment_name[:-4]+'.OSU'
    outputFile_csv=pd.read_csv(outputFile,header=2, delim_whitespace='True')
    outputFile_csv=outputFile_csv.shift(1,axis=1)
    crp_yield=pd.DataFrame(outputFile_csv['HWAM']) # New variable which has yield at harvest 
    crp_yield['Irr num application']=outputFile_csv['IR#M']
    crp_yield['Irr (mm)']=outputFile_csv['IRCM']
    crp_yield['Precip (mm)']=outputFile_csv['PRCM']
    crp_yield['ET (mm)']=outputFile_csv['ETCM']
    crp_yield['Tr (mm)']=outputFile_csv['EPCM']
    crp_yield['Evap (mm)']=outputFile_csv['ESCM']
    crp_yield['AVG rad (MJ/m2/day)']=outputFile_csv['SRADA']
    crp_yield['Dry matter ET productivity(kg/ha/mm)']=outputFile_csv['DMPEM']
    crp_yield['Dry matter irrig productivity(kg/ha/mm)']=outputFile_csv['DMPIM']
    crp_yield['Panel Height (feet)']=float(pheight)
    crp_yield['Panel row']=float(pspace)*3+4
    
    return crp_yield

#%%
def calc_spatial_yields(pheight,row_space,data):
    dat=data.get_group(pheight)
    space_idx=row_spacings.index(row_space)
    y=np.mean(dat['Yield Value Kg/ha'].iloc[0:space_idx+1])
    irr=np.mean(dat['Irr (mm)'].iloc[0:space_idx+1])
    prcp=np.mean(dat['Irr (mm)'].iloc[0:space_idx+1])
    
    return y,irr,prcp
#%%
start=datetime.datetime.now() # just used for timing 

crp_out=[]

for i in range(0,len(all_fi)): # run dssat for all other shade values 
    crp_yield = run_dssat_single(all_fi['hourly'][i],all_fi['Srad'][i])
    crp_out.append(crp_yield)
    
print(datetime.datetime.now()-start)

allc=pd.DataFrame(np.vstack(crp_out))
allc.columns=(['Yield Value Kg/ha','Irr num application','Irr (mm)','Precip (mm)','ET (mm)','Tr (mm)',
   'Evap (mm)','AVG rad (MJ/m2/day)','Dry matter ET productivity(kg/ha/mm)','Dry matter irrig productivity(kg/ha/mm)',
   'Panel Height (feet)','Panel row'])

#%%
allc.loc[0,'Panel row']=0
#%%
row_spacings=[4,7,10,13,16,19,22,25]

pheights=[1,2,3,4,5,6,7,8,9,11,10,12,13,14,15]

agg_y=pd.DataFrame(np.zeros((len(pheights)*len(row_spacings)+1,5)))
agg_y.columns=['Yield (kg/ha)','Irrig (mm)','Precip (mm)','Panel Height', 'Row Spacing']

height_groups=allc.groupby('Panel Height (feet)')

countt=0
for i in pheights:
   for j in row_spacings:
       agg_y.loc[countt,'Panel Height']=i*0.3048
       agg_y.loc[countt,'Row Spacing']=j
       yield_agg,irr_agg,prcp=calc_spatial_yields(i,j,height_groups)
       agg_y.loc[countt,'Yield (kg/ha)']=yield_agg
       agg_y.loc[countt,'Irrig (mm)']=irr_agg
       agg_y.loc[countt,'Precip (mm)']=prcp
       
       
       
       agg_y.loc[countt,'Irrig (mm)']=allc[(allc['Panel Height (feet)']==0)&(allc['Panel row']==0)]['Irr (mm)'][0]
       agg_y.loc[countt,'Precip (mm)']=allc[(allc['Panel Height (feet)']==0)&(allc['Panel row']==0)]['Precip (mm)'][0]
       countt+=1
agg_y.loc[countt,'Yield (kg/ha)']=allc[(allc['Panel Height (feet)']==0)&(allc['Panel row']==0)]['Yield Value Kg/ha'][0]

agg_y['Yieldfrac']= agg_y['Yield (kg/ha)']/agg_y.loc[countt,'Yield (kg/ha)']
agg_y['WUE (kg/mm)']=agg_y['Yield (kg/ha)']/(agg_y['Irrig (mm)']+agg_y['Precip (mm)'])
agg_y['IUE (kg/mm)']=agg_y['Yield (kg/ha)']/agg_y['Irrig (mm)']
#%%
agg_y_space=agg_y[agg_y['Row Spacing']>6]

contour_data=pd.DataFrame(np.zeros((len(agg_y_space),3)))
contour_data.columns=['x','y','z']
contour_data['x']=agg_y_space['Panel Height'].values
contour_data['y']=agg_y_space['Row Spacing'].values
contour_data['z']=agg_y_space['Yieldfrac'].values

Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

X_unique = np.sort(contour_data.x.unique())
Y_unique = np.sort(contour_data.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)


# Initialize plot objects
rcParams['figure.figsize'] = 8, 8 # sets plot size
fig = plt.figure(dpi=1200)
ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels =np.arange(0.7,1,0.015)#np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])
#levels =  np.arange(1,2.51,0.125) # rain fed
minn=0.7
maxx=1

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = ax.contourf(X,Y,Z,levels, cmap='viridis',vmin=minn,vmax=maxx)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = ax.contour(X, Y, Z, levels=levels, colors='Black')
ax.clabel(cp, np.arange(0.7,1,0.03), fontsize=10, colors=line_colors )
#ax.clabel(cp,np.arange(1,2.51,0.25), fontsize=10, colors=line_colors )

plt.xticks(np.arange(0.5,5,0.5))
#plt.yticks([0,0.5,1])
ax.set_xlabel('Panel Height (m)',fontsize=15,labelpad=10)
ax.set_ylabel('Distance from Panel (m)',fontsize=15,labelpad=10)
ax.set_title('Yield Fraction (AV Yield [kg/ha] / Control Yield [kg/ha])', y=1.05, fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1)


#%%

yieldfrac_target=0.8

above_contour=agg_y[agg_y['Yieldfrac']>=yieldfrac_target]
#%%
ec_array=reg_pickle_open(r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\pickles\economics_array.p')
ec=ec_array[['Panel Height (meters)','Panel Spacing (meters)','net $/ha', 'PV rev','NPV']]
#%%

#%%
yieldNecon = (pd.merge(agg_y, ec,  how='left', 
            left_on=['Panel Height','Row Spacing'], right_on = ['Panel Height (meters)','Panel Spacing (meters)']))
#%%
bean_price_Cwt=35 # $/cwt
bean_price_kg=bean_price_Cwt/50.8
project_life=20 # years
yieldNecon['Yield return ($)']=yieldNecon['Yield (kg/ha)']*bean_price_kg*project_life
yieldNecon['Total profit']=yieldNecon['Yield return ($)']+yieldNecon['net $/ha']
yieldNecon['rev comparison']=yieldNecon['Yield return ($)']/yieldNecon['PV rev']



#%%
yieldNecon_space=yieldNecon[yieldNecon['Row Spacing']>5]
contour_data=pd.DataFrame(np.zeros((len(yieldNecon_space),3)))
contour_data.columns=['x','y','z']
contour_data['x']=yieldNecon_space['Panel Height'].values
contour_data['y']=yieldNecon_space['Row Spacing'].values
contour_data['z']=yieldNecon_space['rev comparison'].values

Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

X_unique = np.sort(contour_data.x.unique())
Y_unique = np.sort(contour_data.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)

# Initialize plot objects
rcParams['figure.figsize'] = 8, 8 # sets plot size
fig = plt.figure(dpi=1200)
ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = 9#np.arange(-1075030,0.1,107503) #np.arange(0.7,1,0.015)#np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])
#levels =  np.arange(1,2.51,0.125) # rain fed
#minn=-1075030
#maxx=0

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = ax.contourf(X,Y,Z,levels, cmap='viridis')#vmin=minn,vmax=maxx)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = ax.contour(X, Y, Z, levels=levels, colors='Black', linestyles='-')
ax.clabel(cp, fontsize=10, colors=line_colors )

plt.xticks(np.arange(0.5,5,0.5))
#plt.yticks([0,0.5,1])
ax.set_xlabel('Panel Height (m)',fontsize=15,labelpad=10)
ax.set_ylabel('Row Spacing (m)',fontsize=15,labelpad=10)
ax.set_title('Revenue Comparison (Ag. Yield Rev. / PV Rev.)', y=1.05, fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1)

above_contour=yieldNecon[yieldNecon['Yieldfrac']>=yieldfrac_target]
#%% Tractor stuff 
ft_to_m=0.3048
c_tractor_w=6.5*ft_to_m # meter
c_tractor_h=8*ft_to_m # meter

space_need=c_tractor_w*2
h_need=c_tractor_h+0.3048

#%%
above_contour_space=above_contour[above_contour['Panel Spacing (meters)']>5]

contour_data=pd.DataFrame(np.zeros((len(above_contour_space),3)))
contour_data.columns=['x','y','z']
contour_data['x']=above_contour_space['Panel Height (meters)'].values
contour_data['y']=above_contour_space['Panel Spacing (meters)'].values
contour_data['z']=above_contour_space['NPV'].values

Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

X_unique = np.sort(contour_data.x.unique())
Y_unique = np.sort(contour_data.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)


minn=np.min(contour_data['z'])
maxx=np.max(contour_data['z'])
# Initialize plot objects
rcParams['figure.figsize'] = 15, 12 # sets plot size
fig = plt.figure(dpi=1200)
ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.arange(minn,0.1,np.abs(minn/10)) #np.arange(0.7,1,0.015)#np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])
#levels =  np.arange(1,2.51,0.125) # rain fed
#minn=-1075030
#maxx=0

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = ax.contourf(X,Y,Z,levels, cmap='viridis')#vmin=minn,vmax=maxx)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = ax.contour(X, Y, Z, levels=levels, colors='Black', linestyles='-')
ax.clabel(cp, fontsize=10, colors=line_colors )
#ax.clabel(cp,np.arange(1,2.51,0.25), fontsize=10, colors=line_colors )

plt.xticks(np.arange(0.5,5,0.5))
#plt.yticks([0,0.5,1])
ax.set_xlabel('Panel Height (m)',fontsize=15,labelpad=10)
ax.set_ylabel('Row Spacing (m)',fontsize=15,labelpad=10)
#ax.set_title('Net Present Value of PV Array After 20 Years', y=1.05, fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1)

# Initialize plot objects
#rcParams['figure.figsize'] = 8, 8 # sets plot size
#fig = plt.figure()#dpi=1200)
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.arange(0,maxx+0.1,maxx/10) #np.arange(0.7,1,0.015)#np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])
#levels =  np.arange(1,2.51,0.125) # rain fed
#minn=-1075030
#maxx=0

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = ax.contourf(X,Y,Z,levels, cmap='YlOrRd')#vmin=minn,vmax=maxx)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = ax.contour(X, Y, Z, levels=levels, colors='Black', linestyles='-')
ax.clabel(cp, fontsize=10, colors=line_colors )
#ax.clabel(cp,np.arange(1,2.51,0.25), fontsize=10, colors=line_colors )

plt.xticks(np.arange(0.5,5,0.5))
#plt.yticks([0,0.5,1])
ax.set_xlabel('Panel Height (m)',fontsize=15,labelpad=10)
ax.set_ylabel('Row Spacing (m)',fontsize=15,labelpad=10)
ax.set_title('NPV After 20 Years (Yield Fraction >0.8)', y=1.05, fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1, location='left')
plt.axvline(x = c_tractor_h, color = 'red', label = 'Typical Tractor Height', linestyle='--')

plt.legend(loc='lower right')

