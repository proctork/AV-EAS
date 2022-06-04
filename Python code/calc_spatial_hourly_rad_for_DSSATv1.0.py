# -*- coding: utf-8 -*-
"""
2022-06-03

This is the second script of the AV-EAS framework, it takes
as inputs the spatialRad pickle files from the 
calculate_and_save_radSpatial.py script and outputs a set 
of text files containing hourly solar radiation for each spatial
location and each height.
These .txt files will be used by the run_DSSAT_get_yield.py 
script.

@author: proctork@oregonatat.edu
"""
#Read in files and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime
import pickle
import pvlib
import timezonefinder
import pytz
import glob
import os
import copy
from mpl_toolkits.mplot3d import Axes3D


def reg_pickle_open(filename):
    with open(filename, 'rb') as f:
        pFile=pickle.load(f)
    
    return pFile

def reg_pickle_dump (save_dir,filename,data):
  # pickle files
    with open(save_dir+filename+'.p','wb') as f:
        pickle.dump(data,f)

#%%
def setup_rows_sp(impact_end):
    
    start=int(origin_y/gSpacing)+1
    section_length=3 #m
    num_sections=int(impact_end/section_length)

    row_boundaries=pd.DataFrame(np.zeros((num_sections-1,2)),dtype=object)
    row_boundaries.columns=('Start','Stop')

    for i in range(0,num_sections-1):
         row_boundaries.loc[i,'Start']=i*section_length
         row_boundaries.loc[i,'Stop']=(i+1)*section_length  

    row_boundaries=row_boundaries+start
    xsp=int((origin_x-p_length/2)/gSpacing) # x start point
    xstp=int((origin_x+(num_p_per_row-1)*off_set)/gSpacing) # x stop point
    
    return row_boundaries, xsp, xstp


#%%
def get_row_wise_rad_data(radSpatial,row_boundaries):
    """
    Creates a new dictionary which contains the spatial radiation data for each
    individual row

    Parameters
    ----------
    radSpatial : dict
        Dictionary of spatial solar radiation.
        Keys are day of year (unique_dates) and hour of day (unique_hours)
        While the allRadData holds the solar radiation data for a single panel
        as x, y coordinates and solar radiation values, the radSpatial array,
        holds a list which contains solar radiation for the entire PV array
        x and y coordinates are represented by the rows and columns with the
        distance per grid cell being defined by the gSpacing parameter

    Returns
    -------
    row_wise_radSpatial : dict
        Dictionary of spatial solar radiation seperated by rows.
        This dict is very similair to radSpatial with the major exception 
        being that each hour holds multiple smaller arrays which contain
        the data for a single row. The number of arrays will be equal to the
        number of rows (num_rows).
        If there is no radiation at all (night/early morning) the array will
        be replaced by the int 0.
        The individual rows are labeled based on the y location with units of
        feet. So gridlocation *gSpacing
        
    

    """
    row_wise_radSpatial=copy.deepcopy(radSpatial)
    
    for i in unique_dates:
        for j in unique_hours:
            row_data=[]
            for z in range(0,len(row_boundaries)):
                
                if np.max(np.max(radSpatial[i][j]))==0:
                    current_row=0
                # if no radiation is present it is computationally unnecessary
                # to have an array of 0s
                else:
                    current_row=radSpatial[i][j].iloc[row_boundaries['Start'][z]:row_boundaries['Stop'][z],xsp:xstp]
                row_data.append(current_row)
            
            row_wise_radSpatial[i][j]=dict(zip(row_boundaries['Start'],row_data))
    return row_wise_radSpatial
#%%
def calc_sunup_sundown(year):
    """
    calculate the hour of sunrise and sunset
    These values are dependent on the location but that aspect is calculated 
    before the function during the creation fo the tz variable 
    
    This calculation is the same as the one used in the DSSAT SOLAR.FOR file

    Parameters
    ----------
    year : int
        year of interest 

    Returns
    -------
    su : int 
        sunrise hour 
    sd : int
        sunset hour 

    """
    start_t=f'{year}-01-01'
    end_t=f'{year}-12-31'
    t_range = pd.date_range(start_t, end_t).tolist()


    t_range=pd.to_datetime(t_range)
    t_range=t_range.tz_localize(tz)
    pi_=3.14159 # from DSSAT
    RAD=pi_/180 # from DSSAT 
    doy=t_range.dayofyear
    dec=-23.45*np.cos(2*pi_*(doy+10)/365)
    soc=np.tan(RAD*dec)*np.tan(RAD*lat_AR)
    dayll= 12+24*np.arcsin(soc)/pi_
    snupp= 12 - dayll/2
    sndnn= 12 + dayll/2
    
    hour1=4
    hourn=21


    su=np.int32(np.ceil(snupp)-1)-hour1
    sd=np.int32(np.round(sndnn))-hour1
    
    return su, sd
#%%

def calculate_rad_ratios_sp(radSpatial, row_wise_radSpatial):
    controlrow=[]
    for i in unique_dates:
        for j in unique_hours:
            control_hr=radSpatial[i][j].iloc[-1].iloc[-1]
            controlrow.append(control_hr)
    controlrow=pd.DataFrame(np.vstack(controlrow))

    #plt.plot(controlrow)


    rows=list(row_wise_radSpatial[unique_dates[0]][unique_hours[0]].keys())
    av_rows=pd.DataFrame(np.zeros((6205,len(rows))))
    #why 6205?
    av_rows.columns=list(rows)
                          
   
    for k in rows:
        avrow=[]
        for i in unique_dates:
            for j in unique_hours:
                    onerowhour=row_wise_radSpatial[i][j][k]
                    orh=np.mean(np.mean(onerowhour))
                    avrow.append(orh)
        av_rows[k]=avrow
                
    
    return controlrow, av_rows

#%%
def get_cumulative_rad_hourly(row_wise_radSpatial, row_boundaries):
    """
    Calculate the average hourly solar radiation for each row  

    Parameters
    ----------
    row_wise_radSpatial : TYPE
        Dictionary of spatial solar radiation seperated by rows.
        This dict is very similair to radSpatial with the major exception 
        being that each hour holds multiple smaller arrays which contain
        the data for a single row. The number of arrays will be equal to the
        number of rows (num_rows).
        The individual rows are labeled based on the y location with units of
        feet. So gridlocation *gSpacing
        The final row for a particular time stamp is always the control row
        aka no shading present.

    Returns
    -------
    radCumulative_hourly : dict
        Dictionary containing average cumulative solar radiation 
        for each row on an hourly basis.
        This data can be useful as an input for the DSSAT which can make use
        of hourly radiation data.
        Note that this approach loses the spatial variability which is contained
        in the input data and instead averages across the entire row.   

    """
    radCumulative=[]
    for i in unique_dates:
        hour_wise_list=pd.DataFrame(np.zeros((len(unique_hours),len(row_boundaries)+1)))
        hour_wise_list.index=unique_hours
        for j in unique_hours:
    
            current_rows=row_wise_radSpatial[i][j]
            for k in range(0,len(row_boundaries)):
                
                hour_wise_list.loc[j,k]=np.mean(np.mean(current_rows[row_boundaries['Start'][k]]))
            hour_wise_list.loc[j,len(row_boundaries)]=np.max(np.max(radSpatial[i][j]))
            # Final row is always the control 
            
        radCumulative.append(hour_wise_list)        
            
    radCumulative_hourly=dict(zip(unique_dates,radCumulative))
    
    return radCumulative_hourly

#%%
def get_rad_cumulative_daily(radCumulative_hourly,row_boundaries):
    """
    calculate the daily cumulative solar radaition for each row

    Parameters
    ----------
    radCumulative_hourly : dict
        Dictionary containing average cumulative solar radiation 
        for each row on an hourly basis.
        This data can be useful as an input for the DSSAT which can make use
        of hourly radiation data.

    Returns
    -------
    radCumulative_daily : dict
        Dictionary containing cumulative solar radiation for each row on a
        daily basis.
        This data can be useful as in input for the AquaCrop model which requires
        daily radiation values to estimate reference evapotranspiration (ET0)

    """
    radCumulative_daily=pd.DataFrame(np.zeros((len(unique_dates),len(row_boundaries)+1)))
    radCumulative_daily.index=unique_dates
    for i in unique_dates:
        for j in range(0,len(row_boundaries)+1):
            radCumulative_daily.loc[i][j]=np.sum(radCumulative_hourly[i][j])
    return radCumulative_daily

#%%

def add_night_time_rad(radCumulative_hourly,row_type,row_boundaries):
    """
    The revit calculations does not calculate solar radiation
    during the night time, this function creates a new array
    which includes those hours and sets the rad. value equal to 
    zero


    """
    full_day_index=['00:30','01:30','02:30','03:30','04:30','05:30','06:30','07:30',
                    '08:30','09:30','10:30','11:30','12:30','13:30','14:30','15:30',
                    '16:30','17:30','18:30','19:30','20:30','21:30','22:30','23:30']
    
    all_hours=[]
    
    for i in unique_dates:
        full_day=pd.DataFrame(np.zeros((24,len(row_boundaries)+1)))
        full_day.index=full_day_index
        full_day.loc[unique_hours]=radCumulative_hourly[i]
        
        all_hours.append(full_day)
    
    all_hours=pd.concat(all_hours)
   
    return all_hours
#%%
def save_hourly_rad_file(all_hours,row_type):
    """
    This function writes the hourly rad values to a csv file
    with a set naming convention

    """
    if row_type=='AV':
        for i in range(0,len(all_hours.columns)-1):
            hour_rad_file='DSSAT_hourly_rad_'+'height_'+str(p_height)+f'_row_{i}'
            all_hours.iloc[:,i].to_csv(write_dir+'/'+hour_rad_file+'.txt',index=False, header=False)
    
    
    
    else:
        hour_rad_file='DSSAT_hourly_rad_'+'height_'+'0_space_0'
        all_hours.iloc[:,-1].to_csv(write_dir+'/Con'+hour_rad_file+'.txt',index=False, header=False)
#%%
def save_daily_rad_file(radCumulative_daily_MJ,row_type):
    """
    This function writes the daily rad values to a csv file
    with a set naming convention
    """
    if row_type=='AV':
        for i in range(0,len(radCumulative_daily_MJ.columns)-1):
            hour_rad_file='DSSAT_hourly_rad_'+'height_'+str(p_height)+f'_row_{i}'
            radCumulative_daily_MJ.iloc[:,i].to_csv(write_dir_day+'/'+hour_rad_file+'.txt',index=False, header=False)
    
    
    
    else:
        hour_rad_file='DSSAT_hourly_rad_'+'height_'+'0_space_0'
        radCumulative_daily_MJ.iloc[:,-1].to_csv(write_dir_day+'/Con'+hour_rad_file+'.txt',index=False, header=False)

#%%

read_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\panel_clearance_height'
param_dir=read_dir+'\\parameter_pickles'
# these should be pickled from the other script 
params=reg_pickle_open(param_dir+'\\param_array.p')
#%%
gSpacing=params.loc['gSpacing'][0]
origin_x=params.loc['impact_x'][0]/2
origin_y=params.loc['origin_y'][0]
num_p_per_row=params.loc['num_p_per_row'][0]

num_rows=params.loc['num_rows'][0]
p_length=params.loc['p_length'][0]
p_width=params.loc['p_width'][0]
off_set=p_length
lat_AR=params.loc['lat'][0]
lon_AR=params.loc['lon'][0]

tf=timezonefinder.TimezoneFinder()
timezone_str=tf.certain_timezone_at(lat=lat_AR, lng=lon_AR)
tz=pytz.timezone(timezone_str)
radSdir=read_dir+'\\radSpatial_pickles'

unique_dates=reg_pickle_open(param_dir+'\\Unique_dates.p')
unique_hours=reg_pickle_open(param_dir+'\\Unique_hours.p')

su,sd =calc_sunup_sundown(2020)


#%%

def get_results_for_row_spacing(radSpatial,row_type,row_boundaries, xsp, xstp):
    
    
    row_wise_radSpatial=get_row_wise_rad_data(radSpatial,row_boundaries)
    
    controlrow, av_rows =calculate_rad_ratios_sp(radSpatial, row_wise_radSpatial)
    
    radCumulative_hourly=get_cumulative_rad_hourly(row_wise_radSpatial,row_boundaries)
    radCumulative_daily=get_rad_cumulative_daily(radCumulative_hourly,row_boundaries)
    radCumulative_daily_MJ=radCumulative_daily*0.0036 # MJ/m2/day
    #final column is the control
    all_hours=add_night_time_rad(radCumulative_hourly,row_type,row_boundaries)
    save_hourly_rad_file(all_hours,row_type)
    save_daily_rad_file(radCumulative_daily_MJ,row_type)
    
    su,sd = calc_sunup_sundown(2020) # sun up and sundown for 2020
    
    return controlrow, av_rows
#%%

def get_result_for_panel_height(panel_height) :
  
  fname=panel_height
  p_height=int(h[0:2])
  radSpatial=reg_pickle_open(radSdir+'\\'+fname+'.p')
  impact_end=np.shape(radSpatial[unique_dates[0]][unique_hours[0]])[0]
  row_boundaries, xsp, xstp=setup_rows_sp(impact_end)
  
  return radSpatial, impact_end,p_height,row_boundaries, xsp, xstp
  

#%%

# for every panel and spacing combo save the linear relation
# coefficients between max and min sun each day, the rad ratio,
# and save radiation in DSSAT format 

panel_height_strs=['01_feet','02_feet','03_feet','04_feet','05_feet','06_feet','07_feet','08_feet','09_feet','10_feet','11_feet','12_feet','13_feet','14_feet','15_feet']

h=panel_height_strs[0]
template= get_result_for_panel_height(h)
row_boundaries_ex=template[3]

row_n_str=(np.arange(0,len(row_boundaries_ex))).astype(str)
col_name_av=['AV_' + s for s in row_n_str]
col_name_rad_rat=['radRat_'+ s for s in row_n_str] 

col_names=['Crad']+col_name_av+col_name_rad_rat+['pHeight']

#%%

counter=0
st=17 # time step, 17 hours per day Left out hours that are always zero

write_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\DSSAT_AVrad_spatial'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)
write_dir_day=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\DSSAT_AVrad_spatial_day'  
if not os.path.exists(write_dir_day):
    os.makedirs(write_dir_day)  

rad_array_config=pd.DataFrame(np.zeros((len(panel_height_strs),2+len(row_boundaries_ex)*2)))
rad_array_config.columns=col_names
for h in panel_height_strs:
    fname=h
    p_height=int(h[0:2])
    (radSpatial, impact_end,p_height,
      row_boundaries, xsp, xstp)= get_result_for_panel_height(h)
    
    controlrow, av_rows=( get_results_for_row_spacing(radSpatial,'AV',
                                row_boundaries, xsp, xstp))
    rad_array_config.loc[counter,'Crad']=np.sum(controlrow)[0]
    for n in range(0,len(row_boundaries_ex)):
        rad_array_config.loc[counter,'AV_'+str(n)]=np.sum(av_rows.iloc[:,n])
        rad_array_config.loc[counter,'radRat_'+str(n)]=np.sum(av_rows.iloc[:,n])/np.sum(controlrow)[0]
      
    rad_array_config.loc[counter,'pHeight']=int(h[0:2])
   
 
    
    counter+=1             
#%%

#Get the control values
h="15_feet"
rt='Control'
(radSpatial, impact_end,p_height,
  row_boundaries, xsp, xstp)= get_result_for_panel_height(h)
controlrow, av_rows=( get_results_for_row_spacing(radSpatial,rt,
                             row_boundaries, xsp, xstp))
#%%    
wwrite_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out'
reg_pickle_dump (wwrite_dir,'\\results_array',rad_array_config)

#%%
#rad_array_config=reg_pickle_open(r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\results_array.p')

#%%
# rearange data to make for easier plotting 
rL=len(row_boundaries_ex)

plot_array=pd.DataFrame(np.zeros(((len(panel_height_strs)*rL,5))))
plot_array.columns=['pHeight', 'Distance From Panel','Crad', 'AVrad', 'radrat']

bruh=np.arange(0,rL*(len(panel_height_strs)-1)+1,rL)
section_length=3 # meters
for countit in bruh:
    plot_array.loc[countit:countit+rL,'pHeight']=rad_array_config.loc[countit/rL,'pHeight']*0.3048
    plot_array.loc[countit:countit+rL,'Crad']=rad_array_config.loc[countit/rL,'Crad']
   
    for n in range(0,rL):
     plot_array.loc[countit+n,'AVrad']=rad_array_config.loc[countit/rL,'AV_'+str(n)]
     plot_array.loc[countit+n,'Distance From Panel']=(n+1)*section_length
    
                                         
plot_array['radrat']=plot_array['AVrad']/plot_array['Crad']

#%%
#%%
plot_array_space=plot_array[plot_array['Distance From Panel']>5]

contour_data=pd.DataFrame(np.zeros((len(plot_array_space),3)))
contour_data.columns=['x','y','z']
contour_data['x']=plot_array_space['pHeight'].values
contour_data['y']=plot_array_space['Distance From Panel'].values
contour_data['z']=plot_array_space['radrat'].values

Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

X_unique = np.sort(contour_data.x.unique())
Y_unique = np.sort(contour_data.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)

# Initialize plot objects
rcParams['figure.figsize'] = 8, 8 # sets plot size
fig = plt.figure(dpi=1200)
ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = levels =np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = ax.contourf(X,Y,Z,levels, cmap='viridis',vmin=0.78,vmax=1)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = ax.contour(X, Y, Z, levels=levels, colors='Black')
ax.clabel(cp, fontsize=10, colors=line_colors)
plt.xticks(np.arange(0.5,5,0.5))
#plt.yticks([0,0.5,1])
ax.set_xlabel('Panel Height (m)',fontsize=15,labelpad=10)
ax.set_ylabel('Distance from Panel (m)',fontsize=15,labelpad=10)
ax.set_title('    Annual Solar Radiation Fraction (AV Rad. / Control Rad.)    ', y=1.05, fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1)

#plt.savefig('figure.pdf') # uncomment to save vector/high-res version

#%%
# run for a single height 
"""
h="15_feet"
rt='AV'
(radSpatial, impact_end,p_height,
  row_boundaries, xsp, xstp)= get_result_for_panel_height(h)

controlrow, av_rows=( get_results_for_row_spacing(radSpatial,'AV',
                            row_boundaries, xsp, xstp))
"""

#%%

#%%
"""
# this can be used to plot for figure 
fig = plt.figure()
ax = fig.add_subplot(111)
i=unique_dates[181]
j=unique_hours[7]
z=0
ah=radSpatial[i][j].iloc[row_boundaries['Start'][0]:row_boundaries['Stop'][6],16:116]
okay=ax.pcolormesh(ah,cmap='RdYlGn_r',vmin=0,vmax=780)

for i in row_boundaries['Start']-1:
    plt.axhline(y=i, color='b', linestyle='-')

plt.colorbar(okay)
plt.show()
"""

#%%
h=15
r=0
save_dir_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\dissertation_fig\v2'
read_dir_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\DSSAT_AVrad_spatial'
#%%
for r in range (0,7):
    pday=183
    hday=289
    hr=24
    start_t=pday*hr
    end_t=hday*hr
    fname=f'DSSAT_hourly_rad_height_{h}_row_{r}.txt'
    if (h==0 & r==0):
        fname='ConDSSAT_hourly_rad_height_0_space_0.txt'
    full=pd.read_csv(read_dir_dir+'\\'+fname)
    gs_rad =full.iloc[start_t:end_t]
    plt.ylim(0,780)
    plt.plot(gs_rad.values,linewidth=3,color='b')
    plt.savefig(save_dir_dir+f'\\SS_h_{h}_r_{r}')
    plt.show()
    print(r)


#%% # lets read in the values so I can plot things for the growing season 
h_range=np.arange(1,16,1)
r_range=np.arange(0,7,1)
#%%

def get_gs_rad(h,r):
    pday=183
    hday=289
    hr=24
    start_t=pday*hr
    end_t=hday*hr
    fname=f'DSSAT_hourly_rad_height_{h}_row_{r}.txt'
    if (h==0 & r==0):
        fname='ConDSSAT_hourly_rad_height_0_space_0.txt'
    full=pd.read_csv(read_dir_dir+'\\'+fname)
    gs_rad =full.iloc[start_t:end_t]
    
    return gs_rad

#%%

pday=183
hday=289
hr=24
start_t=pday*hr
end_t=hday*hr
h=15
#r=2

for r in r_range:
    fname=f'DSSAT_hourly_rad_height_{h}_row_{r}.txt'
    chk=pd.read_csv(read_dir_dir+'\\'+fname)
    chkchk=chk.iloc[start_t:end_t]
    plt.ylim(0,800)
    plt.plot(chk,color='b')
    plt.ylim(0,800)
    plt.xticks(np.arange(0,8760,50*24),(np.arange(0,8760,50*24)/24).astype(str))
    plt.plot(chkchk,color='orange')
    plt.savefig(save_dir_dir+f'\\season_rad_h_{h}_r_{r}')
    
    plt.show()

#%%

rad_array_gs=pd.DataFrame(np.zeros((len(panel_height_strs),2+len(row_boundaries_ex)*2)))
rad_array_gs.columns=col_names
counter=0
for h in h_range:
    for r in r_range:
        gs_radiation=get_gs_rad(h,r)
        rad_array_gs.loc[counter,'pHeight']=h
        rad_array_gs.loc[counter,'AV_'+str(r)]=np.sum(gs_radiation)[0]
    counter+=1
        
#%%
control=get_gs_rad(0,0)
rad_array_gs['Crad']=np.sum(control)[0]
for n in range(0,len(row_boundaries_ex)):
               rad_array_gs.loc[:,'radRat_'+str(n)]=rad_array_gs['AV_'+str(n)]/rad_array_gs['Crad']
      

#%%
rL=len(row_boundaries_ex)

plot_array_gs=pd.DataFrame(np.zeros(((len(panel_height_strs)*rL,5))))
plot_array_gs.columns=['pHeight', 'Distance From Panel','Crad', 'AVrad', 'radrat']

bruh=np.arange(0,rL*(len(panel_height_strs)-1)+1,rL)
section_length=3 # meters
for countit in bruh:
    plot_array_gs.loc[countit:countit+rL,'pHeight']=rad_array_gs.loc[countit/rL,'pHeight']*0.3048
    plot_array_gs.loc[countit:countit+rL,'Crad']=rad_array_gs.loc[countit/rL,'Crad']
   
    for n in range(0,rL):
     plot_array_gs.loc[countit+n,'AVrad']=rad_array_gs.loc[countit/rL,'AV_'+str(n)]
     plot_array_gs.loc[countit+n,'Distance From Panel']=(n+1)*section_length
    
                                         
plot_array_gs['radrat']=plot_array_gs['AVrad']/plot_array_gs['Crad']

#%%
plot_array_space=plot_array_gs[plot_array_gs['Distance From Panel']>5]

contour_data=pd.DataFrame(np.zeros((len(plot_array_space),3)))
contour_data.columns=['x','y','z']
contour_data['x']=plot_array_space['pHeight'].values
contour_data['y']=plot_array_space['Distance From Panel'].values
contour_data['z']=plot_array_space['radrat'].values


Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

X_unique = np.sort(contour_data.x.unique())
Y_unique = np.sort(contour_data.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)



"""
# Initialize plot objects
rcParams['figure.figsize'] = 5, 5 # sets plot size
fig = plt.figure()
ax = fig.add_subplot(111)

# Generate a contour plot
cp = ax.contour(X, Y, Z)
"""
# Initialize plot objects
rcParams['figure.figsize'] = 8, 8 # sets plot size
fig = plt.figure(dpi=1200)
ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels =np.arange(0.78,1.02,0.02)# 10#np.array([-0.4,-0.2,0,0.2,0.4])

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = ax.contourf(X,Y,Z,levels, cmap='viridis',vmin=0.78,vmax=1)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = ax.contour(X, Y, Z, levels=levels, colors='Black')
ax.clabel(cp, fontsize=10, colors=line_colors)
plt.xticks(np.arange(0.5,5,0.5))
#plt.yticks([0,0.5,1])
ax.set_xlabel('Panel Height (m)',fontsize=15,labelpad=10)
ax.set_ylabel('Distance from Panel (m)',fontsize=15,labelpad=10)
ax.set_title('Growing Season Solar Radiation Fraction (AV Rad. / Control Rad.)', y=1.05,fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1, ticks=(np.arange(0.78,1,0.04)))

     
        
