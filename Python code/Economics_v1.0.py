# -*- coding: utf-8 -*-
"""
2022-06-03

This script calculates electrical outputs and various 
economic considerations for the AV-EAS framework

@author: proctork@oregonstate.edu
"""
# Import libraries

import numpy as np
import pandas as pd
import glob 
import datetime
from matplotlib import pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rcParams
def reg_pickle_open(filename):
    with open(filename, 'rb') as f:
        pFile=pickle.load(f)
    
    return pFile

def reg_pickle_dump (save_dir,filename,data):
  # pickle files
    with open(save_dir+filename+'.p','wb') as f:
        pickle.dump(data,f)

#%%
def read_in_PVdata(data_folder_path,height_path):
    
    
    data_list_pv=[]

    for file in glob.glob(data_folder_path+'\\'+height_path+'\\PVout*.csv'):
        data_PV=pd.read_csv(file)
        data_list_pv.append(data_PV)

    for file in glob.glob(data_folder_path+'\\'+height_path+'\\model_param*.csv'):
        data_param=pd.read_csv(file)

    pvData_OG=pd.concat(data_list_pv,axis=0,ignore_index=True) # original, column titles describe units


    params=data_param
    p_length=params['Panel length (feet)'][0]
    p_width=params['Panel width (feet)'][0]
    p_height=params['Panel height (feet)'][0]
    p_tilt=params['Panel tilt (degree)'][0]
    p_area=params['PV panel area (ft^2)'][0]
    p_area_m=p_area/10.764
    impact_x=params['Panel rad impact X(feet)'][0]
    impact_y=params['Panel rad impact Y(feet)'][0]
    gSpacing=params["Grid spacing(feet)"][0]

    # Create a data file with only relevant data and shorter column names 
  

    pvData=pvData_OG[["Start Time", "Stop Time","X coordinate (feet)","Y coordinate (feet)","Z coordinate (feet)","Cumulative Solar radiation (Kwh/m^2)"]].copy()
    #It says kwh/m2 but unit is actually wh/m2/day
    pvData.columns='t0','tf','x','y','z','solar'
    
    
    return pvData, p_length, p_width, p_height, p_tilt, p_area_m, impact_x, impact_y, gSpacing

#%% This section calculates factors related to PV electricity 
#   Production 
def get_PV(pvData):
    """
    Calculates daily average cumulative radiation for a single panel.
    Cumulative radiation is averaged spatially over the face of the panel
    

    Parameters
    ----------
    pvData : DataFrame (units=wh/m^2)
        Daily Solar radiation incident on the panel surface 
        at various points defined by their x,y,z coordinates.
        The grid size is described by the gSpacing variable (feet)
        Solar radiation is summed daily for the period 4:30-21:30
        This array is imported from revit

    Returns
    -------
    daily_mean_PV : DataFrame (units=wh/m^2)
       Cumulative radiation per m/^2 for a single panel for each
       day of the period of interest.
    

    """
    
    time_info=pvData['t0'].apply(datetime.datetime.strptime,args=("%Y-%m-%d %H:%M:%S",))
    dates=time_info.apply(datetime.datetime.strftime,args=('%Y/%m/%d',))
    unique_dates_pv=dates.unique()
    unique_dates_pv=np.sort(unique_dates_pv)
    
    pvData['Date']=dates
    daily_mean_PV=pd.DataFrame(np.zeros((len(unique_dates_pv),1)))
    daily_mean_PV.index=unique_dates_pv
    for i in unique_dates_pv:
        day_pv=pvData[pvData['Date']==i]
        mean_pv=np.mean(day_pv['solar'])
        daily_mean_PV.loc[i]=mean_pv
    return daily_mean_PV

#%%
# various economic costs
e_price=0.0899 # $/kWh electricity cost

## PV costs from NREL Cost Benchmark Q1 2021
#costs in $/Wdc
module_cost = 0.34  
inverter_cost=0.08
electrical_cost=0.29 # electrical components between $0.13-0.45/W

module_size=2 #m^2 area

module_watt=300 #watts/module

pvEff=0.199 # efficiency (NREL Cost Benchmark)

p_cost_no_rack=module_cost+inverter_cost+electrical_cost

#%%
ft_to_m=0.3048 # conversion factor
c_tractor_w=6.5*ft_to_m # meter typical tractor width
c_tractor_h=8*ft_to_m # meter typical tractor height
#%%
def racking_cost_per_watt(panel_height):
    """
    Calculate cost per watt based on correspondence with 
    RBI Solar Inc.

    Parameters
    ----------
    panel_height : float
        Panel clearance height

    Returns
    -------
    cost : float
        racking cost #/watt

    """
    if panel_height==0:
        cost=0
    elif (panel_height>0)&(panel_height<=8):
        cost_MW=6062.5*panel_height**2-18175*panel_height+338750
        cost=cost_MW/(1*10**6)
        
    else:
        # use same equation + 3* the cost at height of 8 feet to 
        # account for carport style
        cost_MW=(3*(6062.5*8**2-18175*8+338750)+
                 6062.5*panel_height**2-18175*panel_height+338750)
        cost=cost_MW/(1*10**6)
        
    return cost
#%%
fig=plt.figure(dpi=1200)
xxa=np.arange(0,15,.05)
ab=[]
for i in xxa:
    aa=racking_cost_per_watt(i)
    ab.append(aa)
plt.plot(xxa*ft_to_m,ab,color='Black')
plt.xlabel('PV Panel Height (m)')
plt.ylabel('Cost  ($/Watt)')
plt.title('Racking Cost vs Panel Height', y=1.05)
plt.show()
#%%
ac=pd.DataFrame(ab)+p_cost_no_rack
#%%
# read in data 
# radiation at panel surface 

#panel_height_strs=['03_feet','05_feet','07_feet','09_feet','11_feet','13_feet','15_feet']
panel_height_strs=['01_feet','02_feet','03_feet','04_feet','05_feet','06_feet','07_feet','08_feet','09_feet','10_feet','11_feet','12_feet','13_feet','14_feet','15_feet']
row_spacings=np.arange(4,26,3)

panel_heights=[1,2,3,4,5,6,7,8,9,11,10,12,13,14,15]

field_area=10000 # assume an area of 10000 square meteres, 1 ha
field_area_ft=field_area*10.764
field_depth=np.sqrt(field_area_ft) # assuming square field 
panel_life=20
ONM=17.92 #$/kW/year
#%%
"""
row_spacing=panel_spacings[5]
num_rows=int(np.floor(field_depth/row_spacing)) # number of rows of panels
num_p_per_row=int(np.floor(field_depth/p_length)) # number of panels per row
num_panels=int(num_rows*num_p_per_row) # total panels in array
"""
#%%
econ_array=pd.DataFrame(np.zeros((len(panel_heights)*len(row_spacings),11)))
econ_array.columns=(['Panel Height (meters)', 'Panel Spacing (meters)','Num rows', 'Num panels',
                     'Rad per panel (wh/panel)', 'Rad total(kWh)', 'Energy (kWh)', 'PV cost', 'PV rev','net $/ha','NPV'])
counter=0
data_folder_path=r"C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\panel_clearance_height"
degredation=0.05/100
discount_rate=0.06

panel_height_test=[9]
for i in panel_heights:
    if i <10:
        height_path=f'0{i}_feet'
    else:
        height_path=f'{i}_feet'
    (pvData, p_length, p_width, p_height, p_tilt,
         p_area, impact_x, impact_y, gSpacing)=read_in_PVdata(data_folder_path,height_path)
    # Units of ft except for p_area which is m^2
    module_per_panel=np.round(p_area)/module_size # num module
    racking_cost=racking_cost_per_watt(i) #$/watt
    
    PV_cost_per_watt=p_cost_no_rack+racking_cost # $/watt

    daily_mean_PV=get_PV(pvData) #wh/ft^2
    pvIntensity_panel=daily_mean_PV*p_area #wh/panel/year
    degredation=0.05/100
    discount_rate=0.06
    
    for j in row_spacings:
        row_spacing=j*3.28084
        num_rows=int(np.floor(field_depth/row_spacing)) # number of rows of panels
        num_p_per_row=int(np.floor(field_depth/p_length)) # number of panels per row
        num_panels=int(num_rows*num_p_per_row) # total panels in array

        num_module=module_per_panel*num_panels
        watt_capacity=num_module*module_watt # watts
        np_capacity=watt_capacity/1000 # kW
        #nameplate capacity
        panel_Capex=PV_cost_per_watt*watt_capacity #$
        #that was cost

        ONM_cost=ONM*np_capacity

        # now revenues 

        pv_intensity_total=np.sum(pvIntensity_panel)[0]*num_panels/1000 #kwh
        annual_cost_array=pd.DataFrame(np.zeros((panel_life+1,6)))
        annual_cost_array.columns=['Year of project','Capex','Energy out (kWh)','PV rev', 'Net','NPV']
        annual_cost_array['Year of project']=np.arange(0,panel_life+1,1)
        annual_cost_array.loc[0,'Capex']=panel_Capex
        annual_cost_array['Energy out (kWh)']=pv_intensity_total*(pvEff-degredation*annual_cost_array['Year of project']) #kWh
        annual_cost_array.loc[0,'Energy out (kWh)']=0
        annual_cost_array['PV rev']=annual_cost_array['Energy out (kWh)']*e_price
        annual_cost_array['Net']=annual_cost_array['PV rev']-annual_cost_array['Capex']-ONM_cost
        annual_cost_array.loc[0,'Net']=-annual_cost_array['Capex'][0]
        annual_cost_array['NPV']=annual_cost_array['Net']/(1+discount_rate)**(annual_cost_array['Year of project'])
        # need to apply discount rate to get npv 

        annual_NPV=np.sum(annual_cost_array['NPV'])
        
        econ_array.loc[counter,'Panel Height (meters)']=i*0.3048
        econ_array.loc[counter,'Panel Spacing (meters)']=j
        econ_array.loc[counter,'Num rows']=num_rows
        econ_array.loc[counter,'Num panels']=num_panels
        econ_array.loc[counter,'Rad per panel (wh/panel)']=np.sum(pvIntensity_panel)[0]
        econ_array.loc[counter,'Rad total(kWh)']=pv_intensity_total
        econ_array.loc[counter,'Energy (kWh)']=np.sum(annual_cost_array['Energy out (kWh)'])
        econ_array.loc[counter,'PV cost']=panel_Capex
        econ_array.loc[counter,'PV rev']=np.sum(annual_cost_array['PV rev'])
        econ_array.loc[counter,'net $/ha']=np.sum(annual_cost_array['Net'])
        econ_array.loc[counter,'NPV']=np.sum(annual_cost_array['NPV'])                             
        counter+=1
                                                
#%%
writedir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\pickles'
reg_pickle_dump (writedir,'\\economics_array',econ_array)
#%%
econ_array_space=econ_array[econ_array['Panel Spacing (meters)']>6]


contour_data=pd.DataFrame(np.zeros((len(econ_array_space),3)))
contour_data.columns=['x','y','z']
contour_data['x']=econ_array_space['Panel Height (meters)'].values
contour_data['y']=econ_array_space['Panel Spacing (meters)'].values
contour_data['z']=econ_array_space['NPV'].values

Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

X_unique = np.sort(contour_data.x.unique())
Y_unique = np.sort(contour_data.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)

# Initialize plot objects
rcParams['figure.figsize'] = 15, 12 # sets plot size
fig = plt.figure()#dpi=1200)
ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.arange(-1075030,0.1,107503) #np.arange(0.7,1,0.015)#np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])
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
ax.set_title('Net Present Value of PV Array After 20 Years)', y=1.05, fontsize=15)
plt.colorbar(cpf, pad=0.1)


# Define levels in z-axis where we want lines to appear
levels = np.arange(0,73121,7300) #np.arange(0.7,1,0.015)#np.arange(0.78,1.02,0.02)#np.array([-0.4,-0.2,0,0.2,0.4])
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
ax.set_title('Net Present Value of PV Array After 20 Years', y=1.05, fontsize=15)
#plt.clim(0.78,1)
plt.colorbar(cpf, pad=0.1, location='left')

