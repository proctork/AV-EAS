# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:48:10 2022

@author: proctork

Sensitivity Analysis 
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

e_price=0.0899 # $/kWh electricity cost

## PV costs from NREL Cost Benchmark Q1 2021
#costs in $/Wdc
module_cost = 0.34  
inverter_cost=0.08
electrical_cost=0.29 # electrical components between $0.13-0.45/W

module_size=2 #m^2 area

module_watt=300 #watts/module

pvEff=0.199 # efficiency (NREL Cost Benchmark)


panel_life=20
ONM=17.92 #$/kW/year
degredation=0.05/100
discount_rate=0.06
panel_height_strs=['01_feet','02_feet','03_feet','04_feet','05_feet','06_feet','07_feet','08_feet','09_feet','10_feet','11_feet','12_feet','13_feet','14_feet','15_feet']
row_spacings=np.arange(4,26,3)

panel_heights=[1,2,3,4,5,6,7,8,9,11,10,12,13,14,15]
field_area=10000 # assume an area of 10000 square meteres, 1 ha
field_area_ft=field_area*10.764
field_depth=np.sqrt(field_area_ft) # assuming square field 
bean_price_Cwt=35 # $/cwt
bean_price_kg=bean_price_Cwt/50.8

#%%
read_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\pickles'

yield_array=reg_pickle_open(read_dir+'\\just_yield_ir.p')
#%%

econ_array=pd.DataFrame(np.zeros((len(panel_heights)*len(row_spacings),11)))
econ_array.columns=(['Panel Height (meters)', 'Panel Spacing (meters)','Num rows', 'Num panels',
                     'Rad per panel (wh/panel)', 'Rad total(kWh)', 'Energy (kWh)', 'PV cost', 'PV rev','net $/ha','NPV'])
counter=0
data_folder_path=r"C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\panel_clearance_height"


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
    p_cost_no_rack=module_cost+inverter_cost+electrical_cost
    
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
def calc_econ_1_heightNrow(height,row_space,e_price=e_price,module_cost=module_cost,
                           inverter_cost=inverter_cost,electrical_cost=electrical_cost,pvEff=pvEff,
                           panel_life=panel_life,ONM=ONM,degredation=degredation,discount_rate=discount_rate,
                           bean_price_kg=bean_price_kg,rc=0,sf=0):
    
    econ_array=pd.DataFrame(np.zeros((1,11)))
    econ_array.columns=(['Panel Height (meters)', 'Panel Spacing (meters)','Num rows', 'Num panels',
                         'Rad per panel (wh/panel)', 'Rad total(kWh)', 'Energy (kWh)', 'PV cost', 'PV rev','net $/ha','NPV'])
    counter=0
    data_folder_path=r"C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\panel_clearance_height"


    
    if height <10:
        height_path=f'0{height}_feet'
    else:
        height_path=f'{height}_feet'
    (pvData, p_length, p_width, p_height, p_tilt,
         p_area, impact_x, impact_y, gSpacing)=read_in_PVdata(data_folder_path,height_path)
    # Units of ft except for p_area which is m^2
    module_per_panel=np.round(p_area)/module_size # num module
    racking_cost=racking_cost_per_watt(height) #$/watt
    if rc==1:
        racking_cost=racking_cost*sf
    p_cost_no_rack=module_cost+inverter_cost+electrical_cost
    PV_cost_per_watt=p_cost_no_rack+racking_cost # $/watt

    daily_mean_PV=get_PV(pvData) #wh/ft^2
    pvIntensity_panel=daily_mean_PV*p_area #wh/panel/year
    
    
    j=row_space
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
    
    econ_array.loc[counter,'Panel Height (meters)']=height*0.3048
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
    
    merged_array = (pd.merge(econ_array, yield_array,  how='left', 
                 left_on = ['Panel Height (meters)','Panel Spacing (meters)'],right_on=['Panel Height','Row Spacing'],))

    merged_array['NPV total']=merged_array['NPV']+merged_array['Yield (kg/ha)']*bean_price_kg

    return merged_array['NPV total'][0]
    
#%%

merged_array = (pd.merge(econ_array, yield_array,  how='left', 
             left_on = ['Panel Height (meters)','Panel Spacing (meters)'],right_on=['Panel Height','Row Spacing'],))

merged_array['NPV total']=merged_array['NPV']+merged_array['Yield (kg/ha)']*bean_price_kg
#%%
#%%
#aye=econ_array[econ_array['Panel Height (meters)']==2.1336]

#%%
height_base=6
row_space_base=10
h=6 # ft
h_m=1.8288000000000002 # m
base_npv=calc_econ_1_heightNrow(height_base,row_space_base)

#%%
sens_factor=np.arange(3,10,1)/6
sensitivity_array=pd.DataFrame(np.zeros((11,len(sens_factor))))
sensitivity_array.columns=['-50%','-33%','-17%','0%','+17%','+33%','+50%']
sensitivity_array.index=['Electricity price','Yield Price','Module costs','Inverter costs','Electrical component costs', 'PV efficiency', 'Racking costs','O&M','Panel degredation rate','Discount rate', 'Panel height']
#%%
npv_for_heights=[]
for i in sens_factor:
    aa=merged_array[(merged_array['Panel Height (meters)']==1.8288000000000002*i)&(merged_array['Panel Spacing (meters)']==10)]
    val=aa['NPV total'].values
    npv_for_heights.append(val)

npv_for_heights=pd.DataFrame(npv_for_heights)
val=merged_array[(merged_array['Panel Height (meters)']==1.524)&(merged_array['Panel Spacing (meters)']==10)]['NPV total'].values
npv_for_heights.iloc[2,0]=val[0]
val=merged_array[(merged_array['Panel Height (meters)']==2.1336)&(merged_array['Panel Spacing (meters)']==10)]['NPV total'].values
npv_for_heights.iloc[4,0]=val[0]

sensitivity_array.loc['Panel height',:]=np.transpose(npv_for_heights.values)


# electricity price 
for ii in range(0,len(sens_factor)):
    this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,e_price=e_price*sens_factor[ii])
    sensitivity_array.loc['Electricity price'].iloc[ii]=this_NPV
# module cost
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,module_cost=module_cost*sens_factor[ii])
        sensitivity_array.loc['Module costs'].iloc[ii]=this_NPV
# inverter cost
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,inverter_cost=inverter_cost*sens_factor[ii])
        sensitivity_array.loc['Inverter costs'].iloc[ii]=this_NPV                                    
# Electrical component costs
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,electrical_cost=electrical_cost*sens_factor[ii])
        sensitivity_array.loc['Electrical component costs'].iloc[ii]=this_NPV    
# PV efficiency 
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,pvEff=pvEff*sens_factor[ii])
        sensitivity_array.loc['PV efficiency'].iloc[ii]=this_NPV    
# Yield Price
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,bean_price_kg=bean_price_kg*sens_factor[ii])
        sensitivity_array.loc['Yield Price'].iloc[ii]=this_NPV    
# Operation and Maintenance 
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,ONM=ONM*sens_factor[ii])
        sensitivity_array.loc['O&M'].iloc[ii]=this_NPV  
# Panel degredation rate  
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,degredation=degredation*sens_factor[ii])
        sensitivity_array.loc['Panel degredation rate'].iloc[ii]=this_NPV 
# Discount rate
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,discount_rate=discount_rate*sens_factor[ii])
        sensitivity_array.loc['Discount rate'].iloc[ii]=this_NPV 
# Racking cost
    for ii in range(0,len(sens_factor)):
        this_NPV=calc_econ_1_heightNrow(height_base,row_space_base,rc=1,sf=sens_factor[ii])
        sensitivity_array.loc['Racking costs'].iloc[ii]=this_NPV 
#%%
for_plot=sensitivity_array.transpose()
label=for_plot.columns
plt.figure(dpi=1200)
plt.plot(for_plot)
plt.ylim(-200000,200000)
plt.xlabel('Percent Deviation from Base Input Value', fontsize=15, labelpad=10)
plt.ylabel ('Net Present Value After 20 Years ($)', fontsize=15, labelpad=10)
plt.title ('Sensitivity Analysis \n\n Base Height = 1.83 m (6 ft), Base Spacing = 10 m (32.8 ft)', fontsize=15,pad=20)
plt.legend(label,framealpha=1)
plt.show()

#%%
#  Calculate relative sensitivity
labels = ['Base Value', 'Perturbed Value', 'Response Value ($)', 'Relative Sensitivity']
SI_variables=['Electricity price ($/W)','Yield Price ($/kg)','Module costs ($/W)','Inverter costs ($/W)','Electrical component costs ($/W)', 'PV efficiency (%)', 'Racking costs ($/W)','O&M ($/kW/year)','Panel degredation rate (%/year)','Discount rate (%)', 'Panel height (m)']
racking_cost_base=racking_cost_per_watt(height_base)
r_sensitivity=pd.DataFrame(np.zeros((len(SI_variables),4)))
r_sensitivity.index=SI_variables
r_sensitivity.columns=labels
r_sensitivity['Base Value']=[e_price,bean_price_kg,module_cost,inverter_cost,electrical_cost,pvEff,racking_cost_base,ONM,degredation,discount_rate,height_base*0.3048]
r_sensitivity['Perturbed Value']=r_sensitivity['Base Value']*1.5
r_sensitivity['Response Value ($)']=sensitivity_array['+50%'].values
r_sensitivity['Relative Sensitivity']=(r_sensitivity['Response Value ($)']-base_npv)/base_npv/0.5

#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
display(r_sensitivity) 