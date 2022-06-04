# -*- coding: utf-8 -*-
"""
2022-06-03 

This script is the first step of the python portion of the 
AV-EAS framework it takes as inputs the revit solar radiation
csv files and produces a pickle file representing the spatial
radiation for a single row for all panel heights.

This pickle file will then be utilized in the 
calc_spatial_hourly_rad_for_DSSAT.py script


@author: proctork@oregonstate.edu
"""
#Read in files and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
import pvlib
import timezonefinder
import pytz
import glob
import os
import copy

def reg_pickle_open(filename):
    """
    open a pickle file

    Parameters
    ----------
    filename : str
        file name including data type (.p)

    Returns
    -------
    pFile : 
        Whatever was stored in the pickle file

    """
    with open(filename, 'rb') as f:
        pFile=pickle.load(f)
    
    return pFile

def reg_pickle_dump (save_dir,filename,data):
    """
    create a pickle file

    Parameters
    ----------
    save_dir : str
        save directory
    filename : str
        filename without data type
    data : various
        data to be pickled

    Returns
    -------
    None.

    """
  # pickle files
    with open(save_dir+filename+'.p','wb') as f:
        pickle.dump(data,f)
#%%  Read in input CSV files 


# The run time of the Dynamo solar radiation analysis scales
# somewhat exponentially with the number of days required
# Thus, it is beneficial to split the outputs over multiple 
# csv files. The following lines open the various files and 
# aggregate them into a single large list.
# The data_param file is consistent as the parameters do not 
# change with time

# file path of temporal radiation data and model parameters
# The following line assume the following naming convention:

def get_parameters(data_folder_path,height_path):
    f= glob.glob(data_folder_path+'\\'+height_path+'\\model_param*.csv')
    data_param=pd.read_csv(f[0]) 
    params=data_param
    p_length=params['Panel length (feet)'][0]
    p_width=params['Panel width (feet)'][0]
    p_tilt=params['Panel tilt (degree)'][0]
    p_area=params['PV panel area (ft^2)'][0]
    impact_x=params['Panel rad impact X(feet)'][0]
    impact_y=params['Panel rad impact Y(feet)'][0]
    gSpacing=params["Grid spacing(feet)"][0]
    origin_y=params["Origin_y(feet)"][0]
    lat=params["Lat"][0]
    lon=params["Lon"][0]
    return p_length, p_width, p_tilt, p_area, impact_x, impact_y, gSpacing, origin_y, lat, lon
#%%
def read_in_data(data_folder_path,height_path):
    
    data_list_c=[]
    
    for file in glob.glob(data_folder_path+'\\'+height_path+'\\crop*.csv'):
        data_c=pd.read_csv(file,parse_dates=['Start Time'])
        data_list_c.append(data_c)

    for file in glob.glob(data_folder_path+'\\'+height_path+'\\model_param*.csv'):
        data_param=pd.read_csv(file)

    data_OG=pd.concat(data_list_c,axis=0,ignore_index=True) # original, column titles describe units


    params=data_param
    p_length=params['Panel length (feet)'][0]
    p_width=params['Panel width (feet)'][0]
    p_height=params['Panel height (feet)'][0]
    p_tilt=params['Panel tilt (degree)'][0]
    p_area=params['PV panel area (ft^2)'][0]
    impact_x=params['Panel rad impact X(feet)'][0]
    impact_y=params['Panel rad impact Y(feet)'][0]
    gSpacing=params["Grid spacing(feet)"][0]
    origin_y=params["Origin_y(feet)"][0]
    lat=params["Lat"][0]
    lon=params["Lon"][0]

    # Create a data file with only relevant data and shorter column names 
    Data=data_OG[["Start Time", "Stop Time","X coordinate (feet)","Y coordinate (feet)","Cumulative Solar radiation (Kwh/m^2)"]].copy()
    Data.columns='t0','tf','x','y','solar'
    Data['x']=data_OG['X coordinate (feet)']
    #Data['solar']=Data['solar']/3600

     
    return Data, p_length, p_width, p_height, p_tilt, p_area, impact_x, impact_y, gSpacing
#%%
# This section creates a nested dictionary which holds the solar radiation data
#for each day and each hour of each day

def collect_all_rad_data(data):
    """
    Format radiation data into a single dictionary which is sorted by day of year
    and hour of day

    Parameters
    ----------
    data : DataFrame
        Hourly Solar radiation incident on the ground surface 
        at various points defined by their x,y coordinates.
        The grid size is described by the gSpacing variable (feet).
        Solar radiation is calculated hourly for the period 4:30-21:30.
        This array is imported from revit.

    Returns
    -------
    allRadData : dict
        Dictionary of all radiation data.
        Keys are day of year (unique_dates) and hour of day (unique_hours_
        For each hour within each day there is an array as defined by the 
        "data" input parameter.                                                    )
    unique_dates : DataFrame
        Unique dates within period of interest
    unique_hours : DataFrame
        Unique hours within period of interest, based on starting time (t0)

    """
    # seperate datetime to date and time and add as columns 
    time_info=data['t0']
    
   
    
    dates=time_info.apply(datetime.datetime.strftime,args=('%Y/%m/%d',))
    times=time_info.apply(datetime.datetime.strftime,args=('%H:%M',))
     #these lines take a really long time can I make them quicker?
     
    unique_dates=dates.unique()
    unique_dates=np.sort(unique_dates)
    unique_hours=times.unique()
    unique_hours=unique_hours[:-1] # remove final unique hour because it only
                                   # serves as an end time not a start time (t0)
    
    data['Date']=dates
    data['hour_0']=times 
    
    # These lines create a nested list which seperates data by day and hours within day
    by_day=[]
    for i in unique_dates:
        this_day=data[data['Date']==i]
        by_day.append(this_day)
   
    
    data_nested_list=[]
    for i in range(0,len(unique_dates)):
        daily_data=[]
        for j in unique_hours:
            this_hour=by_day[i][by_day[i]['hour_0']==j]
            daily_data.append(this_hour)
        
        data_nested_list.append(daily_data)
        
    # These lines take the nested list and turn it into a nested dictionary 
    # This process is intended to improve clarity and accessibility of data
    # as it allows for access the data based on day and hour of day vs just using the index
    # The data can be accessed with the following syntax
    # allRadData['~Day~']['~Hour of day']
    # a potential list of days and hours of day can be accessed by viewing
    #'unique_dates' & 'unique_hours', respectively
    
   
    temp_list=[]
    for i in data_nested_list:
        day_dict=dict(zip(unique_hours,i))
        temp_list.append(day_dict)
    allRadData=dict(zip(unique_dates,temp_list))

    return allRadData, unique_dates, unique_hours



#%% This section calculates the minimum row spacing as describe here:
#   https://www.cedgreentech.com/article/determining-module-inter-row-spacing

# First we need to determine the solar elevation angle and 
# solar azimuth correction angle for the site location on the 
# winter solstice (December 21st). We will find these values at 
# 9 am on the solstice.
def calculate_area_based_on_row_spacing(row_spacing):
# The following lines determine the timezone corresponding to the site location
    
    
    
    tf=timezonefinder.TimezoneFinder()
    timezone_str=tf.certain_timezone_at(lat=lat, lng=lon)
    tz=pytz.timezone(timezone_str)
    
    # Create datetime variable for the solstice
    solstice_time=datetime.datetime(year=2020,month=12,day=21,hour=9,tzinfo=tz)
    
    # Use pvlib to calculate solar elevation and azimuth
    
    solstice_sol_pos=pvlib.solarposition.spa_python(solstice_time,
                                                    lat,
                                                    lon,
                                                    alt)
    
    solar_elev_angle=np.deg2rad(solstice_sol_pos['elevation'][0])
    az_correction_angle=np.deg2rad(180-solstice_sol_pos['azimuth'][0])
    tilt_rad=np.deg2rad(p_tilt)
    
    delta_h=np.sin(tilt_rad)*p_width #height difference between top and bottom of panel
    
    # Minimum distance from back of one panel to start of the next
    module_row_spacing=delta_h/np.tan(solar_elev_angle) 
    
    # minimum spacing accounting for azimuth
    minimum_row_spacing=module_row_spacing*np.cos(az_correction_angle) 
    
    # Minimum distance from trailing edge of one row to trailing edge of the next row
    minimum_row_width=minimum_row_spacing+np.cos(tilt_rad)*p_width

    # Ensure no interrow shading occurs, this model is NOT designed to account
    #for shading of panels by other panels and assumes all panels are unshaded
    #if row_spacing<minimum_row_width: 
    #    row_spacing=minimum_row_width
    
    # The following lines are used to create a DataFrame large enough to account 
    # for the full solar configuration (as defined in the inputs section)
    # Each value with in the ray will represent the radiation for a single grid cell
    # with size defined by the gSpacing variable (feet)
    origin_x=impact_x/2
    off_set=p_length
    
    num_x=len(np.unique(example_hour['x'])) # number of individual x points for single panel
    num_y=len(np.unique(example_hour['y'])) # number of individual y points for single panel
    
    total_y=impact_y + row_spacing*(num_rows) # total area used for analysis
    total_x=impact_x+off_set*(num_p_per_row) # total area used for analysis
    
    x_row=int(np.ceil(total_x/gSpacing)) #number of individual x points for full area
    y_col=int(np.ceil(total_y/gSpacing)) #number of individual y points for full area

    return origin_x, num_x, num_y,x_row,y_col,num_y, off_set

#%%
def hourly_rad_2D_unit(hourly_data):
    """
    Create a radiation array that represents the impacted area from a single panel
    at a particular hour
    This is the impact array that the law of superpositon will be applied to

    Parameters
    ----------
    hourly_data : DataFrame
        Radiation data for a particular hour during a particular day.
        Taken from the allRadData Dictionary

    Returns
    -------
    rad_array : DataFrame
       Impacted area from a single panel at a particular hour.
       Contains the difference between the actual radiation at a given point
       and the maximum radiation at that point. Essential captures the impact of 
       the solar panel array
       Grid size is based on gSpacing variable
       
    control_rad : Float
        Maximum radiation for a particular hour during a particular day.
        This is the radiation that would be expected for conditions where
        no solar array was present

    """
    
    control_rad=max(hourly_data['solar'])
    
    #The sunlight deficit is calcluated by subtracting the actual radiation at 
    # every location from the maximum radaition (aka control radiation)
    deficit=control_rad-hourly_data['solar'] 


    rad_array=np.nan*np.empty((num_y,num_x))
    rad_array[(np.round(hourly_data['y']/gSpacing)).astype(int),(np.round(hourly_data['x']/gSpacing)).astype(int)]=deficit
    # y values are placed as rows and x values as columns for intuitive vizualation
    #print(np.shape(rad_array))
    return rad_array,control_rad
#%%
def calc_impact_area(rad_array,point_x,point_y):
    """
    Calculate the impacted area of a single panel at a particular
    location.

    Parameters
    ----------
    rad_array : DataFrame
        Impacted area from a single panel at a particular hour.
        Contains the difference between the actual radiation at a given point
        and the maximum radiation at that point. Essential captures the impact of 
        the solar panel array.
        Grid size is based on gSpacing variable
    point_x : int
        x-coordinate of the panel
    point_y : int
        y-coordinate of the panel

    Returns
    -------
    impact_area_T : DataFrame
        Impact area of a panel at a particular location 

    """
    
    impact_area_T=pd.DataFrame(np.zeros((y_col,x_row))) # Impact area for this panel
    strt_x=int(np.round((point_x-impact_x/2)/gSpacing))
    stp_x=int(np.round((point_x+impact_x/2)/gSpacing)+1)
    # The start and end point of the impact area are dependent on the 
    # panel location and the impact width in the x direction
    # with the panel sitting in directly in the center
    # The location values is divided by the gspacing to convert from
    # units of (ft) to units of (grid cell)
    strt_y=int(np.ceil((point_y-origin_y)/gSpacing))
    stp_y=int(np.ceil((point_y-(origin_y)+impact_y)/gSpacing))
    # The panel does not sit in the center of the impact area in the
    # y direction. This is because (in the northern hemisphere) the panel
    # will have a larger impact on the ground due south. The origin_y factor
    # is used to adjust for this. This factor is set in the dynamo script and 
    # should not be changed unless changes have been made to the dynamo script
    
    
    impact_area_T.iloc[strt_y:stp_y,strt_x:stp_x]=rad_array
    # The rad array is placed in the impact area.
    return impact_area_T
#%%
def hourly_rad_2D(rad_array,control_rad):
    """
    Calculate the total impacted area for an array of PV panels
    on a particular day at a particualr panel.The output has units of wh/ft^2
    Although not direct inputs to this function, this function depends upon the panel configuration 
    ( # of rows & panels per row) specified earlier in this script.
    

    Parameters
    ----------
    rad_array : DataFrame
        Impacted area from a single panel at a particular hour.
        Contains the difference between the actual radiation at a given point
        and the maximum radiation at that point. Essential captures the impact of 
        the solar panel array
        Grid size is based on gSpacing variable
    control_rad : Float
        Maximum radiation for a particular hour during a particular day.
        This is the radiation that would be expected for conditions where
        no solar array was present

    Returns
    -------
    rad_array_T : TYPE
        Total impacted area for an array of PV panels at a particular hour
        on a particular day. While the rad_array parameter represented the 
        difference between the maximum radiation and the shaded aree, the 
        rad_array_T parmeter represents the absolute energy intensity available
        at the crop level with units of wh/ft^2
        The 

    """
    
    impact_area_list=[]
    for i in range(num_rows):
        for j in range(num_p_per_row):
            impact_area_indiv=calc_impact_area(rad_array,origin_x+off_set*j,origin_y+row_spacing*i)
            impact_area_list.append(impact_area_indiv)
    rad_array_T=control_rad-sum(impact_area_list)
    # Subtracting the impacted area from the control radiation converts
    # from the difference in radiation to the radiation available at the 
    # ground surface
    
    # These lines can be used to visualize the impacted area at 
    # every hour, this will create many plots and can potentially cause
    # the code to crash
    
    #plt.pcolormesh(rad_array_T,cmap='RdYlGn_r')
    #plt.show()
    
    return rad_array_T          

#%%
def get_rad_data_spatial(allRadData):
    """
    Create a dictionary which holds the hourly crop level solar radiation 
    for the study period. 
    
    This function utilizes the three previously defined functions. 

    Parameters
    ----------
    allRadData : dict
        Dictionary of all radiation data.
        Keys are day of year (unique_dates) and hour of day (unique_hours)
        For each hour within each day there is an array as defined by the 
        "data" input parameter. 

    Returns
    -------
    radSpatial : dict
        Dictionary of spatial solar radiation.
        Keys are day of year (unique_dates) and hour of day (unique_hours)
        While the allRadData holds the solar radiation data for a single panel
        as x, y coordinates and solar radiation values, the radSpatial array,
        holds a list which contains solar radiation for the entire PV array
        x and y coordinates are represented by the rows and columns with the
        distance per grid cell being defined by the gSpacing parameter
    """
    
    radSpatial=copy.deepcopy(allRadData)
    for i in unique_dates:
        for j in unique_hours:
            rad_array,control_rad=hourly_rad_2D_unit(allRadData[i][j])
            # get defecit radiation and maximum radiation for each hour
            radArray_T=hourly_rad_2D(rad_array,control_rad)
            #create spatial representation of radiation in the array
            
            radArray_T[radArray_T<0]=0
            
            radSpatial[i][j]=radArray_T
            # save for the corresponding time period
            
            
    return radSpatial

#%%
def calc_rad_spatial(data_folder_path,height_path):
    """
    This is the function which calculates radSpatial
    for a particular height 

    Parameters
    ----------
    data_folder_path : TYPE
        DESCRIPTION.
    height_path : TYPE
        DESCRIPTION.

    Returns
    -------
    radSpatial : TYPE
        DESCRIPTION.

    """
    global p_tilt, p_width, p_length, impact_x, example_hour
    global num_p_per_row, impact_y, gSpacing
    global unique_dates, unique_hours
    global num_y, num_x
    global y_col, x_row
    global origin_x, off_set
    
    t0=datetime.datetime.now()
    t1=datetime.datetime.now()
    
    (Data, p_length, p_width, p_height, p_tilt,
         p_area, impact_x, impact_y, gSpacing)=read_in_data(data_folder_path,height_path)
    
 
    data=Data

    
    dates=data['t0'].apply(datetime.datetime.strftime,args=('%Y/%m/%d',))
    print('read in data and collected dates ')
    print(datetime.datetime.now()-t1)
    
    # #Create radiation dicitonary 
    t1=datetime.datetime.now()

    allRadData, unique_dates, unique_hours=collect_all_rad_data(Data)
    print ( 'created rad dictionary' )
    example_hour=allRadData[unique_dates[0]]['11:30']
    print(datetime.datetime.now()-t1)

    
    t1=datetime.datetime.now() 
    origin_x, num_x, num_y,x_row,y_col,num_y, off_set=calculate_area_based_on_row_spacing(row_spacing)

    radSpatial=get_rad_data_spatial(allRadData) # actually run the function
    print(datetime.datetime.now()-t1) 

    print ('total time = ')
    print(datetime.datetime.now()-t0)
    
    return radSpatial 
#%% Inputs

# This is the only section where a user may want to edit the 
# code

# these lines open and get the basic parameters, these parameters should not be
# changing with height
data_folder_path=r"C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\panel_clearance_height"
(p_length, p_width, p_tilt, p_area,
  impact_x, impact_y, gSpacing, origin_y,
  lat, lon)=get_parameters(data_folder_path,'15_feet')

origin_y=origin_y # starting point for first panel in the y axis

field_area=10000 # assume an area of 500 square meteres, ill multiply by 10 later to get 1 ha
field_area_ft=field_area*10.764
field_depth=np.sqrt(field_area_ft) # assuming square field 

num_rows=1
row_spacing=0


# Weather station latitude and altitude
alt= 42.672  # altitude aurora oregon (m)
lat=lat
lon=lon
num_p_per_row=int(np.round(field_depth/p_length)) # number of panels per row

param_array=pd.DataFrame([p_length, p_width, p_tilt,
              impact_x, impact_y, gSpacing, origin_y,
              lat, lon,num_rows,num_p_per_row])
param_array.index=['p_length', 'p_width', 'p_tilt',
              'impact_x', 'impact_y', 'gSpacing', 'origin_y',
              'lat', 'lon','num_rows','num_p_per_row']

tf=timezonefinder.TimezoneFinder()
timezone_str=tf.certain_timezone_at(lat=lat, lng=lon)
tz=pytz.timezone(timezone_str)
#%%
save_dir=r'C:\Users\proctork\Documents\Graduate_School\Research\Thesis\Radiation_modeling\Code\Rad_out\panel_clearance_height'
radSpatial_save_dir=save_dir+'\\radSpatial_pickles'
param_save_dir=save_dir+'\\parameter_pickles'
panel_height_strs=['01_feet','02_feet','03_feet','04_feet','05_feet','06_feet','07_feet','08_feet','09_feet','10_feet','11_feet','12_feet','13_feet','14_feet','15_feet']

#%%
# run for all heights 
for i in panel_height_strs:
    print(i)
    this_radS=calc_rad_spatial(data_folder_path,i)
    reg_pickle_dump(radSpatial_save_dir,'\\'+i,this_radS)

#%%

reg_pickle_dump (param_save_dir,'\\Unique_dates',unique_dates)
reg_pickle_dump (param_save_dir,'\\Unique_hours',unique_hours)
reg_pickle_dump (param_save_dir,'\\param_array',param_array)
#%%
# run for a single file to check things 
"""
this_radS=calc_rad_spatial(data_folder_path,height_path="15_feet")
reg_pickle_dump(radSpatial_save_dir,'\\'+'15_feet',this_radS)
"""
#example_plot=this_radS['2020/07/07']['12:30'].iloc[5:,:]
#plt.pcolormesh(example_plot,cmap='RdYlGn_r')
#plt.colorbar()
#plt.show()
