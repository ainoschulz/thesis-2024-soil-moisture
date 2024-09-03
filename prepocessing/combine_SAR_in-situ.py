"""
This script is to combine the in-situ data with the SAR data.
Author: AS
"""
from datetime import datetime, timedelta
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
import pandas as pd
import rasterio
import seaborn as sns
from pyresample import image, geometry, kd_tree
import pyresample

import warnings
warnings.filterwarnings("ignore")


# location of datasets
path_start = #set to your own directory
#-------------------------------------------------------------------------------
# Data paths
# SAR raw data path
sar_path = path_start + 'SAR_data/data_2016'

# SAR-stack path
stack_path = path_start + 'SAR_data/processed/kilpisjarvi_S1_S2_final.nc'

# Despeckled data path
despeckled_path = path_start + 'SAR_data/despeckled'

# DEM file path
dem_path = path_start + 'DEM/dem_10m.tif'

# In-situ data path and read in field data
path_insitu = path_start + 'Kilpisjarvi_field_data_2016.csv'
field_data = pd.read_csv(path_insitu)
field_data.drop(columns=['saittis'], inplace=True)


# Weather data
weather = pd.read_csv(path_start + 'temp.csv')
#------------------------------------
#functions

def get_lat_lon_from_sar(path):
    """
    This function searches for the latitude and longitude from the SAR stack.

    :param path: The file path to the SAR stack.
    :return: Latitude values and longitude values as type (numpy.ma.core.MaskedArray).
    """

    # latitude from SAR data
    file2read = nc4.Dataset(path,'r')
    lat = file2read.variables['latitude'][:]*1

    # longitude from SAR data
    file2read = nc4.Dataset(path,'r')
    lon = file2read.variables['longitude'][:]*1

    return [lat,lon]

def find_closest_row_column(df, lat_s, lon_s):
    """
    Finds the closest row and column indices in a dataframe to the given
    latitude and longitude coordinates.

    :param df: The dataframe containing latitude and longitude information.
    :param lat_s: Latitude coordinates to find the closest point.
    :param lon_s: Longitude coordinates to find the closest point.
    :return: The input dataframe with additional 'rows' and 'cols' indicating
    the closest row and column.
    """
    print(lat_s.shape)
    row = np.full([len(df)], np.nan, dtype=int)
    column = np.full([len(df)], np.nan, dtype=int)

    for i in range(0, len(df)):
        # lat lon of the in-situ measurement
        lat_i = df.iloc[i]['lat']
        lon_i = df.iloc[i]['lon']

        # find closest points
        difference = np.sqrt((lat_i-lat_s)**2 + (lon_i-lon_s)**2)
        idx = np.unravel_index(np.nanargmin(difference), difference.shape)
        row[i], column[i] = idx[0], idx[1]

    df['rows'] = row
    df['cols'] = column

    return df

def find_closest_date(df, sar_path):
    """
    Finds the closest SAR date for each in-situ date in the dataframe.

    :param df: Dataframe containing in-situ timestamp columns.
    :param sar_path: Path to the directory containing SAR data folders.
    :return: Updated dataframe with additional columns indicating the closest SAR date.
    """
    # change in-situ datetime columns to datetime
    df[['timestamp06', 'timestamp07', 'timestamp08']] = df[['timestamp06', 'timestamp07', 'timestamp08']].apply(pd.to_datetime)

    # construct datetimes for sar data
    dates = []
    for sar_name in os.listdir(sar_path):
        parts = sar_name.split('_')
        dates.append(parts[4])

    sar_dates = [datetime.strptime(date, '%Y%m%dT%H%M%S') for date in dates]

    # find the closest sar date for each in-situ date
    for idx, row in df.iterrows():
        for col in ['timestamp06', 'timestamp07', 'timestamp08']:
            closest_sar_date = None
            closest_diff = timedelta.max
            for sar_date in sar_dates:
                diff = abs(row[col] - sar_date)
                if diff < closest_diff:
                    closest_sar_date = sar_date
                    closest_diff = diff
            df.at[idx, f'closest_sar_date{col[-2:]}'] = closest_sar_date

    return df

def format_date_and_capitalize_month(date_series):
    """
    This function was created when I realized how much of a difference
    a capital letter can do and how annoying different datetime formats are.
    It most likely works/is useful just in this special case.
    :param date_series: A pandas series containing date values to be formatted.
    :return: A list of formatted dated with capitalized month abbreviations.
    """
    def capitalize_month(month_str):
        return month_str[:1].upper() + month_str[1:]
    return pd.to_datetime(date_series).dt.strftime('%d%b%Y').apply(capitalize_month).unique().tolist()

def add_despeckled_to_insitu(despeckled_path, df):
    """
    The function adds despeckled SAR data to the in-situ dataframe.

    :param despeckled_path: Path to the directory containing despeckled SAR data files.
    :param df: Dataframe containing in-situ data.
    :return: Updated dataframe with added columns for VH and VV values.
    """
    # if resampling add this to function --> area_def, lons2, lats2, xs, ys
    # change the datetime format to match with despeckled data (note formatting not saved to df!)
    # and get all unique dates for each month in a list
    unique_dates = np.unique(np.concatenate([
        format_date_and_capitalize_month(df['closest_sar_date06']),
        format_date_and_capitalize_month(df['closest_sar_date07']),
        format_date_and_capitalize_month(df['closest_sar_date08'])
    ])).tolist()

    # handle on date at the time
    for date in unique_dates:
        # find the despeckled file for the current date
        vh_file = os.path.join(despeckled_path, f"*_{date}_*_VH.npy")
        vv_file = os.path.join(despeckled_path, f"*_{date}_*_VV.npy")

        vh_files = glob.glob(vh_file)
        vv_files = glob.glob(vv_file)

        # process both VH and VV files
        for vh_file, vv_file in zip(vh_files, vv_files):
            vh_data = np.load(vh_file)
            vv_data = np.load(vv_file)

            # if resampling, uncomment these
            #vh_data = resample_data(vh_data, area_def, lons2, lats2, xs, ys)
            #vv_data = resample_data(vv_data, area_def, lons2, lats2, xs, ys)

            # extract month from the file name
            file_month = datetime.strptime(date, "%d%b%Y").strftime("%B")

            # extract values from npy files and create new columns
            for index, row in df.iterrows():
                vh_value = vh_data[row['rows'], row['cols']]
                vv_value = vv_data[row['rows'], row['cols']]

                # add the VH and VV values to the df in right column
                df.loc[index, f'VH_{file_month}'] = vh_value
                df.loc[index, f'VV_{file_month}'] = vv_value

    return df

def add_ndvi_lai_from_sar(path, df):
    """
    The function adds NDVI and LAI values to the in-situ dataframe.

    :param path: Path to the SAR stack.
    :param df: Dataframe with the in-situ data.
    :return: Updated dataframe with NDVI and LAI values.
    """
    # if resampling add this to function --> area_def, lons2, lats2, xs, ys
    # NDVI
    file2read = nc4.Dataset(path, 'r')
    ndvi = file2read.variables['NDVI'][:] * 1

    # LAI
    file2read = nc4.Dataset(path, 'r')
    lai = file2read.variables['LAI'][:] * 1

    print(ndvi.shape)

    # if resampling uncomment these
    #ndvi = resample_data(ndvi, area_def, lons2, lats2, xs, ys)
    #lai = resample_data(lai, area_def, lons2, lats2, xs, ys)


    # add values to df based on rows and columns information
    for idx, row in df.iterrows():
        # extract column indices
        row_idx, col_idx = row['rows'], row['cols']

        # assign ndvi and lai values to specific rows
        df.at[idx, 'NDVI'] = ndvi[row_idx, col_idx]
        df.at[idx, 'LAI'] = lai[row_idx, col_idx]

    return df

def calculate_angels(dem_path, stack_path):
    """
    The function calculates various angles including incidence angles, slope, aspect,
    cosine of slope angles relative to the incidence angles, and sine of filament
    orientation angles relative to aspect angles.

    :param dem_path: Path to DEM file.
    :param stack_path: Path to SAR stack file.
    :return: A list containing the calculated angles.
    """
    # if resampling add this to function --> area_def, lons2, lats2, xs, ys
    # read in incidence angle data from SAR stack
    with nc4.Dataset(stack_path, 'r') as file2read:
        ing_1m = file2read.variables['incidenceAngleFromEllipsoid_IW1m'][:]*1
        ing_2m = file2read.variables['incidenceAngleFromEllipsoid_IW2m'][:]*1
        ing_2e = file2read.variables['incidenceAngleFromEllipsoid_IW2e'][:]*1
        ing_3e = file2read.variables['incidenceAngleFromEllipsoid_IW3e'][:]*1

    # read in dem and cut it to be the same shape as the SAR stack
    with rasterio.open(dem_path) as src:
        dem = src.read(1, out_shape=(ing_1m.shape[0], ing_1m.shape[1]))

    # if resampling uncomment these
    #ing_1m = resample_data(ing_1m, area_def, lons2, lats2, xs, ys)
    #ing_2m = resample_data(ing_2m, area_def, lons2, lats2, xs, ys)
    #ing_2e = resample_data(ing_2e, area_def, lons2, lats2, xs, ys)
    #ing_3e = resample_data(ing_3e, area_def, lons2, lats2, xs, ys)
    #dem = resample_data(dem, area_def, lons2, lats2, xs, ys)

    # calculate cosines for IW data
    cos_ing_1m = np.cos(ing_1m*(np.pi/180))
    cos_ing_2m = np.cos(ing_2m*(np.pi/180))
    cos_ing_2e = np.cos(ing_2e*(np.pi/180))
    cos_ing_3e = np.cos(ing_3e*(np.pi/180))
    cos_ing_data = [cos_ing_1m, cos_ing_2m, cos_ing_2e, cos_ing_3e]

    # calculate slope using numpy gradient
    slope_x, slope_y = np.gradient(dem)
    slope_radians = np.arctan(np.sqrt(slope_x**2 + slope_y**2))

    # convert slope to degrees
    slope_degrees = np.degrees(slope_radians)

    # calculate aspect using numpy arctan2
    aspect_radians = np.arctan2(-slope_y, slope_x)
    aspect_degrees = np.degrees(aspect_radians) % 360

    # calculate cosine of slope angles relative to incidence angles
    cslo_1m = np.cos(slope_radians - ing_1m *( np.pi / 180))
    cslo_2m = np.cos(slope_radians - ing_2m * (np.pi / 180))
    cslo_2e = np.cos(slope_radians - ing_2e * (np.pi / 180))
    cslo_3e = np.cos(slope_radians - ing_3e * (np.pi / 180))
    cslo_data = [cslo_1m, cslo_2m, cslo_2e, cslo_3e]

    # default values for filament orientation angles
    filookm = 277.6684131363174
    filooke = 87.3315868636826

    # calculate sine of filament orientation angles relative to aspect angles
    aspm = np.sin(-(aspect_degrees - filookm) * (np.pi / 180))
    aspe = np.sin(-(aspect_degrees - filooke) * (np.pi / 180))
    asp_data = [aspm, aspe]

    # return all calculated angles
    return [dem, cos_ing_data, slope_degrees, aspect_degrees, cslo_data, asp_data]

def add_angles_to_dataframe(df, dem_path, stack_path):
    """
    The function adds angle values calculated from DEM and SAR stack to a dataframe.

    :param df: The dataframe where the angle values will be added.
    :param dem_path: The file path to the DEM.
    :param stack_path: The file path to the SAR stack.
    :return: The dataframe with added angle values.
    """
    # if resampling add this to function --> area_def, lons2, lats2, xs, ys
    # calculate angles, as a list
    angles = calculate_angels(dem_path, stack_path)
    #angles = calculate_angels(dem_path, stack_path,area_def, lons2, lats2, xs, ys)

    # Define the dates of the swath overpass combinations
    dates_1m = ['2016-06-08', '2016-08-19']
    date_2m = '2016-07-09'
    date_3e = '2016-06-12'
    date_2e = '2016-08-18'

    # Check if 'cos_ing' and 'cslo' columns exist, if not create them
    if 'cos_ing' not in df.columns:
        df['cos_ing'] = 0.0
    if 'cslo' not in df.columns:
        df['cslo'] = 0.0

    # add values to df based on rows and columns information
    for idx, row in df.iterrows():
        # extract column indices
        row_idx, col_idx = row['rows'], row['cols']

        # extract all the values that can be extracted
        df.at[idx, 'dem'] = angles[0][row_idx, col_idx]

        df.at[idx, 'cos_ing_1m'] = angles[1][0][row_idx, col_idx]
        df.at[idx, 'cos_ing_2m'] = angles[1][1][row_idx, col_idx]
        df.at[idx, 'cos_ing_2e'] = angles[1][2][row_idx, col_idx]
        df.at[idx, 'cos_ing_3e'] = angles[1][3][row_idx, col_idx]

        df.at[idx, 'slope_deg'] = angles[2][row_idx, col_idx]
        df.at[idx, 'aspect_deg'] = angles[3][row_idx, col_idx]

        df.at[idx, 'cslo_1m'] = angles[4][0][row_idx, col_idx]
        df.at[idx, 'cslo_2m'] = angles[4][1][row_idx, col_idx]
        df.at[idx, 'cslo_2e'] = angles[4][2][row_idx, col_idx]
        df.at[idx, 'cslo_3e'] = angles[4][3][row_idx, col_idx]

        df.at[idx, 'aspm'] = angles[5][0][row_idx, col_idx]
        df.at[idx, 'aspe'] = angles[5][1][row_idx, col_idx]

        # Extract the date part from the timestamp
        timestamp_date = row['closest_sar_date'].date()

         # Add cos_ing and cslo values based on specific dates
        if timestamp_date in [pd.Timestamp(d).date() for d in dates_1m]:
            df.at[idx, 'cos_ing'] = df.at[idx, 'cos_ing'] + df.at[idx, 'cos_ing_1m']
            df.at[idx, 'cslo'] = df.at[idx, 'cslo'] + df.at[idx, 'cslo_1m']
        elif timestamp_date == pd.Timestamp(date_2m).date():
            df.at[idx, 'cos_ing'] = df.at[idx, 'cos_ing'] + df.at[idx, 'cos_ing_2m']
            df.at[idx, 'cslo'] = df.at[idx, 'cslo'] + df.at[idx, 'cslo_2m']
        elif timestamp_date == pd.Timestamp(date_3e).date():
            df.at[idx, 'cos_ing'] = df.at[idx, 'cos_ing'] + df.at[idx, 'cos_ing_3e']
            df.at[idx, 'cslo'] = df.at[idx, 'cslo'] + df.at[idx, 'cslo_3e']
        elif timestamp_date == pd.Timestamp(date_2e).date():
            df.at[idx, 'cos_ing'] = df.at[idx, 'cos_ing'] + df.at[idx, 'cos_ing_2e']
            df.at[idx, 'cslo'] = df.at[idx, 'cslo'] + df.at[idx, 'cslo_2e']


    return df

def add_temp_rain(df, temp_df):
    # Ensure the date columns are in datetime format
    df['closest_sar_date'] = pd.to_datetime(df['closest_sar_date'])
    temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])

    # Create new columns for rain and temp if they don't exist
    if 'rain' not in df.columns:
        df['rain'] = 0.0
    if 'temp' not in df.columns:
        df['temp'] = 0.0

    # Iterate over each row in df
    for idx, row in df.iterrows():
        # Extract the date from the row
        timestamp_date = row['closest_sar_date'].date()

        # Find matching date in temp_df
        matching_row = temp_df[temp_df['date'].dt.date == timestamp_date]

        # If a match is found, update the rain and temp columns
        if not matching_row.empty:
            df.at[idx, 'rain'] = matching_row['rain'].values[0]
            df.at[idx, 'temp'] = matching_row['temp'].values[0]

    return df


def create_dataframe_based_on_sar_pixels(df):
    """
    The function averages all the rows where df['rows'] and df['cols'] are exactly
    the same (i.e. the SAR pixels) and saves it to a new dataframe.

    :param df: The dataframe with the combined data.
    :return: A new dataframe where data is by pixel (10m x 10m).
    """
    # group by rows and cols, and calculate the mean for each group
    grouped_df = df.groupby(['rows', 'cols', 'lat', 'lon'])[['kpeit', 'kkork', 'turve', 'mmaa',
       'mlaji', 'lumip', 'kost', 'lamp', 'VH', 'VV',  'NDVI', 'LAI', 'dem', 'cos_ing',
        'slope_deg', 'aspect_deg','cslo', 'aspm', 'aspe']].mean().reset_index()

    return grouped_df

def define_projection(width1, height1, lat, lon):
    ''' WGS84 projection  '''

    area_id = 'finland'
    description = 'Finland'
    proj_id = 'finland'
    #projection = '+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs'  # projektio
    projection = '+proj=longlat +datum=WGS84 +no_defs'
    width = width1
    height = height1
    #pp = pyproj.Proj(proj='utm', zone=34, ellps='GRS80')
    #xx1, yy1 = pp(np.min(df['lon']), np.min(df['lat']))  # LL_lon, LL_lat: halutun alueen reunapikselit lower left
    #xx2, yy2 = pp(np.max(df['lon']), np.max(df['lat']))  # UR_lon, UR_lat: halutun alueen reunapikselit upper right
    #area_extent = (xx1, yy1, xx2, yy2)
    area_extent = (np.min(lon), np.min(lat), np.max(lon), np.max(lat))
    area_def1 = geometry.AreaDefinition(area_id, description, proj_id,
                                        projection, width, height,
                                        area_extent)
    return area_def1

def resample_data(param, area_def, lons2, lats2, xs, ys):
    swath_def = pyresample.geometry.SwathDefinition(lons=lons2, lats=lats2)
    (valid_product_indices, swath_indices, indices, distances) = \
        pyresample.kd_tree.get_neighbour_info(swath_def, area_def, radius_of_influence=10000,
                                              epsilon=0.5, neighbours=1)
    x_size = xs
    y_size = ys

    # Define a unique fill value for missing or invalid data
    fill_value = -9999  # Choose a large negative integer

    res = pyresample.kd_tree.get_sample_from_neighbour_info('nn', (y_size, x_size), param,
                                                            valid_product_indices, swath_indices,
                                                            indices, fill_value=fill_value)

    return res

def combine_columns(df):
    # create list of columns for each month
    june = df[['kost06', 'lamp06', 'VH_June', 'VV_June','timestamp06',
               'kpeit', 'kkork', 'turve', 'mmaa', 'mlaji', 'lumip',  'NDVI', 'LAI',
               'lat', 'lon', 'rows', 'cols', 'closest_sar_date06']].copy()
    july = df[['kost07', 'lamp07','VH_July', 'VV_July', 'timestamp07',
               'kpeit', 'kkork', 'turve', 'mmaa', 'mlaji', 'lumip',  'NDVI', 'LAI',
               'lat', 'lon', 'rows', 'cols', 'closest_sar_date07']].copy()
    august = df[['kost08', 'lamp08', 'VH_August', 'VV_August','timestamp08',
               'kpeit', 'kkork', 'turve', 'mmaa', 'mlaji', 'lumip',  'NDVI', 'LAI',
               'lat', 'lon', 'rows', 'cols', 'closest_sar_date08']].copy()

    # rename columns
    june = june.rename(columns={'kost06':'kost','VH_June':'VH','VV_June': 'VV','lamp06': 'lamp','timestamp06': 'timestamp', 'closest_sar_date06': 'closest_sar_date'})
    july = july.rename(columns={'kost07': 'kost', 'VH_July': 'VH', 'VV_July': 'VV', 'lamp07': 'lamp', 'timestamp07': 'timestamp', 'closest_sar_date07': 'closest_sar_date'})
    august = august.rename(columns={'kost08': 'kost', 'VH_August': 'VH', 'VV_August': 'VV', 'lamp08': 'lamp', 'timestamp08': 'timestamp', 'closest_sar_date08': 'closest_sar_date'})

    # combine dataframes
    df_combined = pd.concat([june, july, august], axis=0)
    print(df_combined.columns)
    print(df_combined.head())

    return df_combined
#----------------------------------------------------------------------------------

# lat and lon of sar data
lat_s = get_lat_lon_from_sar(stack_path)[0]
lon_s = get_lat_lon_from_sar(stack_path)[1]

# add sar data with the closest row and column as well as date to in-situ measurement
df = find_closest_row_column(field_data, lat_s, lon_s)
df = find_closest_date(df,sar_path)

# add despeckled values to in-situ data
df = add_despeckled_to_insitu(despeckled_path, df)

# add NDVI and LAI values to in-situ data
df = add_ndvi_lai_from_sar(stack_path, df)

print(df.columns)

df = combine_columns(df)

# add angle values to in-situ data
df = add_angles_to_dataframe(df, dem_path, stack_path)
df = add_temp_rain(df, weather)

print(df.columns)
print(df.head())
print(df.shape)

# transform dataframe to represent SAR data pixels
#gen_10m = create_dataframe_based_on_sar_pixels(df)
#print(gen_10m.shape)

#combined_10m.to_csv(path_start + 'kilpisjarvi_combined_data_10m.csv', index=False)

#combined_10m = pd.read_pickle(path_start + 'kilpisjarvi_combined_data_monthly_10m.pkl')

df.to_pickle(path_start + 'kilpisjarvi_combined_data_10m.pkl')
#gen_10m.to_pickle(path_start + 'kilpisjarvi_combined_data_gen_10m.pkl')
#ndvi_lai = calculate_ndvi_lai(stack_path)

#print(combined_10m.columns)

print(df['NDVI'].describe())
print(df['LAI'].describe())
print(df['rain'].isna().sum())

'''
#define projection and resample data to half of the original resolution
data_shape = lat_s.shape
area_def1 = define_projection((data_shape[1] // 2), (data_shape[0] // 2), lat_s, lon_s)
print(area_def1)

# resample lat and lon
lat_s2 = resample_data(lat_s, area_def1, lon_s, lat_s, data_shape[1] // 2, data_shape[0] // 2)
lon_s2 = resample_data(lon_s, area_def1, lon_s, lat_s, data_shape[1] // 2, data_shape[0] // 2)
print(lon_s2.shape)

# make copy of field data dataframe and repeat procedures, remember to resample each layer!
df2 = field_data.copy()
df2 = find_closest_row_column(df2, lat_s2, lon_s2)
df2 = find_closest_date(df2, sar_path)
df2 = add_despeckled_to_insitu(despeckled_path, df2, area_def1, lon_s, lat_s, data_shape[1] // 2, data_shape[0] // 2)
df2 = add_ndvi_lai_from_sar(stack_path, df2, area_def1, lon_s, lat_s, data_shape[1] // 2, data_shape[0] // 2)
df2 = add_angles_to_dataframe(df2, dem_path, stack_path, area_def1, lon_s, lat_s, data_shape[1] // 2, data_shape[0] // 2)
#combined_20m = combine_columns(df2)
combined_20m = create_dataframe_based_on_sar_pixels(df2)
#combined_20m.to_csv(path_start + 'kilpisjarvi_combined_data_40m.csv', index=False)
combined_20m.to_pickle(path_start + 'kilpisjarvi_combined_data_monthly_20m.pkl')

print(combined_20m.shape)
print(combined_20m.columns)
print(combined_20m.head())




#import geopandas as gpd
#from shapely.geometry import Point
# save results
#df.to_csv(path_start + 'kilpisjarvi_combined_data.csv', index=False)
#geometry1 = [Point(xy) for xy in zip(combined_10m['lon'], combined_10m['lat'])]
#gdf = gpd.GeoDataFrame(combined_10m, geometry=geometry1)
#gdf.to_file(path_start + 'kilpisjarvi_combined_data.gpkg', driver="GPKG")
'''

corr_params= ['kost', 'lamp', 'VH', 'VV', 'kpeit', 'kkork', 'turve',
       'mmaa', 'mlaji', 'lumip', 'NDVI', 'LAI', 'dem', 'cos_ing',
        'slope_deg', 'aspect_deg', 'cslo', 'aspm', 'aspe', 'rain', 'temp']


# make correlation plot
df_corr1 = df[corr_params].corr()
plt.figure(1,figsize=(20, 15))
heatmap = sns.heatmap(df_corr1, vmin=-1, vmax=1, annot=True, cmap='BrBG', annot_kws={"size": 12})
# Increase the size of the x and y axis labels
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

plt.savefig(path_start + 'correlation/correlation_heatmap_10m.png')

