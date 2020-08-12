#!/usr/bin/env python

"""
This script creates polygon using Shapely
Written by Siwei He, April 2020
"""
import os
import pylab as pl
from osgeo import gdal, ogr, osr
from rasterstats import zonal_stats
import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, mapping
import Geo_Python_utils
import time

def main():
    '''
    Define a main function 
    '''

    input_Dir = '/Users/siwei.he/Research/Data/HRRR/'
    output_Dir = '/Users/siwei.he/Research/Data/HRRR/'
    input_DEM = 'USGS_NED_100m.tif'
    input_lat = 'HRRR_corner_lat.asc'
    input_lon = 'HRRR_corner_lon.asc'
    output_shp = 'HRRR_subgrid_topo.shp'
    subgrid_info_var = {'meanA1':0,'meanA2':1,'meanA3':2,'variA1':3,'variA2':4,\
            'variA3':5,'covA1A2':6,'covA1A3':7,'covA2A3':8}
    
    # Define input and output files
    output_A1 = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_A1.tif')
    output_A2 = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_A2.tif')
    output_A3 = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_A3.tif')
    output_shp = os.path.join(output_Dir, output_shp)
    
    lat = np.genfromtxt(input_Dir+input_lat, delimiter=',')
    lon = np.genfromtxt(input_Dir+input_lon, delimiter=',')
    
    row, col = np.shape(lat)
    print (row, col)
    
    subgrid_info = np.ones((9, row-1,col-1))*(-9999)
    
    print ('Creating ESRI shape file from the inputted coordinates ... ')
    if (row*col >3E4):
        divide_threshold = 20
        # divide whole domain into small pieces and them merge together
        row_divide = list(range(0,row,divide_threshold))
        if ((row-row_divide[-1]) >(divide_threshold/2) ):
            row_divide.append(row-1)
        else:
            row_divide[-1] = row-1

        col_divide = list(range(0,col,divide_threshold))
        if ((col-col_divide[-1]) >(divide_threshold/2) ):
            col_divide.append(col-1)
        else:
            col_divide[-1] = col-1
        print (row_divide)
        print (col_divide)

        sub_divide = 0
        for sub_row in range(len(row_divide)-1):
            for sub_col in range(len(col_divide)-1):
                sub_divide = sub_divide + 1
                output_shp_divide = os.path.splitext(output_shp)[0] + '_temp{:d}.shp'.format(sub_divide)
                output_shp_divide = os.path.join(output_Dir, output_shp_divide)
                row_i = list(range(row_divide[sub_row],row_divide[sub_row+1]))
                col_j = list(range(col_divide[sub_col],col_divide[sub_col+1]))

                if (False):
                    # save the subdivide grid raster info
                    Geo_Python_utils.Subgrid_raster_save_shape(lat,lon,output_A1,output_A2,output_A3,\
                            row_i,col_j,output_shp_divide)
                else:
                    print ('reading ', output_shp_divide)
                    shp = gpd.read_file(output_shp_divide)
                    index = 0
                    for i in row_i:
                        for j in col_j:
                            for var,loc in subgrid_info_var.items():
                                subgrid_info[loc,i,j] = shp[var][index]
                            index = index + 1

        for var,loc in subgrid_info_var.items():
            file_name = os.path.join(output_Dir, var + '.asc')
            np.savetxt(file_name, subgrid_info[loc,:,:])

        exit()

        # combine shape file
        sub_divide = 0
        temp_file_list = []
        for sub_row in range(len(row_divide)-1):
            for sub_col in range(len(col_divide)-1):
        #for sub_row in range(len(row_divide)-3,len(row_divide)-1):
        #    for sub_col in range(len(col_divide)-3,len(col_divide)-1):
                sub_divide = sub_divide + 1
                output_shp_divide = os.path.splitext(output_shp)[0] + '_temp{:d}.shp'.format(sub_divide)
                output_shp_divide = os.path.join(output_Dir, output_shp_divide)
                temp_file_list.append(output_shp_divide)

        gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(shp) for shp in temp_file_list], ignore_index=True), \
                crs=gpd.read_file(temp_file_list[0]).crs)
        print ('Saving subgrid topography info to one file as '+ output_shp)
        gdf.to_file(output_shp)  # final output

    else:
        # save the subdivide grid raster info
        row_i = list(range(row))
        col_j = list(range(col))
        #row_i = list(range(200,220))
        #col_j = list(range(600,620))
        Geo_Python_utils.Subgrid_raster_save_shape(lat,lon,output_A1,output_A2,output_A3,\
                row_i,col_j,output_shp)


if __name__ == "__main__":
    
    main()
