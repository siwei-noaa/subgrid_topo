#!/usr/bin/env python

"""
This is the step 2 of subgrid pre-processing that was proposed in 

-----------------------------------------------------------------------
He, S., Smirnova, T. G., & Benjamin, S. G. (2019). A scale-aware
parameterization for estimating subgrid variability of downward solar
radiation using high-resolution digital elevation model data. Journal
of Geophysical Research: Atmospheres, 124, 13680????3692.
https://doi.org/10.1029/2019JD031563
-----------------------------------------------------------------------

A full steps can refer to 
/Users/siwei.he/Research/My_documents/Subgrid_pre_processing.note

"""
import os
import subprocess
import pylab as pl
from osgeo import gdal, ogr, osr
from rasterstats import zonal_stats
import geopandas as gpd
import rasterio as rio
import numpy
import fiona
from shapely.geometry import shape, mapping


input_Dir = '/Users/siwei.he/Research/Data/HRRR/'
output_Dir = '/Users/siwei.he/Research/Data/HRRR/'
input_file = 'HRRR_subgrid_topo_temp111.shp'
dem_file = 'USGS_NED_100m.tif'

# Define input and output files
input_shp = os.path.join(input_Dir, input_file)
input_dem = os.path.join(input_Dir, dem_file)
output_shp = os.path.join(output_Dir, os.path.splitext(input_file)[0] + 'rpj.shp')
output_raster = os.path.join(output_Dir, os.path.splitext(input_file)[0] + '.tif')

# get coordinate system
src = rio.open(input_dem)
myCRS = src.crs

x = gpd.read_file(input_shp)
print (x['meanA1'][0:10])

exit()
print (x.crs)
y = x.to_crs(myCRS)
y.to_file(output_shp)
lyr_name = os.path.basename(output_shp)
lyr_name = os.path.splitext(lyr_name)[0]
command = "gdal_rasterize -a meanA1 -a_nodata -9999 -tr 3000 3000 -l %s %s %s" \
        % (lyr_name, output_shp, output_raster)
print (command)

command_run = subprocess.run(command, shell=True)
if command_run.returncode != 0:
    print ('ERROR in ', command)
    exit()

exit()



