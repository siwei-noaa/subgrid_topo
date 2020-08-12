#!/usr/bin/env python

"""

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
from osgeo import gdal

input_Dir = '/Users/siwei.he/Research/Data/HRRR/'
output_Dir = '/Users/siwei.he/Research/Data/HRRR/'
input_DEM = 'USGS_NED_100m.tif'

# Define input and output files
input_dem = os.path.join(input_Dir, input_DEM)
output_asp = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_asp.tif')
output_slp = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_slp.tif')
output_A1 = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_A1.tif')
output_A2 = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_A2.tif')
output_A3 = os.path.join(output_Dir, os.path.splitext(input_DEM)[0] + '_A3.tif')

# generate commands and calculate slope and aspect
command = ['gdaldem', 'slope', input_dem, output_slp]
print (command)
command_run = subprocess.run(command)
if command_run.returncode != 0:
    print ('ERROR in ', command)
    exit()

command = "gdaldem aspect %s %s -zero_for_flat -alg ZevenbergenThorne" %(input_dem, output_asp)
print (command)
command_run = subprocess.run(command, shell=True)
if command_run.returncode != 0:
    print ('ERROR in ', command)
    exit()

# generate commands and calculate A1, A2, and A3 
command = "gdal_calc.py --calc=\"sin(radians(A))*sin(radians(B))\" -A %s -B %s --outfile=%s --NoDataValue=-9999" \
        % (output_slp, output_asp, output_A1)
command_run = subprocess.run(command, shell=True)
if command_run.returncode != 0:
    print ('ERROR in ', command)
    exit()
command = "gdal_calc.py --calc=\"sin(radians(A))*cos(radians(B))\" -A %s -B %s --outfile=%s --NoDataValue=-9999" \
        % (output_slp, output_asp, output_A2)
command_run = subprocess.run(command, shell=True)
if command_run.returncode != 0:
    print ('ERROR in ', command)
    exit()
command = "gdal_calc.py --calc=\"cos(radians(A))\" -A %s --outfile=%s --NoDataValue=-9999" \
        % (output_slp, output_A3)
command_run = subprocess.run(command, shell=True)
if command_run.returncode != 0:
    print ('ERROR in ', command)
    exit()



