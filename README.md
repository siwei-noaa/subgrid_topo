# subgrid_topo

This includes scripts for processing high-resolution DEM data to get the required subgrid information.
To use these scripts, a lot of GIS libraries are required.
For more details, you can refer to [how to install Python + GIS](https://automating-gis-processes.github.io/2016/Installing_Anacondas_GIS.html)

# Steps in getting subgrid information

#### Step 1. Download. 
  High-resolution DEM data are required for getting the subgrid information.
  Generally, DEM can be downloaded from 
  <https://www.usgs.gov/core-science-systems/ngp/tnm-delivery/>.
  
#### Step 2. Calculation.
  Calculate slope and aspect of the study domain from the DEM, and
  then their Trigonometry functions SIN and COS. Then A1, A2,
  and A3. The results are saved in ***output_dir***.
  The script for doing such job is ***subgrid_info_calulate.py***.

#### Step 3. Upscale.
  The high-resolution information has been calculated in the previous step. 
  In this step, we need to calculate mean, variance, and covariance of A1, A2, and A3
  for coarse resolution. This is similar as upscale high-resolution data into
  coarse-resolution.
  The script for doing this job is ***subgrid_info_extract.py***. 

  In this script, ***input_lat*** and ***input_lon*** are ASCII files that include 
  latitudes (corner, XLAT_V in HRRR) and longitudes (corner, XLONG_V in HRRR) of all 
  the grid cells at coarse-resolution.  The final output of this script
  is a shapefile that inlcudes all the required information for each grid cell.
  
  **Step 3 can be very time consume, if there are a lot of grid cells**.


More details on the subgrid information can refer to He et al. (2019). 

-----------------------------------------------------------------------

He, S., Smirnova, T. G., & Benjamin, S. G. (2019). A scale-aware 
parameterization for estimating subgrid variability of downward solar 
radiation using high-resolution digital elevation model data. Journal 
of Geophysical Research: Atmospheres, 124, 13680-13692. 
[https://doi.org/10.1029/2019JD031563](https://doi.org/10.1029/2019JD031563)

-----------------------------------------------------------------------


For any questions, you can reach me with <siwei.he@noaa.gov> or <hesiweide@gmail.com>.
