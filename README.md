# subgrid_topo

This includes scripts for processing raw DEM data to get the required subgrid information.

# Steps in getting subgrid information

##### Step 1. Download the high-resolution DEM data for the study domain.
  Generally, DEM can be downloaded from 
  <https://www.usgs.gov/core-science-systems/ngp/tnm-delivery/>.
  The script for downloading DEM data does not include in the directory.
  
#### Step 2. Calculation.
  Calculate slope and aspect of the study domain from the DEM, and
  then their Trigonometry functions SIN and COS. Then A1, A2,
  and A3. The script for doing such job is *subgrid_info_calulate.py*.

#### Step 3. Upscale.
  The high-resolution information has been calculated in the previous step. 
  In this step, we need to calculate mean, variance, and covariance of A1, A2, and A3
  for coarse resolution. This is similar as upscale high-resolution data into
  coarse-resolution.
  The script for doing this job is *subgrid_info_extract.py* 
  **This step can be very time consume, if there are a lot of grid cells**.


More details on the subgrid information can refer to He et al. (2019). 
-----------------------------------------------------------------------
He, S., Smirnova, T. G., & Benjamin, S. G. (2019). A scale-aware 
parameterization for estimating subgrid variability of downward solar 
radiation using high-resolution digital elevation model data. Journal 
of Geophysical Research: Atmospheres, 124, 13680-13692. 
[https://doi.org/10.1029/2019JD031563](https://doi.org/10.1029/2019JD031563)
-----------------------------------------------------------------------


For any questions, you can reach me with <siwei.he@noaa.gov> or <hesiweide@gmail.com>.
