#!/usr/bin/env python

"""
Geo-Python functions
Written by Siwei He, April 2020
"""
import os, sys
import pylab as pl
from osgeo import gdal, ogr, osr
from rasterstats import zonal_stats
import geopandas as gpd
import rasterio as rio
import numpy as np
import fiona
from shapely.geometry import Polygon, MultiPolygon
from fiona.crs import from_epsg
from multiprocessing import Pool, cpu_count
from functools import partial
import time

'''Define function to run mutiple processors and pool the results together'''
def run_multiprocessing(func, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(func, i)

def Creating_shpfile_polygon(coordinates,*epsg):
    '''
    This function creates a polygon or multipolygon shape files from given points
    The input coordinates is a list, with points in tuples
    CRS is optional, but the default CRS is EPSG:4326
    '''
    # creating shape file
    newdata = gpd.GeoDataFrame()
    newdata['geometry'] = None
    
    # creating polygon file
    if (len(coordinates)==0):
        sys.exit('ERROR: the input list is empty, please double check...')
    elif (len(coordinates)==1):
        polygons = Polygon(coordinates)
        newdata['geometry'] = polygon
    else:
        for i in range(len(coordinates)):
            polygon = Polygon(coordinates[i])
            newdata.loc[i, 'geometry'] = polygon
   
    if epsg:
        newdata.crs = from_epsg(epsg)
    else:
        newdata.crs = from_epsg(4326)

    # return shp file
    return newdata

def Subgrid_raster_save_shape(lat,lon,output_A1,output_A2,output_A3,row_i,col_j,output_shp_divide):
    '''get raster for one shape file, and save it'''

    grid_cell = []
    for i in row_i:
        for j in col_j:
            temp_poly = [(lon[i,j],lat[i,j]),(lon[i+1,j],lat[i+1,j]),\
                    (lon[i+1,j+1],lat[i+1,j+1]),(lon[i,j+1],lat[i,j+1])]
            grid_cell.append(temp_poly)

    newdata = Creating_shpfile_polygon(grid_cell)
    newdata.to_file(output_shp_divide) # first time output
    
    #start = time.process_time()  # for estimating computation time
    start = time.time()

    print ('Calculating subgrid topography info ... ')
    #meanA1,meanA2,meanA3,variA1,variA2,variA3,covA1A2,covA1A3,covA2A3 = \
    #        Subgrid_topo_stats(output_shp_divid, output_A1, output_A2, output_A3)
    meanA1,meanA2,meanA3,variA1,variA2,variA3,covA1A2,covA1A3,covA2A3 = \
            Subgrid_topo_stats_parallel(output_shp_divide, output_A1, output_A2, output_A3)
    
    print("Mutiprocessing time: {}secs\n".format((time.time()-start)))

    newdata['meanA1'] = meanA1
    newdata['meanA2'] = meanA2
    newdata['meanA3'] = meanA3
    newdata['variA1'] = variA1
    newdata['variA2'] = variA2
    newdata['variA3'] = variA3
    newdata['covA1A2'] = covA1A2
    newdata['covA1A3'] = covA1A3
    newdata['covA2A3'] = covA2A3

    # for debugging plot
    if (False):
        newdata.plot(column='meanA1')
        newdata.plot(column='meanA2')
        newdata.plot(column='meanA3')
        newdata.plot(column='variA1')
        newdata.plot(column='variA2')
        newdata.plot(column='variA3')
        newdata.plot(column='covA1A2')
        newdata.plot(column='covA1A3')
        newdata.plot(column='covA2A3')
        pl.show()
    
    print ('Saving subgrid topography info to '+ output_shp_divide)
    newdata.to_file(output_shp_divide)  # final output

def Subgrid_topo_stats_parallel(input_zone_polygon, rasterA1, rasterA2, rasterA3):
    '''
    This function get the stastic info for each features of the shape file.
    '''

    # Open shp file data
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    
    # multiprocessing parallel setting
    n_processors = cpu_count()
    feat_index = list(range(lyr.GetFeatureCount()))
    #for index in feat_index:
    #    a = get_raster_feat(input_zone_polygon,rasterA1,rasterA2,rasterA3,index)
    #exit()
    func = partial(get_raster_feat, input_zone_polygon, rasterA1, rasterA2, rasterA3)
    out_list = run_multiprocessing(func, feat_index, n_processors)

    out_list = np.asarray(out_list)
    shp_mean_A1 = out_list[:,0]
    shp_mean_A2 = out_list[:,1]
    shp_mean_A3 = out_list[:,2]
    shp_vari_A1 = out_list[:,3]
    shp_vari_A2 = out_list[:,4]
    shp_vari_A3 = out_list[:,5]
    shp_cov_A1A2 = out_list[:,6]
    shp_cov_A1A3 = out_list[:,7]
    shp_cov_A2A3 = out_list[:,8]

    return shp_mean_A1, shp_mean_A2, shp_mean_A3, shp_vari_A1, shp_vari_A2, shp_vari_A3, \
            shp_cov_A1A2, shp_cov_A1A3, shp_cov_A2A3

def get_raster_feat(input_zone_polygon, rasterA1, rasterA2, rasterA3, feat_index):
    '''This is a test function for parallel computing'''
    # Open data
    raster_A1 = gdal.Open(rasterA1)
    raster_A2 = gdal.Open(rasterA2)
    raster_A3 = gdal.Open(rasterA3)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster_A1.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster_A1.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    
    total_feat = lyr.GetFeatureCount()
    if ((feat_index % 500) == 0):
        print (feat_index, '/', total_feat)
    feat = lyr[feat_index]
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)
    
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
        count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)
        
    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
    
    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_A1.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    banddataraster_A1 = raster_A1.GetRasterBand(1)
    try:
        dataraster_A1 = banddataraster_A1.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
        banddataraster_A2 = raster_A2.GetRasterBand(1)
        dataraster_A2 = banddataraster_A2.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
        banddataraster_A3 = raster_A3.GetRasterBand(1)
        dataraster_A3 = banddataraster_A3.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
        
        bandmask = target_ds.GetRasterBand(1)
        datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
        
        # Mask zone of raster
        zoneraster_A1 = np.ma.masked_array(dataraster_A1,  np.logical_not(datamask))
        zoneraster_A2 = np.ma.masked_array(dataraster_A2,  np.logical_not(datamask))
        zoneraster_A3 = np.ma.masked_array(dataraster_A3,  np.logical_not(datamask))
        
        # stastic index
        NoData = banddataraster_A1.GetNoDataValue()
        index = np.where(zoneraster_A1==NoData)
        zoneraster_A1[index] = np.nan
        zoneraster_A2[index] = np.nan
        zoneraster_A3[index] = np.nan
        shp_mean_A1 = np.ma.mean(zoneraster_A1)
        shp_mean_A2 = np.ma.mean(zoneraster_A2)
        shp_mean_A3 = np.ma.mean(zoneraster_A3)
        shp_vari_A1 = np.ma.var(zoneraster_A1)
        shp_vari_A2 = np.ma.var(zoneraster_A2)
        shp_vari_A3 = np.ma.var(zoneraster_A3)
        shp_cov_A1A2 = np.ma.cov(zoneraster_A1.flatten(),zoneraster_A2.flatten(),bias=True)[1,0]
        shp_cov_A1A3 = np.ma.cov(zoneraster_A1.flatten(),zoneraster_A3.flatten(),bias=True)[1,0]
        shp_cov_A2A3 = np.ma.cov(zoneraster_A3.flatten(),zoneraster_A2.flatten(),bias=True)[1,0]
    except:
        shp_mean_A1 = np.nan
        shp_mean_A2 = np.nan
        shp_mean_A3 = np.nan
        shp_vari_A1 = np.nan
        shp_vari_A2 = np.nan
        shp_vari_A3 = np.nan
        shp_cov_A1A2 = np.nan
        shp_cov_A1A3 = np.nan
        shp_cov_A2A3 = np.nan

    return shp_mean_A1, shp_mean_A2, shp_mean_A3, shp_vari_A1, shp_vari_A2, shp_vari_A3, \
            shp_cov_A1A2, shp_cov_A1A3, shp_cov_A2A3

def Subgrid_topo_stats(input_zone_polygon, rasterA1, rasterA2, rasterA3):
    '''
    This function get the stastic info for each features of the shape file.
    '''

    # Open data
    raster_A1 = gdal.Open(rasterA1)
    raster_A2 = gdal.Open(rasterA2)
    raster_A3 = gdal.Open(rasterA3)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster_A1.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster_A1.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    
    # statistic indexes
    shp_mean_A1 = np.zeros(lyr.GetFeatureCount())
    shp_mean_A2 = np.zeros(lyr.GetFeatureCount())
    shp_mean_A3 = np.zeros(lyr.GetFeatureCount())
    shp_vari_A1 = np.zeros(lyr.GetFeatureCount())
    shp_vari_A2 = np.zeros(lyr.GetFeatureCount())
    shp_vari_A3 = np.zeros(lyr.GetFeatureCount())
    shp_cov_A1A2 = np.zeros(lyr.GetFeatureCount())
    shp_cov_A1A3 = np.zeros(lyr.GetFeatureCount())
    shp_cov_A2A3 = np.zeros(lyr.GetFeatureCount())
    
    for feat_index in range(lyr.GetFeatureCount()):
        print (feat_index)
        feat = lyr[feat_index]
        geom = feat.GetGeometryRef()
        geom.Transform(coordTrans)
        
        if (geom.GetGeometryName() == 'MULTIPOLYGON'):
            count = 0
            pointsX = []; pointsY = []
            for polygon in geom:
                geomInner = geom.GetGeometryRef(count)
                ring = geomInner.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
        elif (geom.GetGeometryName() == 'POLYGON'):
            ring = geom.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            pointsX = []; pointsY = []
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            
        else:
            sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
        
        xmin = min(pointsX)
        xmax = max(pointsX)
        ymin = min(pointsY)
        ymax = max(pointsY)

        # Specify offset and rows and columns to read
        xoff = int((xmin - xOrigin)/pixelWidth)
        yoff = int((yOrigin - ymax)/pixelWidth)
        xcount = int((xmax - xmin)/pixelWidth)+1
        ycount = int((ymax - ymin)/pixelWidth)+1

        # Create memory target raster
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((
            xmin, pixelWidth, 0,
            ymax, 0, pixelHeight,
        ))

        # Create for target raster the same projection as for the value raster
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster_A1.GetProjectionRef())
        target_ds.SetProjection(raster_srs.ExportToWkt())

        # Rasterize zone polygon to raster
        gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

        # Read raster as arrays
        banddataraster_A1 = raster_A1.GetRasterBand(1)
        try:
            dataraster_A1 = banddataraster_A1.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
            banddataraster_A2 = raster_A2.GetRasterBand(1)
            dataraster_A2 = banddataraster_A2.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
            banddataraster_A3 = raster_A3.GetRasterBand(1)
            dataraster_A3 = banddataraster_A3.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
            
            bandmask = target_ds.GetRasterBand(1)
            datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
            
            # Mask zone of raster
            zoneraster_A1 = np.ma.masked_array(dataraster_A1,  np.logical_not(datamask))
            zoneraster_A2 = np.ma.masked_array(dataraster_A2,  np.logical_not(datamask))
            zoneraster_A3 = np.ma.masked_array(dataraster_A3,  np.logical_not(datamask))
            
            # stastic index
            shp_mean_A1[feat_index] = np.mean(zoneraster_A1)
            shp_mean_A2[feat_index] = np.mean(zoneraster_A2)
            shp_mean_A3[feat_index] = np.mean(zoneraster_A3)
            shp_vari_A1[feat_index] = np.var(zoneraster_A1)
            shp_vari_A2[feat_index] = np.var(zoneraster_A2)
            shp_vari_A3[feat_index] = np.var(zoneraster_A3)
            shp_cov_A1A2[feat_index] = np.ma.cov(zoneraster_A1.flatten(),zoneraster_A2.flatten(),bias=True)[1,0]
            shp_cov_A1A3[feat_index] = np.ma.cov(zoneraster_A1.flatten(),zoneraster_A3.flatten(),bias=True)[1,0]
            shp_cov_A2A3[feat_index] = np.ma.cov(zoneraster_A3.flatten(),zoneraster_A2.flatten(),bias=True)[1,0]
        except:
            shp_mean_A1[feat_index] = np.nan
            shp_mean_A2[feat_index] = np.nan
            shp_mean_A3[feat_index] = np.nan
            shp_vari_A1[feat_index] = np.nan
            shp_vari_A2[feat_index] = np.nan
            shp_vari_A3[feat_index] = np.nan
            shp_cov_A1A2[feat_index] = np.nan
            shp_cov_A1A3[feat_index] = np.nan
            shp_cov_A2A3[feat_index] = np.nan


    # return
    return shp_mean_A1, shp_mean_A2, shp_mean_A3, shp_vari_A1, shp_vari_A2, shp_vari_A3, \
            shp_cov_A1A2, shp_cov_A1A3, shp_cov_A2A3


def Polygon_raster_stats(input_zone_polygon, input_value_raster):
    '''
    This function get the stastic values for each features of the shape file.
    It is largely from https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    '''

    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    
    # statistic indexes
    shp_mean = np.zeros(lyr.GetFeatureCount())
    shp_std = np.zeros(lyr.GetFeatureCount())
    feat_index = 0

    for feat_index in range(lyr.GetFeatureCount()):
        feat = lyr[feat_index]
        geom = feat.GetGeometryRef()
        geom.Transform(coordTrans)
        
        if (geom.GetGeometryName() == 'MULTIPOLYGON'):
            count = 0
            pointsX = []; pointsY = []
            for polygon in geom:
                geomInner = geom.GetGeometryRef(count)
                ring = geomInner.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
        elif (geom.GetGeometryName() == 'POLYGON'):
            ring = geom.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            pointsX = []; pointsY = []
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            
        else:
            sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
        
        xmin = min(pointsX)
        xmax = max(pointsX)
        ymin = min(pointsY)
        ymax = max(pointsY)

        # Specify offset and rows and columns to read
        xoff = int((xmin - xOrigin)/pixelWidth)
        yoff = int((yOrigin - ymax)/pixelWidth)
        xcount = int((xmax - xmin)/pixelWidth)+1
        ycount = int((ymax - ymin)/pixelWidth)+1

        # Create memory target raster
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((
            xmin, pixelWidth, 0,
            ymax, 0, pixelHeight,
        ))

        # Create for target raster the same projection as for the value raster
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster.GetProjectionRef())
        target_ds.SetProjection(raster_srs.ExportToWkt())

        # Rasterize zone polygon to raster
        gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

        # Read raster as arrays
        banddataraster = raster.GetRasterBand(1)
        dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
        bandmask = target_ds.GetRasterBand(1)
        datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

        # Mask zone of raster
        zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))

        # stastic index
        shp_mean[feat_index] = np.mean(zoneraster)
        shp_std[feat_index] = np.std(zoneraster)

    # return
    return shp_mean, shp_std

def loop_zonal_stats(input_zone_polygon, input_value_raster):

    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    statDict = {}

    for FID in featList:
        feat = lyr.GetFeature(FID)
        print ('FEAT in loop: ', feat)
        meanValue = zonal_stats(input_zone_polygon, input_value_raster)
        statDict[FID] = meanValue
        print ('MEAN IN LOOP: ', meanValue)
    return statDict




