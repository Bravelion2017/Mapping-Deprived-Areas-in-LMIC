#%%
from osgeo import gdal
import os
from glob import glob
from subprocess import Popen
import fiona
#%%
spfea10= sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb/"+'*.tif'))
mask10= sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb_mask/"+'*.tif'))
lag_areas= sorted(os.listdir("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb/"))
# lag_areas.pop(1) # remove xml
new_spfea100=[]
new_mask100=[]
for i in lag_areas:
    new_spfea100.append("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb_100m/"+i)
for j in lag_areas:
    new_mask100.append("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb_mask_100m/"+j)
#%%
def resample_tif(in_path,out_path,xres,yres,resampling_algo='bilinear'):
    ds = gdal.Warp(out_path, in_path, warpOptions=dict(xRes=xres, yRes=yres, resampleAlg=resampling_algo),dstSRS='EPSG:4326')
    ds = None

#%%
# for old,new in zip(spfea10,new_spfea100):
#     tif= gdal.Open(old)
#     resample_tif(tif,new,100,100)
# for old,new in zip(mask10,new_mask100):
#     tif = gdal.Open(old)
#     resample_tif(tif,new,100,100)
#%%
# Convert tifs(features) to 1 vrt for resampling
output1 = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt/"
output2 = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt_mask"
vrt_options = gdal.BuildVRTOptions(outputSRS=None) #outputSRS=None
vrt =  gdal.BuildVRT(f'{output1}/lag_rgb.vrt', spfea10, options=vrt_options)
vrt = None
#%%
# Convert tifs(mask) to 1 vrt for resampling
vrt =  gdal.BuildVRT(f'{output2}/lag_rgb_mask.vrt', mask10, options=vrt_options)
vrt = None
#%%
# gdal_warp
# resampling vrt
fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt/lag_rgb.vrt"
outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt/lag_rgb_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
# can use 0.00083 0.00083 or 0.0008333333299999819968 -0.0009722222183333369952
#%%
# resampling vrt
fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt_mask/lag_rgb_mask.vrt"
outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt_mask/lag_rgb_mask_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r near  -of GTiff {fp} {outfile}'
Popen(command, shell=True)

#%%
# Clip using area polygon
# Clip raster(features) using polygon Tiles
polygons = sorted(glob('C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/temp/ClipPolygon/*.shp'))
VRT = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt/lag_rgb_resam.vrt"
outfile = "C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb_100m"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    command = f'gdalwarp -dstnodata -9999 -ts 6 6 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    # command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'

    Popen(command, shell=True)
#%%
# Clip using area polygon
# Clip raster(mask) using polygon Tiles
polygons = sorted(glob('C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/temp/IS_Tiles/*.shp'))
VRT = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/vrt_mask/lag_rgb_mask_resam.vrt"
outfile = "C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb_mask_100m"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    command = f'gdalwarp -dstnodata -9999 -ts 6 6 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    # command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'

    Popen(command, shell=True)
#%%
# Check height and width of created tif files
# import rasterio
# files = sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_rgb/rgb_100m/"+'*.tif'))
# num = 0
# for raster in files:
#     # print(reference_raster)
#     # img = utils_funcs.read_image(raster)
#     img= rasterio.open(raster).read()
#     width = img.shape[1]
#     height = img.shape[2]
#     num += 1
#     print(num, width, height)

#%%
# from toolbox import contextual_features, tif_to_df
# import numpy as np
# import pandas as pd
# import rasterio
# from rasterio.merge import merge

#%%
#check shape of areas
# from toolbox import covariate_features, tif_to_df, contextual_features
# covariate_features= covariate_features()
# cf= contextual_features()
# try:
#     for i in range(30):
#         area0= [f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_covariate/covariate_100m/lag_area{i}.tif"]
#         area0_mask= [f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_covariate/mask_100m/lag_area{i}.tif"]
#         area0_df= tif_to_df(area0,area0_mask,covariate_features)
#         print(area0_df.shape)
#         area0= [f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_contextual_100m/spfea_100m/lag_area{i}.tif"]
#         area0_mask= [f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_contextual_100m/mask_100m/lag_area{i}.tif"]
#         area0_df= tif_to_df(area0,area0_mask,cf)
#         print(area0_df.shape)
# except:
#     print(f'GIS ERROR')