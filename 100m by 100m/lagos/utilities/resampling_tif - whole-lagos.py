#%%
from osgeo import gdal
import os
from glob import glob
from subprocess import Popen
import fiona
import rasterio
import numpy as np
import pandas as pd
from rasterio.merge import merge

#%%
fourier= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/fourier/"+'*.tif'))
gabor= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/gabor/"+'*.tif'))
hog= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/hog/"+'*.tif'))
lac= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/lac/"+'*.tif'))
lbpm= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/lbpm/"+'*.tif'))
lsr= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/lsr/"+'*.tif'))
mean= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/mean/"+'*.tif'))
ndvi= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/ndvi/"+'*.tif'))
orb= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/orb/"+'*.tif'))
pantex= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/pantex/"+'*.tif'))
sfs= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_10m/sfs/"+'*.tif'))

#%%
spfea= fourier+gabor+hog+lac+lbpm+lsr+mean+ndvi+orb+pantex+sfs
#%%
# spfea10= sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_covariate/covariate_100m/"+'*.tif'))
#%%
# Convert tifs to 1 vrt for resampling contextual
output1 = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_100m"
vrt_options = gdal.BuildVRTOptions(separate=True) #outputSRS=None or separate=True
vrt =  gdal.BuildVRT(f'{output1}/lag_contextual.vrt', spfea, options=vrt_options)
vrt = None

#%%
# Convert tifs to 1 vrt for resampling rgbn
rgbn= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_10m/lag_bgrn.tif"
output1 = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_100m"
vrt_options = gdal.BuildVRTOptions(outputSRS=None) #outputSRS=None or separate=True
vrt =  gdal.BuildVRT(f'{output1}/lag_rgbn.vrt', rgbn, options=vrt_options)
vrt = None
#%%
# Convert tifs(mask) to 1 vrt for resampling
# vrt =  gdal.BuildVRT(f'{output2}/lag_covariate_mask.vrt', mask10, options=vrt_options)
# vrt = None
#%%
# gdal_warp
# resampling vrt contextual
fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_100m/lag_contextual.vrt"
outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_100m/lag_contextual_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
# can use 0.00083 0.00083 or 0.0008333333299999819968 -0.0009722222183333369952

#%%
# gdal_warp
# resampling vrt rgbn
fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_100m/lag_rgbn.vrt"
outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_100m/lag_rgbn_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
# can use 0.00083 0.00083 or 0.0008333333299999819968 -0.0009722222183333369952
#%%
# Try vrt to Tif
ctx= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_contextual_100m/lag_contextual_resam.vrt"
rgb= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_100m/lag_rgbn_resam.vrt"
com= f"gdal_translate {ctx} lagos_contextual_100m.tif"
Popen(com, shell=True)
com2= f"gdal_translate {rgb} lagos_rgbn_100m.tif"
Popen(com2, shell=True)

#%%
