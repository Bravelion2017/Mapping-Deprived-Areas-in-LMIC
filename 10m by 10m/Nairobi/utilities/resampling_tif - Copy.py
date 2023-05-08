#%%
from osgeo import gdal
import os
from glob import glob
from subprocess import Popen
import fiona
#%%
spfea10= sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/spfea_144/"+'*.tif'))
mask10= sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/mask/"+'*.tif'))
nai_areas= sorted(os.listdir("C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/spfea_144/"))
nai_areas.pop(1) # remove xml
new_spfea100=[]
new_mask100=[]
for i in nai_areas:
    new_spfea100.append("C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/spfea_144_100m/"+i)
for j in nai_areas:
    new_mask100.append("C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/mask_100m/"+j)
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
output1 = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_spfea/"
output2 = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_mask"
vrt_options = gdal.BuildVRTOptions(outputSRS=None)
vrt =  gdal.BuildVRT(f'{output1}/nai_spfea_144.vrt', spfea10, options=vrt_options)
vrt = None
#%%
# Convert tifs(mask) to 1 vrt for resampling
vrt =  gdal.BuildVRT(f'{output2}/nai_mask.vrt', mask10, options=vrt_options)
vrt = None
#%%
# gdal_warp
# resampling vrt
fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_spfea/nai_spfea_144.vrt"
outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_spfea/nai_spfea_144_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.00083 0.00083 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
#%%
# resampling vrt
fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_mask/nai_mask.vrt"
outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_mask/nai_mask_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.00083 0.00083 -r near  -of GTiff {fp} {outfile}'
Popen(command, shell=True)

#%%
# Clip using area polygon
# Clip raster(features) using polygon Tiles
polygons = sorted(glob('C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/temp/ClipPolygon/*.shp'))
VRT = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_spfea/nai_spfea_144_resam.vrt"
outfile = "C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/spfea_144_100m"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    command = f'gdalwarp -dstnodata -9999 -ts 8 8 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    # command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'

    Popen(command, shell=True)
#%%
# Clip using area polygon
# Clip raster(mask) using polygon Tiles
polygons = sorted(glob('C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/temp/IS_Tiles/*.shp'))
VRT = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/vrt_mask/nai_mask_resam.vrt"
outfile = "C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi/nairobi/nairobi_processed_TrainData_10m/mask_100m"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    command = f'gdalwarp -dstnodata -9999 -ts 8 8 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    # command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'

    Popen(command, shell=True)
#%%
