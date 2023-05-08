#%%
from osgeo import gdal
import os
from glob import glob
from subprocess import Popen
import fiona
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from PIL import Image

#%%
rgbn= sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/rgb/"+'*.tif'))

#%%
# Convert tifs(rgbn features) to 1 vrt
output = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/vrt_rgbn"
vrt_options = gdal.BuildVRTOptions(outputSRS=None) #seperate=True | outputSRS=None
vrt =  gdal.BuildVRT(f'{output}/lag_rgbn.vrt', rgbn, options=vrt_options)
vrt = None

#%%
# Clip polygons
# Clip using areas(30) polygon
# Clip raster(144 features) using polygon Tiles
# polygons = sorted(glob('C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/temp/ClipPolygon/*.shp'))
# VRT = f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/vrt_spfea/lag_spfea_144.vrt"
# outfile = "C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_contextual_10m/spfea"
# # print(polygons)
#
# for polygon in polygons:
#     # print(polygon)
#     feat = fiona.open(polygon, 'r')
#     # add output file name
#     head, tail = os.path.split(polygon)
#     name=tail[:-4]
#     # print(name)
#     command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
#     # command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
#
#     Popen(command, shell=True)

#%%
# Create mask (label tif) by rasterizing shape file
polygons = sorted(glob('C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/temp/IS_Tiles/*.shp'))
RasterTiles = sorted(glob(f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/rgb/"+'*.tif'))
outfile = "C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/rgb_mask/"

# zip polygon and raster in to a list of tuple
shp_rast = zip (polygons, RasterTiles)
lst = list(shp_rast)

def rasterize_me(in_shp, in_raster, outfile):
    # read shapfile
    for i in lst:
        df = gpd.read_file(i[0])
        # add output file name
        head, tail = os.path.split(i[0])
        name=tail[:-4]
    # read raster
        with rasterio.open(i[1], mode="r") as src:
            out_arr = src.read(1)
            out_profile = src.profile.copy()
            out_profile.update(count=1,
                            nodata=-9999,
                            dtype='float32',
                            width=src.width,
                            height=src.height,
                            crs=src.crs)
            dst_height = src.height
            dst_width = src.width
            shapes = ((geom,value) for geom, value in zip(df.geometry, df.CID))
            # print(shapes)
            burned = features.rasterize(shapes=shapes, out_shape=(dst_height, dst_width),fill=0, transform=src.transform)
            # im= Image.open(burned)
            # im.show()

        # open in 'write' mode, unpack profile info to dst
        with rasterio.open(f'{outfile}{name}.tif',
                        'w', **out_profile) as dst:
            dst.write_band(1, burned)

rasterize_me(in_shp=polygons, in_raster=RasterTiles, outfile=outfile)