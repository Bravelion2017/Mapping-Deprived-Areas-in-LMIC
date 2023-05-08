#%%
# Libraries Importation
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from osgeo import gdal
# import gdalwrap
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
import matplotlib.pyplot as plt
import random
from toolbox import contextual_features

seed= 42
np.random.seed(seed)
random.seed(seed)
#%%
#=======================LAGOS CITY=======================================

abspath_curr=os.getcwd()+os.sep+'Lagos'+os.sep
# abspath_curr=os.getcwd()+os.sep+'nairobi'+os.sep+'nairobi'+os.sep
spfea144= sorted(glob(abspath_curr+'lagos_contextual_10m'+os.sep+'spfea'+os.sep+'*.tif'))
mask= sorted(glob(abspath_curr+'lagos_contextual_10m'+os.sep+'mask'+os.sep+'*.tif'))
print(f'Length of shapes vs labels: {len(mask),len(spfea144)}')

contextual_features= contextual_features()

#%%
Tiles = []
for item in zip(spfea144, mask):
    # print(item)
    Tiles.append(item)
# Shuffle the samples
random.shuffle(Tiles)

# Seperate features and labels
RasterTiles = [] #features
MaskTiles = [] #labels
for raw_image, label in Tiles:
    RasterTiles.append(raw_image)
    MaskTiles.append(label)

#np.ceil(0.7*43)=31
#Create train validation & test lists
X_test_list= [f'/home/ubuntu/capstone/Lagos/lagos_contextual_10m/spfea/lag_area{i}.tif' for i in [11,13,20,23,25,28]]
RasterTiles =np.setdiff1d(RasterTiles,X_test_list).tolist()
X_train_list = RasterTiles[0:20]
X_val_list = RasterTiles[20:]

y_test_list= [f'/home/ubuntu/capstone/Lagos/lagos_contextual_10m/mask/lag_area{i}.tif' for i in [11,13,20,23,25,28]]
MaskTiles =np.setdiff1d(MaskTiles,y_test_list).tolist()
y_train_list = MaskTiles[0:20]
y_val_list = MaskTiles[20:]

# Preparing training

raster_to_mosiac = []

for p in X_train_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)


Xtrain, out_transform = merge(raster_to_mosiac)
print(Xtrain.shape)

raster_to_mosiac = []

for p in y_train_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)

ytrain, out_transform = merge(raster_to_mosiac)
print(ytrain.shape)

# Preparing validation
raster_to_mosiac = []

for p in X_val_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)


Xval, out_transform = merge(raster_to_mosiac)
print(Xval.shape)

raster_to_mosiac = []

for p in y_val_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)

yval, out_transform = merge(raster_to_mosiac)
print(yval.shape)



# Preparing test data

raster_to_mosiac = []

for p in X_test_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)


Xtest, out_transform = merge(raster_to_mosiac)
print(Xtest.shape)

raster_to_mosiac = []

for p in y_test_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)

ytest, out_transform = merge(raster_to_mosiac)
print(ytest.shape)

# Dataframe Setup

#Training Data
X_train = Xtrain[:, Xtrain[0,...]!=-9999]
y_train = ytrain[:, ytrain[0,...]!=-9999]
y_train =y_train.astype(int)
X_train = np.transpose(X_train)
y_train = np.transpose(y_train)

#Validation Data
X_val = Xval[:, Xval[0,...]!=-9999]
y_val = yval[:, yval[0,...]!=-9999]
y_val = y_val.astype(int)
X_val = np.transpose(X_val)
y_val = np.transpose(y_val)
#Testing Data
X_test = Xtest[:, Xtest[0,...]!=-9999]
y_test = ytest[:, ytest[0,...]!=-9999]
y_test = y_test.astype(int)
X_test = np.transpose(X_test)
y_test = np.transpose(y_test)

# To DataFrames- Train, Validation & Test
df_train = pd.DataFrame(X_train,columns=contextual_features)  # contextual features
df_train['labels'] = y_train
df_train['type']= 'train'

df_val = pd.DataFrame(X_val,columns=contextual_features)  # contextual features
df_val['labels'] = y_val
df_val['type']= 'validation'

df_test = pd.DataFrame(X_test,columns=contextual_features)  # contextual features
df_test['labels'] = y_test
df_test['type'] = 'test'
#%%
lagos_df = pd.concat([df_train,df_val,df_test])
# Export to parquet
lagos_df.to_parquet('lagos_df.parquet.gzip',compression='gzip')