# Libraries Importation
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
import gdalwrap
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
import matplotlib.pyplot as plt
import random

from utils import utils
from utils import util_preprocess

seed= 42
np.random.seed(seed)
random.seed(seed)
#=======================NAIROBI CITY=======================================

# abspath_curr=os.getcwd()+os.sep+'Lagos'+os.sep
abspath_curr=os.getcwd()+os.sep+'nairobi'+os.sep+'nairobi'+os.sep
spfea144= sorted(glob(abspath_curr+'nairobi_processed_TrainData_10m'+os.sep+'spfea_144'+os.sep+'*.tif'))
mask= sorted(glob(abspath_curr+'nairobi_processed_TrainData_10m'+os.sep+'mask'+os.sep+'*.tif'))
print(f'Length of shapes vs labels: {len(mask),len(spfea144)}')

contextual_features= ['fourier_sc31_mean',
 'fourier_sc31_variance',
 'fourier_sc51_mean',
 'fourier_sc51_variance',
 'fourier_sc71_mean',
 'fourier_sc71_variance',
 'gabor_sc3_filter_1',
 'gabor_sc3_filter_10',
 'gabor_sc3_filter_11',
 'gabor_sc3_filter_12',
 'gabor_sc3_filter_13',
 'gabor_sc3_filter_14',
 'gabor_sc3_filter_2',
 'gabor_sc3_filter_3',
 'gabor_sc3_filter_4',
 'gabor_sc3_filter_5',
 'gabor_sc3_filter_6',
 'gabor_sc3_filter_7',
 'gabor_sc3_filter_8',
 'gabor_sc3_filter_9',
 'gabor_sc3_mean',
 'gabor_sc3_variance',
 'gabor_sc5_filter_1',
 'gabor_sc5_filter_10',
 'gabor_sc5_filter_11',
 'gabor_sc5_filter_12',
 'gabor_sc5_filter_13',
 'gabor_sc5_filter_14',
 'gabor_sc5_filter_2',
 'gabor_sc5_filter_3',
 'gabor_sc5_filter_4',
 'gabor_sc5_filter_5',
 'gabor_sc5_filter_6',
 'gabor_sc5_filter_7',
 'gabor_sc5_filter_8',
 'gabor_sc5_filter_9',
 'gabor_sc5_mean',
 'gabor_sc5_variance',
 'gabor_sc7_filter_1',
 'gabor_sc7_filter_10',
 'gabor_sc7_filter_11',
 'gabor_sc7_filter_12',
 'gabor_sc7_filter_13',
 'gabor_sc7_filter_14',
 'gabor_sc7_filter_2',
 'gabor_sc7_filter_3',
 'gabor_sc7_filter_4',
 'gabor_sc7_filter_5',
 'gabor_sc7_filter_6',
 'gabor_sc7_filter_7',
 'gabor_sc7_filter_8',
 'gabor_sc7_filter_9',
 'gabor_sc7_mean',
 'gabor_sc7_variance',
 'hog_sc3_kurtosis',
 'hog_sc3_max',
 'hog_sc3_mean',
 'hog_sc3_skew',
 'hog_sc3_variance',
 'hog_sc5_kurtosis',
 'hog_sc5_max',
 'hog_sc5_mean',
 'hog_sc5_skew',
 'hog_sc5_variance',
 'hog_sc7_kurtosis',
 'hog_sc7_max',
 'hog_sc7_mean',
 'hog_sc7_skew',
 'hog_sc7_variance',
 'lac_sc3_lac',
 'lac_sc5_lac',
 'lac_sc7_lac',
 'lbpm_sc3_kurtosis',
 'lbpm_sc3_max',
 'lbpm_sc3_mean',
 'lbpm_sc3_skew',
 'lbpm_sc3_variance',
 'lbpm_sc5_kurtosis',
 'lbpm_sc5_max',
 'lbpm_sc5_mean',
 'lbpm_sc5_skew',
 'lbpm_sc5_variance',
 'lbpm_sc7_kurtosis',
 'lbpm_sc7_max',
 'lbpm_sc7_mean',
 'lbpm_sc7_skew',
 'lbpm_sc7_variance',
 'lsr_sc31_line_contrast',
 'lsr_sc31_line_length',
 'lsr_sc31_line_mean',
 'lsr_sc51_line_contrast',
 'lsr_sc51_line_length',
 'lsr_sc51_line_mean',
 'lsr_sc71_line_contrast',
 'lsr_sc71_line_length',
 'lsr_sc71_line_mean',
 'mean_sc3_mean',
 'mean_sc3_variance',
 'mean_sc5_mean',
 'mean_sc5_variance',
 'mean_sc7_mean',
 'mean_sc7_variance',
 'ndvi_sc3_mean',
 'ndvi_sc3_variance',
 'ndvi_sc5_mean',
 'ndvi_sc5_variance',
 'ndvi_sc7_mean',
 'ndvi_sc7_variance',
 'orb_sc31_kurtosis',
 'orb_sc31_max',
 'orb_sc31_mean',
 'orb_sc31_skew',
 'orb_sc31_variance',
 'orb_sc51_kurtosis',
 'orb_sc51_max',
 'orb_sc51_mean',
 'orb_sc51_skew',
 'orb_sc51_variance',
 'orb_sc71_kurtosis',
 'orb_sc71_max',
 'orb_sc71_mean',
 'orb_sc71_skew',
 'orb_sc71_variance',
 'pantex_sc3_min',
 'pantex_sc5_min',
 'pantex_sc7_min',
 'sfs_sc31_max_line_length',
 'sfs_sc31_max_ratio_of_orthogonal_angles',
 'sfs_sc31_mean',
 'sfs_sc31_min_line_length',
 'sfs_sc31_std',
 'sfs_sc31_w_mean',
 'sfs_sc51_max_line_length',
 'sfs_sc51_max_ratio_of_orthogonal_angles',
 'sfs_sc51_mean',
 'sfs_sc51_min_line_length',
 'sfs_sc51_std',
 'sfs_sc51_w_mean',
 'sfs_sc71_max_line_length',
 'sfs_sc71_max_ratio_of_orthogonal_angles',
 'sfs_sc71_mean',
 'sfs_sc71_min_line_length',
 'sfs_sc71_std',
 'sfs_sc71_w_mean']

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
X_test_list= [f'/home/ubuntu/capstone/nairobi/nairobi/nairobi_processed_TrainData_10m/spfea_144/nai_area{i}.tif' for i in [11,13,20,23,25,28]]
RasterTiles =np.setdiff1d(RasterTiles,X_test_list).tolist()
X_train_list = RasterTiles[0:30]
X_val_list = RasterTiles[30:]

y_test_list= [f'/home/ubuntu/capstone/nairobi/nairobi/nairobi_processed_TrainData_10m/mask/nai_area{i}.tif' for i in [11,13,20,23,25,28]]
MaskTiles =np.setdiff1d(MaskTiles,y_test_list).tolist()
y_train_list = MaskTiles[0:30]
y_val_list = MaskTiles[30:]

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

nairobi_df = pd.concat([df_train,df_val,df_test])
# Export to parquet
nairobi_df.to_parquet('nairobi_df.parquet.gzip',compression='gzip')