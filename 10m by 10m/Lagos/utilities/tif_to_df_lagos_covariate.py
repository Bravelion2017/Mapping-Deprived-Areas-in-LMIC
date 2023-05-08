#%%
# Libraries Importation
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
import matplotlib.pyplot as plt
import random

seed= 42
np.random.seed(seed)
random.seed(seed)

#%%
#=======================LAGOS CITY=======================================

abspath_curr=os.getcwd()+os.sep+'Lagos'+os.sep
# Get Lagos's path then populate 53 covariate features and their labels (mask)
spfea= sorted(glob(abspath_curr+'lagos_covariate_feature_53'+os.sep+'covariate_100m'+os.sep+'*.tif'))
mask= sorted(glob(abspath_curr+'lagos_covariate_feature_53'+os.sep+'mask_100m'+os.sep+'*.tif'))
print(f'Length of shapes vs labels: {len(mask),len(spfea)}')

# contextual features names
covariate_features= ['fs_dist_fs_2020',
'fs_dist_school_2020',
'in_dist_rd_2016',
'in_dist_rd_intersect_2016',
'in_dist_waterway_2016',
'in_night_light_2016',
'ph_base_water_2010',
'ph_bio_dvst_2015',
'ph_climate_risk_2020',
'ph_dist_aq_veg_2015',
'ph_dist_art_surface_2015',
'ph_dist_bare_2015',
'ph_dist_cultivated_2015',
'ph_dist_herb_2015',
'ph_dist_inland_water_2018',
'ph_dist_open_coast_2020',
'ph_dist_shrub_2015',
'ph_dist_sparse_veg_2015',
'ph_dist_woody_tree_2015',
'ph_gdmhz_2005',
'ph_grd_water_2000',
'ph_hzd_index_2011',
'ph_land_c1_2019',
'ph_land_c2_2020',
'ph_max_tem_2019',
'ph_ndvi_2019',
'ph_pm25_2016',
'ph_slope_2000',
'ses_an_visits_2014',
'ses_child_stuned_2014',
'ses_dtp3_2014',
'ses_hf_delivery_2014',
'ses_impr_water_src_2014',
'ses_ITN_2014',
'ses_m_lit_2014',
'ses_measles_2014',
'ses_odef_2014',
'ses_pfpr_2017',
'ses_preg_2017',
'ses_unmet_need_2014',
'ses_w_lit_2016',
'sh_dist_mnr_pofw_2019',
'sh_dist_pofw_2019',
'sh_ethno_den_2020',
'uu_bld_count_2020',
'uu_bld_den_2020',
'ho_impr_housing_2015',
'fs_dist_hf_2019',
'po_hrsl_2018',
'po_wp_2020',
'ph_dist_riv_network_2007',
'uu_urb_bldg_2018',
'ses_dist_gov_office_2022']
# Combining feature & label files together
Tiles = []
for item in zip(spfea, mask):
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


#Create train validation & test lists
X_test_list= [f'/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif' for i in [11,13,20,23,25,28]]
RasterTiles =np.setdiff1d(RasterTiles,X_test_list).tolist() #Removing items existing in X_test_list from RasterTiles
X_train_list = RasterTiles[0:20] #split train
X_val_list = RasterTiles[20:] #split validation

#Replicate the above for labels
y_test_list= [f'/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif' for i in [11,13,20,23,25,28]]
MaskTiles =np.setdiff1d(MaskTiles,y_test_list).tolist()
y_train_list = MaskTiles[0:20]
y_val_list = MaskTiles[20:]

# Preparing training
raster_to_mosiac = []
# Read the tif files with rasterio then parse into list
for p in X_train_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)

# Get combined numpy array from the populated list
Xtrain, out_transform = merge(raster_to_mosiac)
print(Xtrain.shape)

# Replication for train, validation, test files on features and labels
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
X_train = Xtrain[:, Xtrain[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
y_train = ytrain[:, ytrain[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
y_train =y_train.astype(int) #change the label to integer
X_train = np.transpose(X_train) #flip the array
y_train = np.transpose(y_train) #flip the array

# Replication of the above for validation & test data
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
df_train = pd.DataFrame(X_train,columns=covariate_features)  # covariate features
df_train['labels'] = y_train
df_train['type']= 'train'

df_val = pd.DataFrame(X_val,columns=covariate_features)  # covariate features
df_val['labels'] = y_val
df_val['type']= 'validation'

df_test = pd.DataFrame(X_test,columns=covariate_features)  # convariate features
df_test['labels'] = y_test
df_test['type'] = 'test'

lagos_df = pd.concat([df_train,df_val,df_test]) #Combine train, validation & test
# Export to parquet
lagos_df.to_parquet('lagos_covariate_df.parquet.gzip',compression='gzip')