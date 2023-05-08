# Packages importing
#%%
import numpy as np
import pandas as pd
import pickle
import tkinter
import matplotlib
# matplotlib.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.plot import show
import seaborn as sns
import scipy, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from toolbox import get_train_val_ps, important_features, contextual_features, tif_to_df, cm,covariate_features, important_136
random_seed=123
target_names = ['Not Deprived', 'Deprived']
rgbn=['r','g','b','n']
covariate_features=covariate_features()
contextual_features= contextual_features()
important_136= important_136()
#Helper
def plot_pred_area(nrow,ncol,pred):
    pred_map = pred.reshape(nrow, ncol)
    values = np.unique(pred_map.ravel())
    im = plt.imshow(pred_map, interpolation='none')
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Class {l}".format(l=values[i])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def confused_matrix(y_test,y_pred):
    cm=confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
    ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
    plt.show()
#%%
best_55=['uu_bld_count_2020',
 'ph_dist_inland_water_2018',
 'ses_child_stuned_2014',
 'ses_m_lit_2014',
 'uu_bld_den_2020',
 'gabor_sc7_filter_11',
 'lbpm_sc7_mean',
 'ph_dist_open_coast_2020',
 'ndvi_sc5_variance',
 'ndvi_sc7_variance',
 'ph_land_c2_2020',
 'gabor_sc3_filter_10',
 'gabor_sc3_filter_11',
 'ph_dist_cultivated_2015',
 'in_night_light_2016',
 'ph_dist_shrub_2015',
 'fourier_sc71_mean',
 'lbpm_sc5_mean',
 'sfs_sc71_std',
 'ph_ndvi_2019',
 'n',
 'po_hrsl_2018',
 'ses_unmet_need_2014',
 'ses_pfpr_2017',
 'in_dist_waterway_2016',
 'r',
 'ph_dist_aq_veg_2015',
 'ph_dist_art_surface_2015',
 'uu_urb_bldg_2018',
 'ph_pm25_2016',
 'ph_base_water_2010',
 'sfs_sc31_max_line_length',
 'ses_preg_2017',
 'ph_dist_riv_network_2007',
 'pantex_sc3_min',
 'b',
 'hog_sc7_kurtosis',
 'hog_sc3_mean',
 'fs_dist_school_2020',
 'hog_sc7_skew','fs_dist_fs_2020',
 'sh_ethno_den_2020',
 'hog_sc7_max',
 'po_wp_2020',
 'orb_sc51_max',
 'fourier_sc31_variance',
 'lbpm_sc3_mean',
 'sfs_sc31_std',
 'sfs_sc51_mean',
 'lbpm_sc3_kurtosis',
 'ph_grd_water_2000',
 'fourier_sc71_variance',
 'lsr_sc31_line_mean',
 'sfs_sc51_std',
 'hog_sc3_skew',
'labels']
#%%
# Load Data into Pandas Dataframe
lagos1= pd.read_parquet('lagos_cv100m_df.parquet.gzip').drop(['labels','type'],axis=1)
print(f'Null values: {lagos1.isna().sum().any()}')
lagos2= pd.read_parquet('lagos_ctx100m_df.parquet.gzip')
print(f'Null values: {lagos2.isna().sum().any()}')
lagos3= pd.read_parquet('lagos_rgbn100m_df.parquet.gzip').drop(['labels','type'],axis=1)
print(f'Null values: {lagos3.isna().sum().any()}')

#Concatenate the 3-feature dataset
lagos= pd.concat([lagos3,lagos1,lagos2],axis=1)

# Split Train, Validation & Test
train= lagos[lagos.type=='train'].copy(deep=True).drop('type',axis=1)
test= lagos[lagos.type=='test'].copy(deep=True).drop('type',axis=1)
validation= train.iloc[792:,:]
train= train.iloc[:792,:]

#Null imputer
values = {i: lagos[i].mean() for i in lagos.columns[:-2]}
#%%
# Logistic Regression | Random Forest Classifier | XGBoost Classifier
# Get data in numpy array
# Split into features and target
train_c= train.loc[:,train.columns.isin(np.append(best_55,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(best_55,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(best_55,'labels'))]

x_train, y_train= train_c.iloc[:,:-1].values, train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1].values, val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1].values, test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

#%%
# Scale features
# with open('mms104.pickle', 'rb') as f:
#     mms= pickle.load(f)
mms= MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

#%%
# Models (based on Random Search CV)
# loading model

# with open('model_xgbce_55.pickle', 'rb') as f:
#     xgbc= pickle.load(f)
# with open('model_lre_55.pickle', 'rb') as f:
#     lr= pickle.load(f)
# with open('model_rfce_55.pickle', 'rb') as f:
#     rfc = pickle.load(f)
with open('model_rfce_55.pickle', 'rb') as f:
    rfc55= pickle.load(f)

#%%
# Predictions
# y_pred_rfc = rfc.predict(x_test) # predict test set
# y_pred_lr = lr.predict(x_test) # predict test set
# y_pred_xg = xgbc.predict(x_test) # predict test set
y_pred_rfc55 = rfc55.predict(x_test) # predict test set
#%%
# Ensembling 3 models
# rez= pd.DataFrame()
# rez['res1']=np.array(y_pred_lr)
# rez['res2'] = np.array(y_pred_xg)
# rez['res3'] = np.array(y_pred_rfc)
# rez_final= rez.mode(axis=1).values
# rez_final= rez_final.ravel()
#%%
# Classification Report
# y_test_pred= pd.Series(xgbc.predict_proba(x_test)[:,1]).apply(lambda x:0 if x>0.41 else 1).values #41
confused_matrix(y_test, y_pred_rfc55) # Confusion matrix
print(classification_report(y_test, y_pred_rfc55, target_names=target_names))

#%%
with open('model_lre_201.pickle', 'rb') as f:
    lr= pickle.load(f)

#%%
# FOR 1 MODEL AT A TIME (201 features)
for i in [11,13,20,23,25,28]:
    area0= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    area0_df.drop(['labels'],axis=1,inplace=True)

    area00 = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/spfea_100m/lag_area{i}.tif"]
    area00_mask = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/mask_100m/lag_area{i}.tif"]
    area00_df = tif_to_df(area00, area00_mask, contextual_features)

    area000 = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif"]
    area000_mask = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif"]
    area000_df = tif_to_df(area000, area000_mask, covariate_features)
    area000_df.drop(['labels'],axis=1,inplace=True)

    area_df= pd.concat([area0_df,area000_df,area00_df],axis=1)
    area_df = area_df[~(area_df == -9999.000000)]
    area_df.loc[:, area_df.columns[:-1]] = area_df.loc[:, area_df.columns[:-1]].fillna(value=values)
    # area0_pred = pd.Series(rfc.predict_proba(mms.transform(area0_df.drop('labels',
    #              axis=1).values))[:,1]).apply(lambda x:1 if x>0.41 else 0).values #41)
    area0_pred = lr.predict(mms.transform(area_df.drop('labels', axis=1).values))
    # area0_pred= rfc.predict(mms.transform(area0_df.loc[:,area0_df.columns.isin(important_104)].drop('labels',axis=1).values))
    # area0_pred = pd.Series(rfc.predict_proba(mms.transform(area0_df.loc[:,
    #                         area0_df.columns.isin(contextual_features)].drop('labels',
    #                         axis=1).values))[:,1]).apply(lambda x:0 if x>0.5 else 1).values #41)

    #plot original area
    img= rasterio.open(area0_mask[0]).read(1)
    show(img)
    #Plot predicted area
    plot_pred_area(6,6,area0_pred)
    #Plot confusion matrix
    confused_matrix(area_df.labels.values, area0_pred)
#%%
with open('model_lre_136.pickle', 'rb') as f:
    lr136= pickle.load(f)

#%%
# FOR 1 MODEL AT A TIME (136 features)
mms= MinMaxScaler()
for i in [11,13,20,23,25,28]:
    area0= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    area0_df.drop(['labels'],axis=1,inplace=True)

    area00 = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/spfea_100m/lag_area{i}.tif"]
    area00_mask = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/mask_100m/lag_area{i}.tif"]
    area00_df = tif_to_df(area00, area00_mask, contextual_features)

    area000 = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif"]
    area000_mask = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif"]
    area000_df = tif_to_df(area000, area000_mask, covariate_features)
    area000_df.drop(['labels'],axis=1,inplace=True)

    area_df= pd.concat([area0_df,area000_df,area00_df],axis=1)
    area_df = area_df[~(area_df == -9999.000000)]
    area_df.loc[:, area_df.columns[:-1]] = area_df.loc[:, area_df.columns[:-1]].fillna(value=values)
    # area0_pred = pd.Series(lr136.predict_proba(mms.transform(area0_df.drop('labels',
    #              axis=1).values))[:,1]).apply(lambda x:1 if x>0.41 else 0).values #41)
    # area0_pred = lr136.predict(mms.transform(area_df.drop('labels', axis=1).values))
    area0_pred= lr136.predict(mms.fit_transform(area_df.loc[:,area_df.columns.isin(important_136)].values))
    # area0_pred = pd.Series(lr136.predict_proba(mms.transform(area0_df.loc[:,
    #                         area0_df.columns.isin(contextual_features)].drop('labels',
    #                         axis=1).values))[:,1]).apply(lambda x:0 if x>0.5 else 1).values #41)

    #plot original area
    img= rasterio.open(area0_mask[0]).read(1)
    show(img)
    #Plot predicted area
    plot_pred_area(6,6,area0_pred)
    #Plot confusion matrix
    confused_matrix(area_df.labels.values, area0_pred)

#%%
# FOR 1 MODEL AT A TIME (55 features)
mms= MinMaxScaler()
for i in [11,13,20,23,25,28]:
    area0= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    area0_df.drop(['labels'],axis=1,inplace=True)

    area00 = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/spfea_100m/lag_area{i}.tif"]
    area00_mask = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/mask_100m/lag_area{i}.tif"]
    area00_df = tif_to_df(area00, area00_mask, contextual_features)

    area000 = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif"]
    area000_mask = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif"]
    area000_df = tif_to_df(area000, area000_mask, covariate_features)
    labels_copy= area000_df['labels'].values.copy()
    area000_df.drop(['labels'],axis=1,inplace=True)

    area_df= pd.concat([area0_df,area000_df,area00_df],axis=1)
    area_df = area_df[~(area_df == -9999.000000)]
    area_df.loc[:, area_df.columns[:-1]] = area_df.loc[:, area_df.columns[:-1]].fillna(value=values)
    # area0_pred = pd.Series(lr136.predict_proba(mms.transform(area0_df.drop('labels',
    #              axis=1).values))[:,1]).apply(lambda x:1 if x>0.41 else 0).values #41)
    # area0_pred = lr136.predict(mms.transform(area_df.drop('labels', axis=1).values))
    area0_pred= rfc55.predict(mms.fit_transform(area_df.loc[:,area_df.columns.isin(best_55)].drop(['labels'],axis=1).values))
    # area0_pred = pd.Series(lr136.predict_proba(mms.transform(area0_df.loc[:,
    #                         area0_df.columns.isin(contextual_features)].drop('labels',
    #                         axis=1).values))[:,1]).apply(lambda x:0 if x>0.5 else 1).values #41)

    #plot original area
    img= rasterio.open(area0_mask[0]).read(1)
    show(img)
    #Plot predicted area
    plot_pred_area(6,6,area0_pred)
    #Plot confusion matrix
    confused_matrix(area_df.labels.values, area0_pred)
    print(classification_report(area_df.labels.values, area0_pred, target_names=target_names))


    # Save to GeoTiff
    import datetime

    T = datetime.datetime.now()
    time = T.strftime("%y%m%d")

    filename = 'lagos_rfc55'
    out_file = f"/home/ubuntu/capstone/areas/{i}_{time}.tif"
    fp =area0_mask[0]
    with rasterio.open(fp, mode="r") as src:
        out_profile = src.profile.copy()
        out_profile.update(count=1,
                           nodata=-9999,
                           dtype='float32',
                           width=src.width,
                           height=src.height,
                           crs=src.crs)

    # open in 'write' mode, unpack profile info to dst
    with rasterio.open(out_file,
                       'w', **out_profile) as dst:
        # dst.write_band(1, c.labels.values.reshape(544,805))
        dst.write_band(1, area0_pred.reshape(6, 6))