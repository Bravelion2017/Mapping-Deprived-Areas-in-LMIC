# Packages importing
#%%
import numpy as np
import pandas as pd
import pickle
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
from toolbox import get_train_val_ps, important_features, contextual_features, \
    tif_to_df2, cm,covariate_features, important_55,tif_to_df3, plot_pred_area
random_seed=123
target_names = ['Not Deprived', 'Deprived']
rgbn_f=['r','g','b','n']
covariate_features=covariate_features()
contextual_features= contextual_features()
important_55= important_55()
#%%
# Import data
base= "/home/ubuntu/capstone/Lagos/big_tif"
ctx= [base + "/lagos_contextual_100m.tif"]
cov= [base + "/lag_covariate_compilation_53bands.tif"]
rgbn= [base + "/lagos_rgbn_100m.tif"]

contextual_df= tif_to_df3(ctx,contextual_features)
covariate_df= tif_to_df3(cov,covariate_features)
rgbn_df= tif_to_df3(rgbn,rgbn_f)

print(f'Shapes:\n {contextual_df.shape, covariate_df.shape, rgbn_df.shape}')

# concatenate the three datasets
lagos= pd.concat([rgbn_df,covariate_df,contextual_df],axis=1)
#%%
# c is the copy of lagos
c=lagos.loc[:,lagos.columns.isin(important_55)].copy(deep=True) # get only the important 55 features
c=c[~(c==-9999.000000)]
c_clean= c.dropna(how='all').copy() #Drops any row with all null values
c_clean.fillna(0, inplace=True) #fill remaining nan with 0
print(f"Null values: {c_clean.isna().sum().any()}")
c['labels']= np.nan #pre-fill the original data with nan for labels (for wiped out rows during clean-up)

#%%
# Scale features
mms= MinMaxScaler()
x_test = mms.fit_transform(c_clean.values)

#%%
# Prediction

# Import saved model
with open('model_rfce_55.pickle', 'rb') as f:
    rfc55= pickle.load(f)

pred= rfc55.predict(x_test) #prediction

# returning predictions to main file
c_clean['labels']=pred
c.loc[c_clean.index,'labels']=c_clean.labels.astype(int)
rgbn_df['labels']= c.labels
#%%
# Plots

#plot whole city
img= rasterio.open(rgbn[0]).read(1)
show(img)

# plot predictions of city
# plot_pred_area(544,805,pred)
plot_pred_area(544,805,c.labels.values)
#%%
# fig = px.imshow(c.labels.values.reshape(544,805))
# fig.show(renderer='browser')
#%%
# Save to GeoTiff
import datetime

T = datetime.datetime.now()
time = T.strftime("%y%m%d")

filename = 'lagos_rfc55'
out_file = f"/home/ubuntu/capstone/Lagos/results/{filename}_{time}.tif"
fp ='/home/ubuntu/capstone/Lagos/big_tif/lagos_rgbn_100m.tif'
with rasterio.open(fp, mode="r") as src:
    out_profile = src.profile.copy()
    out_profile.update(count=5,
                       nodata=-9999,
                       dtype='float32',
                       width=src.width,
                       height=src.height,
                       crs=src.crs)

# open in 'write' mode, unpack profile info to dst
flag=1
with rasterio.open(out_file,
                   'w', **out_profile) as dst:
    # dst.write_band(1, c.labels.values.reshape(544,805))
    for column in rgbn_df.columns:
        dst.write_band(flag, rgbn_df[column].values.reshape(544, 805))
        flag +=1