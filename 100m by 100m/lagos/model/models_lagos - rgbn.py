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
from toolbox import get_train_val_ps, important_features, contextual_features, tif_to_df, cm,covariate_features
random_seed=123
target_names = ['Not Deprived', 'Deprived']
rgbn=['r','g','b','n']
covariate_features=covariate_features()
contextual_features= contextual_features()

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
# Load Data into Pandas Dataframe
lagos= pd.read_parquet('lagos_rgbn100m_df.parquet.gzip')
print(f'Null values: {lagos.isna().sum().any()}')

# Split Train, Validation & Test
train= lagos[lagos.type=='train'].copy(deep=True).drop('type',axis=1)
test= lagos[lagos.type=='test'].copy(deep=True).drop('type',axis=1)
validation= train.iloc[792:,:]
train= train.iloc[:792,:]

#%%
# Logistic Regression | Random Forest Classifier | XGBoost Classifier
# Get data in numpy array
# Split into features and target
x_train, y_train= train.iloc[:,:-1].values, train.iloc[:,-1].values
x_val, y_val= validation.iloc[:,:-1].values, validation.iloc[:,-1].values
x_test, y_test= test.iloc[:,:-1].values, test.iloc[:,-1].values
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
with open('model_rfc_4.pickle', 'rb') as f:
    rfc= pickle.load(f)
with open('model_xgbc_4.pickle', 'rb') as f:
    xgbc= pickle.load(f)
with open('model_lr_4.pickle', 'rb') as f:
    lr= pickle.load(f)


#%%
# Logistic was best
y_pred_rfc = rfc.predict(x_test) # predict test set
y_pred_lr = lr.predict(x_test) # predict test set
y_pred_xg = xgbc.predict(x_test) # predict test set

#%%
# Ensembling 3 models
rez= pd.DataFrame()
rez['res1']=np.array(y_pred_lr)
rez['res2'] = np.array(y_pred_xg)
rez['res3'] = np.array(y_pred_rfc)
rez_final= rez.mode(axis=1).values
rez_final= rez_final.ravel()
#%%
# y_test_pred= pd.Series(xgbc.predict_proba(x_test)[:,1]).apply(lambda x:0 if x>0.41 else 1).values #41
confused_matrix(y_test, rez_final) # Confusion matrix
print(classification_report(y_test, rez_final, target_names=target_names))

#%%
# FOR 1 MODEL AT A TIME (Logistic regression)
for i in [11,13,20,23,25,28]:
    area0= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask= [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    # area0_pred = pd.Series(rfc.predict_proba(mms.transform(area0_df.drop('labels',
    #              axis=1).values))[:,1]).apply(lambda x:1 if x>0.41 else 0).values #41)
    area0_pred = lr.predict(mms.transform(area0_df.drop('labels', axis=1).values))
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
    confused_matrix(area0_df.labels.values, area0_pred)

#%%
# ENSEMBLE MODEL
for i in [11,13,20,23,25,28]:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    #area0_df.loc[:,area0_df.columns.isin(important_37)]
    # area0_pred1= lr.predict(mms.transform(area0_df.loc[:,area0_df.columns.isin(important_3)].drop('labels',axis=1).values))
    # area0_pred2= rfc.predict(mms.transform(area0_df.loc[:,area0_df.columns.isin(important_3)].drop('labels',axis=1).values))
    # area0_pred3= xgbc.predict(mms.transform(area0_df.loc[:,area0_df.columns.isin(important_3)].drop('labels',axis=1).values))
    area0_pred1 = lr.predict(mms.transform(area0_df.drop('labels', axis=1).values))
    area0_pred2 = rfc.predict(mms.transform(area0_df.drop('labels', axis=1).values))
    area0_pred3 = xgbc.predict(mms.transform(area0_df.drop('labels', axis=1).values))
    reze= pd.DataFrame()
    reze['res1']= np.array(area0_pred1)
    reze['res2'] = np.array(area0_pred2)
    reze['res3'] = np.array(area0_pred3)
    rez_finale= reze.mode(axis=1).values
    rez_finale= rez_finale.ravel()
    # area0_pred= lr.predict(mms.transform(area0_df.drop(['labels','pantex_sc3_min'],axis=1).values))

    #plot original area
    img= rasterio.open(area0_mask[0]).read(1)
    show(img)
    #Plot predicted area
    plot_pred_area(6,6,rez_finale)
    #Plot confusion matrix
    confused_matrix(area0_df.labels.values, rez_finale)