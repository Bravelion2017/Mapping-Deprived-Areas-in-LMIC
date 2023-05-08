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
lagos= pd.read_parquet('lagos_cv100m_df.parquet.gzip')
print(f'Null values: {lagos.isna().sum().any()}')

# Split Train, Validation & Test
train= lagos[lagos.type=='train'].copy(deep=True).drop('type',axis=1)
test= lagos[lagos.type=='test'].copy(deep=True).drop('type',axis=1)
validation= train.iloc[792:,:]
train= train.iloc[:792,:]
#%%
# Get important features for contextual features from PCA and VIF
from toolbox import  important_38, important_62, important_104,important_96
important_104= important_104()
important_96=important_96()
important_38= important_38()
important_62= important_62()
important_104.append('labels')
important_96.append('labels')
important_38.append('labels')
important_62.append('labels')

#%%
# Get the important 96 features
train_c= train.loc[:,important_96].copy()
validation_c= validation.loc[:,important_96].copy()
test_c= test.loc[:,important_96].copy()

#%%
# Logistic Regression | Random Forest Classifier | XGBoost Classifier
# Get data in numpy array
# Split into features and target
# x_train, y_train= train_c.iloc[:,:-1].values, train_c.iloc[:,-1].values
# x_val, y_val= validation_c.iloc[:,:-1].values, validation_c.iloc[:,-1].values
# x_test, y_test= test_c.iloc[:,:-1].values, test_c.iloc[:,-1].values
# print(x_train.shape,x_val.shape,x_test.shape)

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

with open('model_lr_53.pickle', 'rb') as f:
    lr= pickle.load(f)


#%%
y_pred_lr = lr.predict(x_test) # predict test set

#%%
y_test_pred= pd.Series(lr.predict_proba(x_test)[:,1]).apply(lambda x:0 if x>0.41 else 1).values #41
confused_matrix(y_test, y_pred_lr) # Confusion matrix
print(classification_report(y_test, y_pred_lr, target_names=target_names))

#%%
# FOR 1 MODEL AT A TIME
for i in [11,13,20,23,25,28]:
    area0= [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif"]
    area0_mask= [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,covariate_features)
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