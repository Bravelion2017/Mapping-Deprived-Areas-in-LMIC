# Packages importing
#%%
import numpy as np
import pandas as pd
import pickle
import tkinter
import matplotlib
matplotlib.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.plot import show
import seaborn as sns
# import plotly.express as px # Use if you have installed plotly
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
from toolbox import get_train_val_ps, important_features, contextual_features, tif_to_df2, cm,covariate_features, important_55,tif_to_df3
random_seed=123
target_names = ['Not Deprived', 'Deprived']
rgbn_f=['r','g','b','n']
covariate_features=covariate_features()
contextual_features= contextual_features()
important_55= important_55()
#Helper
def plot_pred_area(nrow,ncol,pred):
    pred_map = pred.reshape(nrow, ncol)
    values = np.unique(pred_map.ravel())
    im = plt.imshow(pred_map, interpolation='none')
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Class {l}".format(l=values[i])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    # plt.savefig('whole_nairobi.png', dpi=5000)
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
base= "C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi_whole/big_tif"
ctx= [base + "/nairobi_contextual_100m.tif"]
cov= [base + "/nai_covariate_compilation_53bands.tif"]
rgbn= [base + "/nairobi_rgbn_100m.tif"]

contextual_df= tif_to_df3(ctx,contextual_features)
covariate_df= tif_to_df3(cov,covariate_features)
rgbn_df= tif_to_df3(rgbn,rgbn_f)
rgbn_df.fillna(0,inplace=True)

print(f'Shapes:\n {contextual_df.shape, covariate_df.shape, rgbn_df.shape}')

# nairobi= pd.concat([rgbn_df,covariate_df.iloc[:-2,:],contextual_df.iloc[:-2,:]],axis=1)
nairobi= pd.concat([rgbn_df,covariate_df,contextual_df],axis=1)
nairobi_copy= nairobi.loc[:,nairobi.columns.isin(important_55)].copy(deep=True)
nairobi_copy= nairobi_copy[~(nairobi_copy==-9999.000000)]
# Check for Nulls $ replace with mean
print(nairobi_copy.isna().sum().sort_values(ascending=False))
nairobi_copy.fillna(0,inplace=True)
print(nairobi_copy.isna().sum().any())
#c.dropna(inplace=True)

#%%
from wordcloud import WordCloud
WordCloud()
words = " ".join([sent for sent in covariate_features ])
wordcloud = WordCloud(width=1000,
                      height=500,
                      background_color='skyblue',
                      max_words = 144).generate(words)

plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.title("Covariate Features")
plt.show()

#%%
c=nairobi.loc[:,nairobi.columns.isin(important_55)].copy(deep=True)
# c= nairobi.copy(deep=True)
c=c[~(c==-9999.0)]
c_clean= c.dropna(how='all').copy() #Drops any row with all null values
c_clean.fillna(0, inplace=True)
print(f"Null values: {c_clean.isna().sum().any()}")
c['labels']= np.nan

#%%
# Scale features
mms= MinMaxScaler()
x_test = mms.fit_transform(c_clean.values)

#%%
# Prediction
with open("C:/Users/oseme/Desktop/Capstone/models/model_rfce_55.pickle", 'rb') as f:
    rfc= pickle.load(f)
# with open("C:/Users/oseme/Desktop/Capstone/models/model_lre_55.pickle", 'rb') as f:
#     lr= pickle.load(f)
# with open("C:/Users/oseme/Desktop/Capstone/models/model_xgbce_55.pickle", 'rb') as f:
#     xgbc= pickle.load(f)

pred= rfc.predict(x_test)

# rgbn_df['labels']= pred
c_clean['labels']=pred
c.loc[c_clean.index,'labels']=c_clean.labels.astype(int)
rgbn_df['labels']= c.labels

#%%
# Plots
# src = rasterio.open(rgbn[0])
# with rasterio.open('out_rgbn_pred.tif', 'w', **rgbn_df) as f:
#     f.write(src.read())


#Plots
img= rasterio.open(rgbn[0]).read(1)
show(img)
# plot_pred_area(544,805,pred)
plot_pred_area(739,654,c.labels.values)
#%%
# fig = px.imshow(c.labels.values.reshape(739,654),title=f'City of Nairobi, Kenya Prediction')
# fig.show(renderer='browser')
#%%
# Save to GeoTiff
import datetime

T = datetime.datetime.now()
time = T.strftime("%y%m%d")

filename = 'nairobi_xgb55'
out_file = f"C:/Users/oseme/Desktop/Capstone/results_local/{filename}_{time}.tif"
fp ='C:/Users/oseme/Desktop/Capstone/data_Ryan/nairobi_whole/big_tif/nairobi_rgbn_100m.tif'
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
        dst.write_band(flag, rgbn_df[column].values.reshape(739, 654))
        flag +=1