# Packages importing
import numpy as np
import pandas as pd
import tkinter
import matplotlib
matplotlib.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
import seaborn as sns
import scipy, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Helper function
def calculate_vif(df, features):
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)

        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1 / (tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


# Load Data into Pandas Dataframe
nairobi= pd.read_parquet('nairobi_df.parquet.gzip')

# Split Train, Validation & Test
train= nairobi[nairobi.type=='train'].copy(deep=True).drop('type',axis=1)
validation= nairobi[nairobi.type=='validation'].copy(deep=True).drop('type',axis=1)
test= nairobi[nairobi.type=='test'].copy(deep=True).drop('type',axis=1)

# SVD

# Get the features into a numpy array
X = nairobi.copy(deep=True).drop(['labels','type'],axis=1)
# computing singular values using numpy
H= X.values.T @ X.values
_,d,_= np.linalg.svd(H)
res=pd.DataFrame(d,index=X.columns, columns=['Singular Values'])
print(tabulate(res,headers='keys',tablefmt="fancy_grid"))
# compute condition number
condition=np.linalg.cond(X)
condition_df=pd.DataFrame(data=[condition],columns=['Condition Number'])
print(tabulate(condition_df,headers='keys',tablefmt="fancy_grid"))

# PCA Analysis

#scale features
sc=StandardScaler()
# data_scaled= sc.fit_transform(X)
data_scaled= pd.DataFrame(sc.fit_transform(X),columns = X.columns)
pca=PCA(n_components='mle',svd_solver='full') #Initialize PCA
transformed= pca.fit_transform(data_scaled)
n_pcs= pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names= X.columns.to_list()
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
important= pd.DataFrame(dic.items(),columns=['PCs','features'])
most_important_features= important.features.unique()
# pd.DataFrame(pca.components_,columns=data_scaled.columns)

# Plot explained variance
plt.figure(figsize=(25,10))
x=np.arange(1,len(pca.explained_variance_ratio_)+1)
plt.xticks(x, fontsize=6,rotation=90)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_),c='red',marker='*')
plt.axvline(x = 72, color = 'b', label = 'axvline - full height')
plt.axvline(x = 105, color = 'b', label = 'axvline - full height')
plt.axhline(y = 1, color = 'g', linestyle = '-')
plt.grid()
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
# plt.savefig('pca.png')
plt.show()

pca_df= pd.DataFrame(np.array([np.array(x,dtype=int),np.cumsum(pca.explained_variance_ratio_)]),
                     index=['n_features','variance']).T
print(f'With half the feature size, you get {round(pca_df.head(72).iloc[-1,-1],4)*100}% variance explained')
print(f'With 105  feature size, you get {round(pca_df.head(105).iloc[-1,-1],4)*100}% variance explained')

# Check for collinearity again (2)
X2= X.loc[:,X.columns.isin(most_important_features)]
# computing singular values using numpy
H2= X2.values.T @ X2.values
_,d2,_= np.linalg.svd(H2)
res2 = pd.DataFrame(d2,index=X2.columns, columns=['Singular Values'])
print(tabulate(res2,headers='keys',tablefmt="fancy_grid"))
# compute condition number
condition2=np.linalg.cond(X2)
# condition number reduces by half

# Calculate VIF (variance inflation factor)
# 1- features are not correlated
# 1<vif<5 features moderately correlated
# vif>5- features are highly correlated
VIF= calculate_vif(df=X,features=list(X.columns))
# Get features with moderate correlation
vif_features= VIF[(VIF.VIF<5)][(VIF.VIF>=1)]
moderate_cor_features= vif_features.index

# SVD check (3) on moderate VIF
X3= X.loc[:,X.columns.isin(moderate_cor_features)]
# computing singular values using numpy
H3= X3.values.T @ X3.values
_,d3,_= np.linalg.svd(H3)
res3 = pd.DataFrame(d3,index=X3.columns, columns=['Singular Values'])
print(tabulate(res3,headers='keys',tablefmt="fancy_grid"))
# compute condition number
condition3=np.linalg.cond(X3)
# VIF does not reduce the multi-collinearity

#=============== K-Means ===============
k_features = X.copy(deep=True).reset_index().drop('index',axis=1)
# Removing outliers (Z-Score)
zscore = np.abs(stats.zscore(k_features))
threshold = 3
mask = (zscore <= threshold).all(axis=1)
cleaned_features= k_features[mask]

print(f'Missing Values:{cleaned_features.isna().sum().any()}')

# Scale features
mms =MinMaxScaler()
features_scaled= pd.DataFrame(mms.fit_transform(cleaned_features),
                              columns = cleaned_features.columns,
                              index=cleaned_features.index)
#Choosing best k
ks=list(range(1,10))
sse=[]
for k in ks:
    km=KMeans(n_clusters=k)
    km.fit(features_scaled.values)
    sse.append(km.inertia_)
plt.plot(ks,sse,'o-')
plt.ylabel("SSE")
plt.xlabel("K")
plt.show()

#Choose the best k using the ELBOW METHOD
km=KMeans(n_clusters=2)# choose your k from above graph elbow
km.fit(features_scaled.values)
y_pred=km.predict(features_scaled.values)
