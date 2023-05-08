# Packages importing
#%%
import numpy as np
import pandas as pd
import tkinter
import matplotlib
# matplotlib.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
import seaborn as sns
import scipy, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from toolbox import get_train_val_ps, important_features
random_seed=123
#%%
# Helper function
imp_f= important_features()
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

#%%
# Load Data into Pandas Dataframe
nairobi= pd.read_parquet('nairobi_df.parquet.gzip')
nairobi= nairobi[~(nairobi==-9999.000000)]

# Check for Nulls $ replace with zero
print(nairobi.isna().sum().sort_values(ascending=False))
nairobi.drop('pantex_sc3_min',axis=1,inplace=True) #Drop most missing value column
# nairobi.fillna(0,inplace=True)
# Replace Nulls with mean values of respective columns
values= {i:nairobi[i].mean() for i in nairobi.columns[:-2]}
nairobi.loc[:,nairobi.columns[:-2]]= nairobi.loc[:,nairobi.columns[:-2]].fillna(value=values)


# Split Train, Validation & Test
train= nairobi[nairobi.type=='train'].copy(deep=True).drop('type',axis=1)
validation= nairobi[nairobi.type=='validation'].copy(deep=True).drop('type',axis=1)
test= nairobi[nairobi.type=='test'].copy(deep=True).drop('type',axis=1)
print(f'Null values: {nairobi.isna().sum().any()}')

#%%
# SVD

# Get the features into a numpy array
X = nairobi.copy(deep=True).drop(['labels','type'],axis=1)
# computing singular values using numpy
H= X.values.T @ X.values
_,d,_= np.linalg.svd(H)
res=pd.DataFrame(d,index=X.columns, columns=['Singular Values'])
# print(tabulate(res,headers='keys',tablefmt="fancy_grid"))
print(tabulate(res.tail(10),headers='keys',tablefmt="fancy_grid"))
# compute condition number
condition=np.linalg.cond(X)
condition_df=pd.DataFrame(data=[condition],columns=['Condition Number'])
print(tabulate(condition_df,headers='keys',tablefmt="fancy_grid"))

#%%
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
important_68= important.head(105).features.unique()
important_120= important.head(120).features.unique()
important_97= important.features.unique()
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
print(f'With 108  feature size, you get {round(pca_df.head(108).iloc[-1,-1],4)*100}% variance explained')
print(f'With 120  feature size, you get {round(pca_df.head(120).iloc[-1,-1],4)*100}% variance explained')

#%%
# Trying SVD again with most important 68 features
X1= X.loc[:,X.columns.isin(important_68)]
# computing singular values using numpy
H1= X1.values.T @ X1.values
_,d1,_= np.linalg.svd(H1)
res1=pd.DataFrame(d1,index=X1.columns, columns=['Singular Values'])
print(tabulate(res1,headers='keys',tablefmt="fancy_grid"))
condition1=np.linalg.cond(X1)
condition_df1=pd.DataFrame(data=[condition1],columns=['Condition Number'])
print(tabulate(condition_df1,headers='keys',tablefmt="fancy_grid"))
#The Condition number reduced by more than half

#%%
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
condition_df3=pd.DataFrame(data=[condition3],columns=['Condition Number'])
print(tabulate(condition_df3,headers='keys',tablefmt="fancy_grid"))
# VIF does not reduce the multi-collinearity

#%%
#=============== K-Means ===============
k_features_w_label= train.copy(deep=True).reset_index().drop(['index'],axis=1)
k_features_w_label_test= test.copy(deep=True).reset_index().drop(['index'],axis=1)
k_features = train.copy(deep=True).reset_index().drop(['index','labels'],axis=1)
k_features_test = test.copy(deep=True).reset_index().drop(['index','labels'],axis=1)
k_features_test= k_features_test.loc[:,k_features_test.columns.isin(important_68)]
k_features= k_features.loc[:,k_features.columns.isin(important_68)]
# Removing outliers (Z-Score)
zscore = np.abs(stats.zscore(k_features))
threshold = 3
mask = (zscore <= threshold).all(axis=1)
cleaned_features= k_features[mask]

print(f'Missing Values:{cleaned_features.isna().sum().any()}')

# Scale features
mms =MinMaxScaler()
mms2 =MinMaxScaler()
features_scaled= pd.DataFrame(mms.fit_transform(cleaned_features),
                              columns = cleaned_features.columns,
                              index=cleaned_features.index)
features_scaled_test= pd.DataFrame(mms2.fit_transform(k_features_test),
                              columns = k_features_test.columns,
                              index=k_features_test.index)
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
y_pred_test=km.predict(features_scaled_test.values)

#Add labels from kmeans
cleaned_features['pred']= y_pred
k_features_test['pred']= y_pred_test

# Merge original labels based on index
cleaned_features['label'] =k_features_w_label.filter(items = cleaned_features.index, axis=0)['labels']
k_features_test['label'] =k_features_w_label_test.filter(items = k_features_test.index, axis=0)['labels']

from sklearn.metrics import confusion_matrix, accuracy_score

# cleaned_features['pred']= cleaned_features['pred'].apply(lambda x:1 if x==0 else 0)
# k_features_test['pred']= k_features_test['pred'].apply(lambda x:1 if x==0 else 0)
cm=confusion_matrix(cleaned_features['label'], cleaned_features['pred'])
cm2=confusion_matrix(k_features_test['label'], k_features_test['pred'])
ax= plt.subplot()
sns.heatmap(cm2, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
plt.show()

#Classification Report
from sklearn.metrics import classification_report
target_names = ['Not Deprived', 'Deprived']
print(classification_report(k_features_test['label'], k_features_test['pred'], target_names=target_names))

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (68 FEATURES)
train_c= train.loc[:,train.columns.isin(np.append(important_68,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(important_68,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(important_68,'labels'))]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()

# Augmenting Minority Target Variabe
from imblearn.over_sampling import RandomOverSampler
# The RandomOverSampler
ros = RandomOverSampler(random_state=random_seed)
# Augment the training data
X_ros_train, y_ros_train = ros.fit_resample(x_train, y_train)

# Target Balance Re-Check
sns.barplot(x=pd.Series(y_ros_train).value_counts().index,y=pd.Series(y_ros_train).value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()


# models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          'xgbc':XGBClassifier(random_state=random_seed)}
# Scale features
mms =MinMaxScaler()
# Standardize the training, val & test feature data
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

# Creating Dictionary for Pipeline
from sklearn.pipeline import Pipeline
pipes={}
for acronym, model in models.items():
  pipes[acronym]= Pipeline([('model', model)])

# Getting the predefined split cross-validator
# Get the:
# feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit

x_train_val, y_train_val, ps = get_train_val_ps(x_train, y_train, x_val, y_val)

from scipy.stats import uniform

param_dists = {}

# FOR LOGISTIC REGRESSION
# The distribution for tol_grid: a uniform distribution over [loc, loc + scale]
tol_grid = uniform(loc=0.000001, scale=15)

# The distribution for C_grid & solver: a uniform distribution over [loc, loc + scale]
C_grid = uniform(loc=0.001, scale=10)
solver = ['lbfgs', 'sag']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]
# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__n_estimators':n_estimators,
     'model__eval_metric':eval_metric,
     'model__objective':objective}
]

# ============

from sklearn.model_selection import RandomizedSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by RandomizedSearchCV
best_score_params_estimator_rs = []

for acronym in pipes.keys():
    # RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipes[acronym],
                            param_distributions=param_dists[acronym],
                            n_iter=2,
                            scoring='f1_macro',
                            n_jobs=2,
                            cv=ps,
                            random_state=random_seed,
                            return_train_score=True)

    # Fit the pipeline
    rs = rs.fit(x_train_val, y_train_val)

    # Update best_score_param_estimators
    best_score_params_estimator_rs.append([rs.best_score_, rs.best_params_, rs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(rs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(acronym + '68.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)

cm2=confusion_matrix(test_c.labels.values, y_test_pred)
ax= plt.subplot()
sns.heatmap(cm2, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
plt.show()

print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (108 FEATURES)
train_c= train.loc[:,train.columns.isin(np.append(imp_f,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(imp_f,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(imp_f,'labels'))]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()

# Augmenting Minority Target Variabe
from imblearn.over_sampling import RandomOverSampler
# The RandomOverSampler
ros = RandomOverSampler(random_state=random_seed)
# Augment the training data
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

# Target Balance Re-Check
sns.barplot(x=pd.Series(y_ros_train).value_counts().index,y=pd.Series(y_ros_train).value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()


# models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          'xgbc':XGBClassifier(random_state=random_seed)}
# Scale features
mms =MinMaxScaler()
# Standardize the training, val & test feature data
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

# Creating Dictionary for Pipeline
from sklearn.pipeline import Pipeline
pipes={}
for acronym, model in models.items():
  pipes[acronym]= Pipeline([('model', model)])

# Getting the predefined split cross-validator
# Get the:
# feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit

x_train_val, y_train_val, ps = get_train_val_ps(x_train, y_train, x_val, y_val)

from scipy.stats import uniform

param_dists = {}

# FOR LOGISTIC REGRESSION
# The distribution for tol_grid: a uniform distribution over [loc, loc + scale]
tol_grid = uniform(loc=0.000001, scale=15)

# The distribution for C_grid & solver: a uniform distribution over [loc, loc + scale]
C_grid = uniform(loc=0.001, scale=10)
solver = ['lbfgs', 'sag']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]

# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__n_estimators':n_estimators,
     'model__eval_metric':eval_metric,
     'model__objective':objective}
]
# ============

from sklearn.model_selection import RandomizedSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by RandomizedSearchCV
best_score_params_estimator_rs = []

for acronym in pipes.keys():
    # RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipes[acronym],
                            param_distributions=param_dists[acronym],
                            n_iter=2,
                            scoring='f1_macro',
                            n_jobs=2,
                            cv=ps,
                            random_state=random_seed,
                            return_train_score=True)

    # Fit the pipeline
    rs = rs.fit(x_train_val, y_train_val)

    # Update best_score_param_estimators
    best_score_params_estimator_rs.append([rs.best_score_, rs.best_params_, rs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(rs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(acronym + '108.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)

cm3=confusion_matrix(test_c.labels.values, y_test_pred)
ax= plt.subplot()
sns.heatmap(cm3, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
plt.show()

print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (143 FEATURES)
# Split into features and target
x_train, y_train= train.iloc[:,:-1], train.iloc[:,-1].values
x_val, y_val= validation.iloc[:,:-1], validation.iloc[:,-1].values
x_test, y_test= test.iloc[:,:-1], test.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()

# Augmenting Minority Target Variabe
from imblearn.over_sampling import RandomOverSampler
# The RandomOverSampler
ros = RandomOverSampler(random_state=random_seed)
# Augment the training data
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

# Target Balance Re-Check
sns.barplot(x=pd.Series(y_ros_train).value_counts().index,y=pd.Series(y_ros_train).value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          'xgbc':XGBClassifier(random_state=random_seed)}
# Scale features
mms =MinMaxScaler()
# Standardize the training, val & test feature data
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

# Creating Dictionary for Pipeline
from sklearn.pipeline import Pipeline
pipes={}
for acronym, model in models.items():
  pipes[acronym]= Pipeline([('model', model)])

# Getting the predefined split cross-validator
# Get the:
# feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit

x_train_val, y_train_val, ps = get_train_val_ps(x_train, y_train, x_val, y_val)

from scipy.stats import uniform

param_dists = {}

# FOR LOGISTIC REGRESSION
# The distribution for tol_grid: a uniform distribution over [loc, loc + scale]
tol_grid = uniform(loc=0.000001, scale=15)

# The distribution for C_grid & solver: a uniform distribution over [loc, loc + scale]
C_grid = uniform(loc=0.001, scale=10)
solver = ['lbfgs', 'sag']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]

# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__n_estimators':n_estimators,
     'model__eval_metric':eval_metric,
     'model__objective':objective}
]
# ============

from sklearn.model_selection import RandomizedSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by RandomizedSearchCV
best_score_params_estimator_rs = []

for acronym in pipes.keys():
    # RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipes[acronym],
                            param_distributions=param_dists[acronym],
                            n_iter=2,
                            scoring='f1_macro',
                            n_jobs=2,
                            cv=ps,
                            random_state=random_seed,
                            return_train_score=True)

    # Fit the pipeline
    rs = rs.fit(x_train_val, y_train_val)

    # Update best_score_param_estimators
    best_score_params_estimator_rs.append([rs.best_score_, rs.best_params_, rs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(rs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(acronym + '108.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)

cm3=confusion_matrix(test_c.labels.values, y_test_pred)
ax= plt.subplot()
sns.heatmap(cm3, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
plt.show()

print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (81 FEATURES)
train_c= train.loc[:,train.columns.isin(np.append(important_120,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(important_120,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(important_120,'labels'))]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()

# Augmenting Minority Target Variabe
from imblearn.over_sampling import RandomOverSampler
# The RandomOverSampler
ros = RandomOverSampler(random_state=random_seed)
# Augment the training data
X_ros_train, y_ros_train = ros.fit_resample(x_train, y_train)

# Target Balance Re-Check
sns.barplot(x=pd.Series(y_ros_train).value_counts().index,y=pd.Series(y_ros_train).value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()


# models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          'xgbc':XGBClassifier(random_state=random_seed)}
# Scale features
mms =MinMaxScaler()
# Standardize the training, val & test feature data
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

# Creating Dictionary for Pipeline
from sklearn.pipeline import Pipeline
pipes={}
for acronym, model in models.items():
  pipes[acronym]= Pipeline([('model', model)])

# Getting the predefined split cross-validator
# Get the:
# feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit

x_train_val, y_train_val, ps = get_train_val_ps(x_train, y_train, x_val, y_val)

from scipy.stats import uniform

param_dists = {}

# FOR LOGISTIC REGRESSION
# The distribution for tol_grid: a uniform distribution over [loc, loc + scale]
tol_grid = uniform(loc=0.000001, scale=15)

# The distribution for C_grid & solver: a uniform distribution over [loc, loc + scale]
C_grid = uniform(loc=0.001, scale=10)
solver = ['lbfgs', 'sag']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]
# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__n_estimators':n_estimators,
     'model__eval_metric':eval_metric,
     'model__objective':objective}
]

# ============

from sklearn.model_selection import RandomizedSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by RandomizedSearchCV
best_score_params_estimator_rs = []

for acronym in pipes.keys():
    # RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipes[acronym],
                            param_distributions=param_dists[acronym],
                            n_iter=2,
                            scoring='f1_macro',
                            n_jobs=2,
                            cv=ps,
                            random_state=random_seed,
                            return_train_score=True)

    # Fit the pipeline
    rs = rs.fit(x_train_val, y_train_val)

    # Update best_score_param_estimators
    best_score_params_estimator_rs.append([rs.best_score_, rs.best_params_, rs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(rs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(acronym + '81.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)

cm81=confusion_matrix(test_c.labels.values, y_test_pred)
ax= plt.subplot()
sns.heatmap(cm81, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
plt.show()

print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))


#%%
# LOGISTIC REGRESSION & RANDOM FOREST (97 FEATURES (most important from PCA))
train_c= train.loc[:,train.columns.isin(np.append(important_97,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(important_97,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(important_97,'labels'))]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()

# Augmenting Minority Target Variabe
from imblearn.over_sampling import RandomOverSampler
# The RandomOverSampler
ros = RandomOverSampler(random_state=random_seed)
# Augment the training data
X_ros_train, y_ros_train = ros.fit_resample(x_train, y_train)

# Target Balance Re-Check
sns.barplot(x=pd.Series(y_ros_train).value_counts().index,y=pd.Series(y_ros_train).value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()


# models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          'xgbc':XGBClassifier(random_state=random_seed)}
# Scale features
mms =MinMaxScaler()
# Standardize the training, val & test feature data
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

# Creating Dictionary for Pipeline
from sklearn.pipeline import Pipeline
pipes={}
for acronym, model in models.items():
  pipes[acronym]= Pipeline([('model', model)])

# Getting the predefined split cross-validator
# Get the:
# feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit

x_train_val, y_train_val, ps = get_train_val_ps(x_train, y_train, x_val, y_val)

from scipy.stats import uniform

param_dists = {}

# FOR LOGISTIC REGRESSION
# The distribution for tol_grid: a uniform distribution over [loc, loc + scale]
tol_grid = uniform(loc=0.000001, scale=15)

# The distribution for C_grid & solver: a uniform distribution over [loc, loc + scale]
C_grid = uniform(loc=0.001, scale=10)
solver = ['lbfgs', 'sag']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]
# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__n_estimators':n_estimators,
     'model__eval_metric':eval_metric,
     'model__objective':objective}
]

# ============

from sklearn.model_selection import RandomizedSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by RandomizedSearchCV
best_score_params_estimator_rs = []

for acronym in pipes.keys():
    # RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipes[acronym],
                            param_distributions=param_dists[acronym],
                            n_iter=2,
                            scoring='f1_macro',
                            n_jobs=2,
                            cv=ps,
                            random_state=random_seed,
                            return_train_score=True)

    # Fit the pipeline
    rs = rs.fit(x_train_val, y_train_val)

    # Update best_score_param_estimators
    best_score_params_estimator_rs.append([rs.best_score_, rs.best_params_, rs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(rs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(acronym + '97.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)

cm97=confusion_matrix(test_c.labels.values, y_test_pred)
ax= plt.subplot()
sns.heatmap(cm97, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
plt.show()

print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))



# >>> from sklearn.metrics import hamming_loss
# hamming_loss(test_c.labels.values, y_test_pred)
