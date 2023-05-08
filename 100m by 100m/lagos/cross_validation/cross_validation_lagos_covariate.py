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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
random_seed=123
target_names = ['Not Deprived', 'Deprived']
THRESHOLD= 0.8
#%%
# Helper function
from toolbox import get_train_val_ps, important_features, cm, calculate_vif, covariate_features
covariate_features= covariate_features()
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
# Get the features into a numpy array
X = lagos.copy(deep=True).drop(['labels','type'],axis=1)

#%%
# PCA Analysis

#scale features
sc=StandardScaler()
# data_scaled= sc.fit_transform(X)
data_scaled= pd.DataFrame(sc.fit_transform(X),columns = X.columns)
pca=PCA(n_components='mle',svd_solver='full') #Initialize PCA
transformed= pca.fit_transform(data_scaled)
n_pcs= pca.components_.shape[0]
#
pcs=pd.DataFrame(np.abs(pca.components_[0]),columns=['PC1'])
pcs['features']= covariate_features
pcs.sort_values(by='PC1',axis=0,ascending=False,inplace=True)
#
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names= X.columns.to_list()
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
important= pd.DataFrame(dic.items(),columns=['PCs','features'])
most_important_features= important.features.unique() #37
important_37= important.features.unique()


# Plot explained variance
plt.figure(figsize=(25,10))
x=np.arange(1,len(pca.explained_variance_ratio_)+1)
plt.xticks(x, fontsize=6,rotation=90)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_),c='red',marker='*')
plt.axvline(x = 27, color = 'b', label = 'axvline - full height')
plt.axvline(x = 37, color = 'b', label = 'axvline - full height')
plt.axhline(y = 1, color = 'g', linestyle = '-')
plt.grid()
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.savefig('pca_lagos.png')
plt.show()

pca_df= pd.DataFrame(np.array([np.array(x,dtype=int),np.cumsum(pca.explained_variance_ratio_)]),
                     index=['n_features','variance']).T
print(f'With half the feature size, you get {round(pca_df.head(27).iloc[-1,-1],4)*100}% variance explained')
print(f'With 37 feature size, you get {round(pca_df.head(37).iloc[-1,-1],4)*100}% variance explained')

#%%
plt.figure(figsize=(25,10))
sns.barplot(data=pcs,x=list(range(1,54)),y='PC1')
plt.xticks(rotation=90)
plt.title('Feature Importance (PC1)')
plt.savefig('pc1.png')
plt.show()

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (37 FEATURES)
train_c= train.loc[:,train.columns.isin(np.append(important_37,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(important_37,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(important_37,'labels'))]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
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
penalty=['l1','l2']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
n_estimators= [10,30,100,200,500]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       #'model__n_estimators':n_estimators,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]
# FOR XGBOOST
n_estimators= [10,30,100,200,500]
max_leaves= [10,15,20,25,30]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__max_leaves':max_leaves,
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
    cv_results.to_csv(acronym + '96.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)
# y_test_pred= pd.Series(best_estimator_gs.predict_proba(x_test)[:,1]).apply(lambda x:0 if x>THRESHOLD else 1)

cm(test_c.labels.values, y_test_pred) #Confusion matrix
print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (53 FEATURES)
# Split into features and target
x_train, y_train= train.iloc[:,:-1], train.iloc[:,-1].values
x_val, y_val= validation.iloc[:,:-1], validation.iloc[:,-1].values
x_test, y_test= test.iloc[:,:-1], test.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

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
# y_test_pred= pd.Series(best_estimator_gs.predict_proba(x_test)[:,1]).apply(lambda x:0 if x>THRESHOLD else 1)
cm(test.labels.values, y_test_pred) #Confusion matrix
print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))

#%%