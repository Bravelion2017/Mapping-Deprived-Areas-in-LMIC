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
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from toolbox import get_train_val_ps, important_features, calculate_vif, cm
random_seed=123

#Helper Function
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
def cm(ytrue,ypred):
    from sklearn.metrics import confusion_matrix
    cm2 = confusion_matrix(ytrue, ypred)
    ax = plt.subplot()
    sns.heatmap(cm2, annot=True, fmt='g', ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Not Deprived', 'Deprived'])
    ax.yaxis.set_ticklabels(['Not Deprived', 'Deprived'])
    plt.show()
target_names = ['Not Deprived', 'Deprived']
#%%
# Load Data into Pandas Dataframe
lagos= pd.read_parquet('lagos_rgbn_df.parquet.gzip')
print(lagos.describe())

#%%
# Check for Nulls
print(lagos.isna().sum().sort_values(ascending=False))

#%%
# Split Train, Validation & Test
train= lagos[lagos.type=='train'].copy(deep=True).drop('type',axis=1)
validation= lagos[lagos.type=='validation'].copy(deep=True).drop('type',axis=1)
test= lagos[lagos.type=='test'].copy(deep=True).drop('type',axis=1)
print(f'Null values: {lagos.isna().sum().any()}')

#%%
# SVD

# Get the features into a numpy array
X = lagos.copy(deep=True).drop(['labels','type'],axis=1)
# computing singular values using numpy
H= X.values.T @ X.values
_,d,_= np.linalg.svd(H)
res=pd.DataFrame(d,index=X.columns, columns=['Singular Values'])
# print(tabulate(res,headers='keys',tablefmt="fancy_grid"))
print(tabulate(res,headers='keys',tablefmt="fancy_grid"))
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
most_important_features= important.features.unique() #3
# pd.DataFrame(pca.components_,columns=data_scaled.columns)


# Plot explained variance
plt.figure()
x=np.arange(1,len(pca.explained_variance_ratio_)+1)
plt.xticks(x, fontsize=6,rotation=90)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_),c='red',marker='*')
plt.axvline(x = 2, color = 'b', label = 'axvline - full height')
plt.axvline(x = 3, color = 'b', label = 'axvline - full height')
plt.axhline(y = 1, color = 'g', linestyle = '-')
plt.grid()
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
# plt.savefig('pca_lagos_covariate.png')
plt.show()

pca_df= pd.DataFrame(np.array([np.array(x,dtype=int),np.cumsum(pca.explained_variance_ratio_)]),
                     index=['n_features','variance']).T
print(f'With 3  feature size, you get {round(pca_df.head(3).iloc[-1,-1],4)*100}% variance explained')

#%%
# Trying SVD again with most important 38 features
X1= X.loc[:,X.columns.isin(most_important_features)]
# computing singular values using numpy
H1= X1.values.T @ X1.values
_,d1,_= np.linalg.svd(H1)
res1=pd.DataFrame(d1,index=X1.columns, columns=['Singular Values'])
print(tabulate(res1.tail(),headers='keys',tablefmt="fancy_grid"))
condition1=np.linalg.cond(X1)
condition_df1=pd.DataFrame(data=[condition1],columns=['Condition Number'])
print(tabulate(condition_df1,headers='keys',tablefmt="fancy_grid"))

#%%
# LOGISTIC REGRESSION & RANDOM FOREST & XGBOOST (3 FEATURES)
train_c= train.loc[:,train.columns.isin(np.append(most_important_features,'labels'))]
val_c= validation.loc[:,validation.columns.isin(np.append(most_important_features,'labels'))]
test_c= test.loc[:,test.columns.isin(np.append(most_important_features,'labels'))]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
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
n_estimators= [10,30,100,200,500]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]
# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
max_leaves= [10,15,20,25,30]
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
    cv_results.to_csv(acronym + '38.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)

cm(test_c.labels.values, y_test_pred) #Confusion matrix
print(classification_report(test_c.labels.values, y_test_pred, target_names=target_names))


#%%
# LOGISTIC REGRESSION & RANDOM FOREST & XGBOOST (4 FEATURES)
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
n_estimators= [10,30,100,200,500]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]

# FOR XGBOOST
n_estimators= [10,30,100]
eval_metric= ['logloss','error']
max_leaves= [10,15,20,25,30]
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

cm(test.labels.values, y_test_pred) #Confusion matrix
print(classification_report(test.labels.values, y_test_pred, target_names=target_names))

#%%