import pandas as pd
import numpy as np


def common_var_checker(df_train, df_val, df_test, target):
    """
    The common variables checker

    Parameters
    ----------
    df_train : the dataframe of training data
    df_val : the dataframe of validation data
    df_test : the dataframe of test data
    target : the name of the target

    Returns
    ----------
    The dataframe of common variables between the training, validation and test data
    """

    # Get the dataframe of common variables between the training, validation and test data
    df_common_var = pd.DataFrame(
        np.intersect1d(np.intersect1d(df_train.columns, df_val.columns), np.union1d(df_test.columns, [target])),
        columns=['common var'])

    return df_common_var


def id_checker(df, dtype='float'):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe
    dtype : the data type identifiers cannot have, 'float' by default
            i.e., if a feature has this data type, it cannot be an identifier

    Returns
    ----------
    The dataframe of identifiers
    """

    # Get the dataframe of identifiers
    df_id = df[[var for var in df.columns
                # If the data type is not dtype
                if (df[var].dtype != dtype
                    # If the value is unique for each sample
                    and df[var].nunique(dropna=True) == df[var].notnull().sum())]]

    return df_id


def datetime_transformer(df, datetime_vars):
    """
    The datetime transformer

    Parameters
    ----------
    df : the dataframe
    datetime_vars : the datetime variables

    Returns
    ----------
    The dataframe where datetime_vars are transformed into the following 6 datetime types:
    year, month, day, hour, minute and second
    """

    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year': lambda x: x.dt.year,
             'month': lambda x: x.dt.month,
             'day': lambda x: x.dt.day,
             'hour': lambda x: x.dt.hour,
             'minute': lambda x: x.dt.minute,
             'second': lambda x: x.dt.second}

    # Make a copy of df
    df_datetime = df.copy(deep=True)

    # For each variable in datetime_vars
    for var in datetime_vars:
        # Cast the variable to datetime
        df_datetime[var] = pd.to_datetime(df_datetime[var])

        # For each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # Add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' + datetime_type] = datetime_type_operator(df_datetime[var])

    # Remove datetime_vars from df_datetime
    df_datetime = df_datetime.drop(columns=datetime_vars)

    return df_datetime


def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : the dataframe

    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and data type
    """

    # Get the dataframe of variables with NaN, their proportion of NaN and data type
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan


def cat_var_checker(df, dtype='object'):
    """
    The categorical variable checker

    Parameters
    ----------
    df : the dataframe
    dtype : the data type categorical variables should have, 'object' by default
            i.e., if a variable has this data type, it should be a categorical variable

    Returns
    ----------
    The dataframe of categorical variables and their number of unique value
    """

    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           # If the data type is dtype
                           for var in df.columns if df[var].dtype == dtype],
                          columns=['var', 'nunique'])

    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    return df_cat


def separate_duplicate_original(X_aug_train, y_aug_train, minor_class):
    """
    Separate the duplicated class from the original class

    Parameters
    ----------
    X_aug_train : The augmented feature matrix
    y_aug_train : The augmented target vector
    minor_class : The minority class

    Returns
    ----------
    The separated duplicated class and original class
    """

    # Make a copy of y_aug_train
    y_aug_dup_ori_train = np.array(y_aug_train)

    # For each sample in the augmented data
    for i in range(X_aug_train.shape[0]):
        # If the sample has the minor class
        if y_aug_dup_ori_train[i] == minor_class:
            # Flag variable, indicating whether a sample in the augmented data is the same as a sample in the original data
            same = False

            # For each sample in the original data
            for j in range(X_aug_train.shape[0]):
                if j == i:
                    continue

                # If the sample has the minor class
                if y_aug_dup_ori_train[j] == minor_class:
                    if len(np.setdiff1d(X_aug_train[i, :], X_aug_train[j, :])) == 0:
                        # The two samples are the same
                        same = True
                        break

            # If the two samples are different
            if same is False:
                y_aug_dup_ori_train[i] = 2

    return y_aug_dup_ori_train


def separate_generate_original(X_aug_train, y_aug_train, X_train, y_train, minor_class):
    """
    Separate the generated class from the original class

    Parameters
    ----------
    X_aug_train : The augmented feature matrix
    y_aug_train : The augmented target vector
    X_train : The original feature matrix
    y_train : The original target vector
    minor_class : The minority class

    Returns
    ----------
    The separated generated class and original class
    """

    # Make a copy of y_aug_train
    y_aug_gen_ori_train = np.array(y_aug_train)

    # For each sample in the augmented data
    for i in range(X_aug_train.shape[0]):
        # If the sample has the minor class
        if y_aug_gen_ori_train[i] == minor_class:
            # Flag variable, indicating whether a sample in the augmented data is the same as a sample in the original data
            same = False

            # For each sample in the original data
            for j in range(X_train.shape[0]):
                # If the sample has the minor class
                if y_train[j] == minor_class:
                    if len(np.setdiff1d(X_aug_train[i, :], X_train[j, :])) == 0:
                        # The two samples are the same
                        same = True
                        break

            # If the two samples are different
            if same is False:
                y_aug_gen_ori_train[i] = 2

    return y_aug_gen_ori_train


from sklearn.model_selection import PredefinedSplit


def get_train_val_ps(X_train, y_train, X_val, y_val):
    """
    Get the:
    feature matrix and target velctor in the combined training and validation data
    target vector in the combined training and validation data
    PredefinedSplit

    Parameters
    ----------
    X_train : the feature matrix in the training data
    y_train : the target vector in the training data
    X_val : the feature matrix in the validation data
    y_val : the target vector in the validation data

    Return
    ----------
    The feature matrix in the combined training and validation data
    The target vector in the combined training and validation data
    PredefinedSplit
    """

    # Combine the feature matrix in the training and validation data
    X_train_val = np.vstack((X_train, X_val))

    # Combine the target vector in the training and validation data
    y_train_val = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).reshape(-1)

    # Get the indices of training and validation data
    train_val_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_val.shape[0], 0))

    # The PredefinedSplit
    ps = PredefinedSplit(train_val_idxs)

    return X_train_val, y_train_val, ps


import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


def training_valation_test(X_train, y_train, X_test, y_test, ps, abspath_curr, name):
    """
    Training, validation and test

    Parameters
    ----------
    X_train : the feature matrix in the training data
    y_train : the target vector in the training data
    X_test : the feature matrix in the test data
    y_test : the target vector in the test data
    ps : the PredefinedSplit
    abspath_curr : the absolute path of the current folder
    name : the name of the cv_results folder

    Return
    ----------
    The dataframe of [precision, recall, best_estimator]
    """

    # ************************************************************************************************
    # Creating the directory for the cv results
    directory = os.path.dirname(abspath_curr + name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ************************************************************************************************
    # Training and validation

    # The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
    best_score_param_estimator_gs = []

    for acronym in pipes.keys():
        # GridSearchCV
        gs = GridSearchCV(estimator=pipes[acronym],
                          param_grid=param_grids[acronym],
                          scoring='f1',
                          n_jobs=2,
                          cv=ps,
                          return_train_score=True)

        # Fit the pipeline
        gs = gs.fit(X_train, y_train)

        # Update best_score_param_estimator_gs
        best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

        # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
        cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

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
        cv_results.to_csv(path_or_buf=abspath_curr + name + '/' + acronym + '.csv', index=False)

    # ************************************************************************************************
    # Test

    # The list of [precision, recall, fscore, auc, best_estimator]
    precision_recall_fscore_auc_best_estimator = []

    for best_score, best_param, best_estimator in best_score_param_estimator_gs:
        # Get the prediction
        y_pred = best_estimator.predict(X_test)

        # Get the precision, recall, fscore, support
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

        # Get the auc
        auc = roc_auc_score(y_test, y_pred)

        # Update precision_recall_fscore_auc_best_estimator
        precision_recall_fscore_auc_best_estimator.append([precision, recall, fscore, auc, best_estimator])

    # Return precision_recall_fscore_best_estimator
    return pd.DataFrame(precision_recall_fscore_auc_best_estimator,
                        columns=['Precision', 'Recall', 'F1-score', 'AUC', 'Model'])
def important_features():
    res = ['gabor_sc7_filter_13',
           'gabor_sc5_filter_14',
           'orb_sc71_variance',
           'gabor_sc7_filter_10',
           'gabor_sc5_filter_5',
           'gabor_sc5_filter_1',
           'gabor_sc5_filter_10',
           'gabor_sc3_filter_13',
           'orb_sc51_variance',
           'fourier_sc71_mean',
           'lsr_sc71_line_mean',
           'gabor_sc3_filter_1',
           'lsr_sc71_line_contrast',
           'gabor_sc3_filter_5',
           'gabor_sc5_filter_8',
           'gabor_sc3_filter_3',
           'ndvi_sc5_mean',
           'gabor_sc7_filter_11',
           'gabor_sc7_filter_12',
           'gabor_sc5_filter_13',
           'gabor_sc3_filter_7',
           'orb_sc31_variance',
           'ndvi_sc7_mean',
           'gabor_sc5_filter_6',
           'gabor_sc5_filter_12',
           'gabor_sc7_filter_4',
           'mean_sc5_mean',
           'lsr_sc71_line_length',
           'gabor_sc5_filter_9',
           'gabor_sc3_filter_11',
           'gabor_sc5_filter_11',
           'ndvi_sc3_mean',
           'orb_sc71_max',
           'gabor_sc5_filter_2',
           'gabor_sc5_mean',
           'gabor_sc7_filter_6',
           'orb_sc71_kurtosis',
           'gabor_sc7_mean',
           'gabor_sc7_filter_2',
           'orb_sc31_kurtosis',
           'mean_sc7_mean',
           'ndvi_sc5_variance',
           'orb_sc71_mean',
           'lbpm_sc5_variance',
           'sfs_sc51_mean',
           'hog_sc3_variance',
           'lsr_sc51_line_contrast',
           'orb_sc71_skew',
           'sfs_sc31_mean',
           'sfs_sc71_max_line_length',
           'orb_sc31_skew',
           'sfs_sc51_std',
           'hog_sc5_max',
           'lsr_sc31_line_mean',
           'hog_sc5_mean',
           'orb_sc51_kurtosis',
           'sfs_sc71_max_ratio_of_orthogonal_angles',
           'lac_sc5_lac',
           'lbpm_sc5_mean',
           'fourier_sc31_variance',
           'hog_sc7_skew',
           'lsr_sc31_line_contrast',
           'hog_sc5_skew',
           'sfs_sc31_min_line_length',
           'sfs_sc51_min_line_length',
           'lac_sc3_lac',
           'hog_sc3_mean',
           'lbpm_sc3_mean',
           'lbpm_sc7_mean',
           'hog_sc7_kurtosis',
           'ndvi_sc3_variance',
           'sfs_sc51_max_ratio_of_orthogonal_angles',
           'gabor_sc3_variance',
           'sfs_sc31_w_mean',
           'sfs_sc51_w_mean',
           'gabor_sc7_variance',
           'lbpm_sc7_kurtosis',
           'fourier_sc71_variance',
           'gabor_sc7_filter_14',
           'mean_sc3_variance',
           'fourier_sc51_variance',
           'orb_sc51_mean',
           'sfs_sc31_max_line_length',
           'fourier_sc31_mean',
           'lbpm_sc5_max',
           'pantex_sc5_min',
           'hog_sc3_kurtosis',
           'lsr_sc51_line_mean',
           'ndvi_sc7_variance',
           'sfs_sc51_max_line_length',
           'hog_sc7_variance',
           'fourier_sc51_mean',
           'gabor_sc5_filter_4',
           'orb_sc31_max',
           'gabor_sc5_variance',
           'lsr_sc51_line_length',
           'lbpm_sc3_variance',
           'hog_sc5_variance',
           'mean_sc5_variance',
           'gabor_sc3_mean',
           'lsr_sc31_line_length',
           'orb_sc31_mean',
           'orb_sc51_skew',
           'orb_sc51_max',
           'mean_sc7_variance',
           'gabor_sc7_filter_3',
           'lbpm_sc3_skew',
           'gabor_sc3_filter_14']
    return res
def important_97():
    res= ['mean_sc7_mean', 'ndvi_sc5_variance', 'fourier_sc51_mean',
       'lbpm_sc5_variance', 'sfs_sc51_mean', 'sfs_sc51_max_line_length',
       'lsr_sc51_line_contrast', 'hog_sc3_variance', 'hog_sc7_variance',
       'sfs_sc31_mean', 'sfs_sc71_max_line_length', 'orb_sc31_skew',
       'lsr_sc31_line_mean', 'gabor_sc7_variance', 'orb_sc51_kurtosis',
       'orb_sc71_kurtosis', 'lac_sc5_lac', 'lbpm_sc7_mean',
       'lsr_sc31_line_contrast', 'lbpm_sc5_mean', 'fourier_sc31_variance',
       'hog_sc7_skew', 'hog_sc5_skew', 'ndvi_sc3_variance',
       'sfs_sc31_min_line_length', 'sfs_sc51_min_line_length',
       'lac_sc3_lac', 'lbpm_sc3_mean', 'hog_sc7_mean', 'lac_sc7_lac',
       'hog_sc7_kurtosis', 'sfs_sc71_min_line_length', 'pantex_sc7_min',
       'sfs_sc71_max_ratio_of_orthogonal_angles', 'hog_sc3_mean',
       'sfs_sc51_max_ratio_of_orthogonal_angles', 'hog_sc5_mean',
       'sfs_sc31_w_mean', 'hog_sc3_max', 'sfs_sc51_w_mean',
       'lbpm_sc7_kurtosis', 'fourier_sc51_variance', 'mean_sc3_variance',
       'orb_sc51_mean', 'sfs_sc31_max_line_length', 'fourier_sc31_mean',
       'lbpm_sc5_max', 'pantex_sc5_min', 'orb_sc51_max',
       'hog_sc3_kurtosis', 'lsr_sc51_line_mean', 'ndvi_sc7_variance',
       'gabor_sc5_filter_4', 'orb_sc31_max', 'gabor_sc5_filter_14',
       'orb_sc31_mean', 'lbpm_sc3_variance', 'hog_sc5_variance',
       'orb_sc31_variance', 'lsr_sc31_line_length', 'gabor_sc3_mean',
       'orb_sc71_skew', 'mean_sc5_variance', 'orb_sc51_skew',
       'lsr_sc51_line_length', 'mean_sc7_variance', 'orb_sc51_variance',
       'gabor_sc7_filter_3', 'lbpm_sc3_skew', 'gabor_sc3_filter_14',
       'gabor_sc7_filter_12', 'sfs_sc51_std', 'ndvi_sc3_mean',
       'sfs_sc71_std', 'lbpm_sc7_skew', 'gabor_sc5_variance',
       'mean_sc3_mean', 'gabor_sc7_filter_2', 'gabor_sc3_filter_4',
       'gabor_sc3_filter_6', 'gabor_sc3_filter_8', 'gabor_sc3_filter_13',
       'gabor_sc7_mean', 'gabor_sc5_filter_1', 'gabor_sc3_filter_3',
       'gabor_sc3_filter_11', 'gabor_sc3_filter_1', 'gabor_sc3_filter_7',
       'gabor_sc5_filter_2', 'ndvi_sc5_mean', 'gabor_sc7_filter_10',
       'gabor_sc5_filter_9', 'gabor_sc5_filter_7', 'mean_sc5_mean',
       'gabor_sc5_filter_3', 'gabor_sc5_filter_8', 'gabor_sc7_filter_9']
    return res
def important_38():
    res= ['ses_measles_2014', 'ph_max_tem_2019', 'ph_dist_open_coast_2020',
       'ho_impr_housing_2015', 'ph_base_water_2010', 'uu_bld_count_2020',
       'ph_pm25_2016', 'ph_climate_risk_2020', 'ph_land_c2_2020',
       'fs_dist_hf_2019', 'ph_slope_2000', 'ph_land_c1_2019',
       'ph_bio_dvst_2015', 'ph_ndvi_2019', 'ph_grd_water_2000',
       'in_dist_waterway_2016', 'ses_preg_2017', 'po_hrsl_2018',
       'po_wp_2020', 'ph_dist_herb_2015', 'in_night_light_2016',
       'ph_dist_shrub_2015', 'ses_an_visits_2014',
       'ph_dist_woody_tree_2015', 'fs_dist_school_2020',
       'ses_impr_water_src_2014', 'ses_dist_gov_office_2022',
       'ph_dist_inland_water_2018', 'ph_dist_aq_veg_2015',
       'in_dist_rd_2016', 'ses_pfpr_2017', 'ses_w_lit_2016',
       'ph_dist_cultivated_2015', 'ses_odef_2014', 'ses_dtp3_2014',
       'sh_dist_mnr_pofw_2019', 'ph_dist_sparse_veg_2015',
       'uu_bld_den_2020']
    return res
def important_6():
    res= ['ph_bio_dvst_2015', 'ph_land_c1_2019', 'ph_land_c2_2020',
       'ph_ndvi_2019', 'ph_slope_2000', 'uu_urb_bldg_2018']
    return res

# function to load raster data
def read_image(dir):
    """This function read raster image, convert to array and get the geoprojection"""
    img = gdal.Open(os.path.join(dir))
    img_arr = img.ReadAsArray()
    img_gt = img.GetGeoTransform()
    img_georef = img.GetProjectionRef()
    return [img_arr, img_gt, img_georef]

def contextual_features():
    con = ['fourier_sc31_mean',
                           'fourier_sc31_variance',
                           'fourier_sc51_mean',
                           'fourier_sc51_variance',
                           'fourier_sc71_mean',
                           'fourier_sc71_variance',
                           'gabor_sc3_filter_1',
                           'gabor_sc3_filter_10',
                           'gabor_sc3_filter_11',
                           'gabor_sc3_filter_12',
                           'gabor_sc3_filter_13',
                           'gabor_sc3_filter_14',
                           'gabor_sc3_filter_2',
                           'gabor_sc3_filter_3',
                           'gabor_sc3_filter_4',
                           'gabor_sc3_filter_5',
                           'gabor_sc3_filter_6',
                           'gabor_sc3_filter_7',
                           'gabor_sc3_filter_8',
                           'gabor_sc3_filter_9',
                           'gabor_sc3_mean',
                           'gabor_sc3_variance',
                           'gabor_sc5_filter_1',
                           'gabor_sc5_filter_10',
                           'gabor_sc5_filter_11',
                           'gabor_sc5_filter_12',
                           'gabor_sc5_filter_13',
                           'gabor_sc5_filter_14',
                           'gabor_sc5_filter_2',
                           'gabor_sc5_filter_3',
                           'gabor_sc5_filter_4',
                           'gabor_sc5_filter_5',
                           'gabor_sc5_filter_6',
                           'gabor_sc5_filter_7',
                           'gabor_sc5_filter_8',
                           'gabor_sc5_filter_9',
                           'gabor_sc5_mean',
                           'gabor_sc5_variance',
                           'gabor_sc7_filter_1',
                           'gabor_sc7_filter_10',
                           'gabor_sc7_filter_11',
                           'gabor_sc7_filter_12',
                           'gabor_sc7_filter_13',
                           'gabor_sc7_filter_14',
                           'gabor_sc7_filter_2',
                           'gabor_sc7_filter_3',
                           'gabor_sc7_filter_4',
                           'gabor_sc7_filter_5',
                           'gabor_sc7_filter_6',
                           'gabor_sc7_filter_7',
                           'gabor_sc7_filter_8',
                           'gabor_sc7_filter_9',
                           'gabor_sc7_mean',
                           'gabor_sc7_variance',
                           'hog_sc3_kurtosis',
                           'hog_sc3_max',
                           'hog_sc3_mean',
                           'hog_sc3_skew',
                           'hog_sc3_variance',
                           'hog_sc5_kurtosis',
                           'hog_sc5_max',
                           'hog_sc5_mean',
                           'hog_sc5_skew',
                           'hog_sc5_variance',
                           'hog_sc7_kurtosis',
                           'hog_sc7_max',
                           'hog_sc7_mean',
                           'hog_sc7_skew',
                           'hog_sc7_variance',
                           'lac_sc3_lac',
                           'lac_sc5_lac',
                           'lac_sc7_lac',
                           'lbpm_sc3_kurtosis',
                           'lbpm_sc3_max',
                           'lbpm_sc3_mean',
                           'lbpm_sc3_skew',
                           'lbpm_sc3_variance',
                           'lbpm_sc5_kurtosis',
                           'lbpm_sc5_max',
                           'lbpm_sc5_mean',
                           'lbpm_sc5_skew',
                           'lbpm_sc5_variance',
                           'lbpm_sc7_kurtosis',
                           'lbpm_sc7_max',
                           'lbpm_sc7_mean',
                           'lbpm_sc7_skew',
                           'lbpm_sc7_variance',
                           'lsr_sc31_line_contrast',
                           'lsr_sc31_line_length',
                           'lsr_sc31_line_mean',
                           'lsr_sc51_line_contrast',
                           'lsr_sc51_line_length',
                           'lsr_sc51_line_mean',
                           'lsr_sc71_line_contrast',
                           'lsr_sc71_line_length',
                           'lsr_sc71_line_mean',
                           'mean_sc3_mean',
                           'mean_sc3_variance',
                           'mean_sc5_mean',
                           'mean_sc5_variance',
                           'mean_sc7_mean',
                           'mean_sc7_variance',
                           'ndvi_sc3_mean',
                           'ndvi_sc3_variance',
                           'ndvi_sc5_mean',
                           'ndvi_sc5_variance',
                           'ndvi_sc7_mean',
                           'ndvi_sc7_variance',
                           'orb_sc31_kurtosis',
                           'orb_sc31_max',
                           'orb_sc31_mean',
                           'orb_sc31_skew',
                           'orb_sc31_variance',
                           'orb_sc51_kurtosis',
                           'orb_sc51_max',
                           'orb_sc51_mean',
                           'orb_sc51_skew',
                           'orb_sc51_variance',
                           'orb_sc71_kurtosis',
                           'orb_sc71_max',
                           'orb_sc71_mean',
                           'orb_sc71_skew',
                           'orb_sc71_variance',
                           'pantex_sc3_min',
                           'pantex_sc5_min',
                           'pantex_sc7_min',
                           'sfs_sc31_max_line_length',
                           'sfs_sc31_max_ratio_of_orthogonal_angles',
                           'sfs_sc31_mean',
                           'sfs_sc31_min_line_length',
                           'sfs_sc31_std',
                           'sfs_sc31_w_mean',
                           'sfs_sc51_max_line_length',
                           'sfs_sc51_max_ratio_of_orthogonal_angles',
                           'sfs_sc51_mean',
                           'sfs_sc51_min_line_length',
                           'sfs_sc51_std',
                           'sfs_sc51_w_mean',
                           'sfs_sc71_max_line_length',
                           'sfs_sc71_max_ratio_of_orthogonal_angles',
                           'sfs_sc71_mean',
                           'sfs_sc71_min_line_length',
                           'sfs_sc71_std',
                           'sfs_sc71_w_mean']
    return con

def tif_to_df(feature_path,label_path,feature_names):
 """
 Get tif files then return dataframe
 feature_path: sorted paths of tif files (features)
 label_path: sorted paths of tif files (label)
 """
 import numpy as np
 import pandas as pd
 import rasterio
 from rasterio.merge import merge

 RasterTiles = sorted(feature_path) #features
 MaskTiles = sorted(label_path) #labels

 # Preparing training
 raster_to_mosiac = []
 # Read the tif files with rasterio then parse into list
 for p in RasterTiles:
     raster = rasterio.open(p)
     raster_to_mosiac.append(raster)

 # Get combined numpy array from the populated list
 X, out_transform = merge(raster_to_mosiac)

 # Replication for train, validation, test files on features and labels
 raster_to_mosiac = []

 for p in MaskTiles:
     raster = rasterio.open(p)
     raster_to_mosiac.append(raster)

 Y, out_transform = merge(raster_to_mosiac)

 # Dataframe Setup
 #Training Data
 X_train = X[:, X[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
 y_train = Y[:, Y[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
 y_train =y_train.astype(int) #change the label to integer
 X_train = np.transpose(X_train) #flip the array
 y_train = np.transpose(y_train) #flip the array
 # To DataFrames- Train, Validation & Test
 df_train = pd.DataFrame(X_train,columns=feature_names)  # contextual features
 df_train['labels'] = y_train

 return df_train

def tif_to_df2(feature_path,feature_names):
 """
 Get tif files then return dataframe
 feature_path: sorted paths of tif files (features)
 label_path: sorted paths of tif files (label)
 """
 import numpy as np
 import pandas as pd
 import rasterio
 from rasterio.merge import merge

 RasterTiles = sorted(feature_path) #features

 # Preparing training
 raster_to_mosiac = []
 # Read the tif files with rasterio then parse into list
 for p in RasterTiles:
     raster = rasterio.open(p)
     raster_to_mosiac.append(raster)

 # Get combined numpy array from the populated list
 X, out_transform = merge(raster_to_mosiac)


 # Dataframe Setup
 #Training Data
 X_train = X[:, X[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
 X_train = np.transpose(X_train) #flip the array
 # To DataFrames- Train, Validation & Test
 df_train = pd.DataFrame(X_train,columns=feature_names)

 return df_train


def plot_pred_area(nrow,ncol,pred):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    pred_map = pred.reshape(nrow, ncol)
    values = np.unique(pred_map.ravel())
    im = plt.imshow(pred_map, interpolation='none')
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Class {l}".format(l=values[i])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def cm(ytrue,ypred):
    import matplotlib.pyplot as plt
    import seaborn as sns
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
def important_62():
    res= ['fourier_sc51_variance', 'fourier_sc71_mean', 'gabor_sc3_filter_12',
       'gabor_sc3_filter_14', 'gabor_sc3_filter_2', 'gabor_sc3_filter_4',
       'gabor_sc3_filter_8', 'gabor_sc3_filter_9', 'gabor_sc3_variance',
       'gabor_sc5_filter_7', 'hog_sc3_max', 'hog_sc3_mean', 'hog_sc3_variance',
       'hog_sc5_kurtosis', 'hog_sc5_max', 'hog_sc5_mean', 'hog_sc5_skew',
       'hog_sc7_kurtosis', 'hog_sc7_max', 'hog_sc7_mean', 'hog_sc7_skew',
       'hog_sc7_variance', 'lac_sc3_lac', 'lac_sc5_lac', 'lac_sc7_lac',
       'lbpm_sc3_max', 'lbpm_sc3_mean', 'lbpm_sc5_kurtosis', 'lbpm_sc5_mean',
       'lbpm_sc5_variance', 'lbpm_sc7_kurtosis', 'lbpm_sc7_mean',
       'lbpm_sc7_skew', 'lbpm_sc7_variance', 'lsr_sc31_line_length',
       'lsr_sc31_line_mean', 'lsr_sc51_line_length', 'lsr_sc71_line_mean',
       'mean_sc3_variance', 'ndvi_sc3_mean', 'ndvi_sc3_variance',
       'ndvi_sc5_mean', 'ndvi_sc5_variance', 'ndvi_sc7_mean',
       'ndvi_sc7_variance', 'orb_sc51_variance', 'orb_sc71_max',
       'orb_sc71_mean', 'orb_sc71_variance', 'pantex_sc3_min',
       'pantex_sc5_min', 'sfs_sc31_max_line_length',
       'sfs_sc31_max_ratio_of_orthogonal_angles', 'sfs_sc31_mean',
       'sfs_sc31_min_line_length', 'sfs_sc31_std', 'sfs_sc51_max_line_length',
       'sfs_sc51_mean', 'sfs_sc51_min_line_length', 'sfs_sc71_max_line_length',
       'sfs_sc71_max_ratio_of_orthogonal_angles', 'sfs_sc71_min_line_length']
    return res
def calculate_vif(df, features):
    from sklearn.linear_model import LinearRegression
    import pandas as pd
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
def important_104():
    res= ['mean_sc7_mean', 'lbpm_sc7_skew', 'sfs_sc71_mean', 'orb_sc51_mean',
       'lsr_sc71_line_mean', 'sfs_sc71_std', 'hog_sc3_variance',
       'orb_sc71_mean', 'lsr_sc31_line_length', 'orb_sc31_skew',
       'sfs_sc31_std', 'sfs_sc51_mean', 'sfs_sc31_w_mean',
       'lbpm_sc3_mean', 'lsr_sc31_line_mean', 'hog_sc5_kurtosis',
       'pantex_sc7_min', 'lbpm_sc7_mean', 'pantex_sc5_min',
       'lbpm_sc3_max', 'lac_sc3_lac',
       'sfs_sc51_max_ratio_of_orthogonal_angles', 'hog_sc7_mean',
       'lsr_sc31_line_contrast',
       'sfs_sc71_max_ratio_of_orthogonal_angles', 'pantex_sc3_min',
       'hog_sc7_skew', 'sfs_sc31_min_line_length',
       'sfs_sc51_min_line_length', 'orb_sc71_kurtosis',
       'hog_sc7_kurtosis', 'ndvi_sc3_variance',
       'sfs_sc31_max_line_length', 'hog_sc3_mean', 'hog_sc5_mean',
       'lac_sc5_lac', 'fourier_sc31_variance', 'gabor_sc7_filter_7',
       'orb_sc51_kurtosis', 'sfs_sc71_w_mean', 'hog_sc3_max',
       'lsr_sc51_line_contrast', 'lbpm_sc7_kurtosis', 'mean_sc3_variance',
       'lbpm_sc7_variance', 'sfs_sc51_max_line_length', 'sfs_sc31_mean',
       'gabor_sc7_filter_14', 'lbpm_sc5_max', 'lbpm_sc7_max',
       'fourier_sc71_variance', 'sfs_sc71_max_line_length',
       'hog_sc3_skew', 'lsr_sc51_line_mean', 'ndvi_sc7_variance',
       'orb_sc31_variance', 'fourier_sc51_mean', 'hog_sc7_variance',
       'lbpm_sc3_variance', 'orb_sc51_variance', 'gabor_sc3_filter_4',
       'gabor_sc5_filter_8', 'hog_sc5_variance', 'orb_sc31_mean',
       'orb_sc71_variance', 'mean_sc5_variance', 'gabor_sc3_filter_12',
       'gabor_sc5_variance', 'gabor_sc3_filter_6', 'sfs_sc51_std',
       'gabor_sc3_filter_2', 'gabor_sc3_filter_8', 'gabor_sc7_filter_4',
       'gabor_sc7_filter_13', 'lbpm_sc3_skew', 'lbpm_sc5_variance',
       'ndvi_sc7_mean', 'orb_sc71_skew', 'gabor_sc5_filter_14',
       'lsr_sc51_line_length', 'orb_sc51_skew', 'mean_sc7_variance',
       'lbpm_sc5_skew', 'gabor_sc3_filter_5', 'gabor_sc7_filter_12',
       'lsr_sc71_line_length', 'gabor_sc3_filter_3', 'gabor_sc3_mean',
       'gabor_sc5_filter_12', 'gabor_sc5_filter_6', 'gabor_sc5_filter_2',
       'gabor_sc3_filter_13', 'gabor_sc3_filter_11', 'gabor_sc7_filter_2',
       'gabor_sc5_filter_3', 'gabor_sc5_filter_1', 'gabor_sc3_filter_1',
       'gabor_sc3_filter_7', 'gabor_sc7_filter_8', 'gabor_sc5_filter_11',
       'gabor_sc5_filter_7', 'gabor_sc7_filter_1', 'gabor_sc7_filter_3',
       'ndvi_sc5_mean']
    return res
def covariate_features():
    covariate_features = ['fs_dist_fs_2020',
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
    return covariate_features

def important_37():
    res= ['ses_measles_2014', 'ph_max_tem_2019', 'ph_dist_open_coast_2020',
       'ho_impr_housing_2015', 'ph_base_water_2010', 'uu_bld_count_2020',
       'ph_pm25_2016', 'ph_dist_art_surface_2015', 'ph_land_c2_2020',
       'ph_slope_2000', 'ph_climate_risk_2020', 'ph_land_c1_2019',
       'ph_bio_dvst_2015', 'ph_ndvi_2019', 'in_dist_waterway_2016',
       'uu_urb_bldg_2018', 'po_hrsl_2018', 'po_wp_2020',
       'ph_dist_herb_2015', 'in_night_light_2016', 'ph_dist_shrub_2015',
       'ses_an_visits_2014', 'ph_grd_water_2000',
       'ph_dist_woody_tree_2015', 'fs_dist_school_2020',
       'ses_impr_water_src_2014', 'ph_dist_aq_veg_2015',
       'ph_dist_inland_water_2018', 'in_dist_rd_2016', 'ses_m_lit_2014',
       'ses_pfpr_2017', 'ses_w_lit_2016', 'ph_dist_cultivated_2015',
       'ses_odef_2014', 'sh_dist_mnr_pofw_2019',
       'ph_dist_sparse_veg_2015', 'uu_bld_den_2020']
    return res

def important_96():
    res=['gabor_sc3_filter_7', 'lbpm_sc7_skew', 'sfs_sc71_std',
       'orb_sc51_max', 'lsr_sc71_line_mean', 'lbpm_sc7_max',
       'sfs_sc51_max_ratio_of_orthogonal_angles', 'orb_sc71_variance',
       'sfs_sc51_mean', 'pantex_sc7_min', 'orb_sc31_kurtosis',
       'lsr_sc51_line_mean', 'fourier_sc51_mean', 'hog_sc5_kurtosis',
       'lbpm_sc7_mean', 'sfs_sc51_min_line_length', 'pantex_sc3_min',
       'sfs_sc71_min_line_length', 'sfs_sc31_min_line_length',
       'hog_sc7_skew', 'fourier_sc71_variance', 'lsr_sc31_line_length',
       'hog_sc7_kurtosis', 'sfs_sc71_w_mean',
       'sfs_sc31_max_ratio_of_orthogonal_angles', 'fourier_sc31_mean',
       'ndvi_sc3_variance', 'sfs_sc71_max_ratio_of_orthogonal_angles',
       'orb_sc51_kurtosis', 'sfs_sc31_std', 'sfs_sc31_w_mean',
       'lac_sc3_lac', 'hog_sc5_skew', 'hog_sc5_mean',
       'lsr_sc31_line_mean', 'sfs_sc51_max_line_length',
       'fourier_sc31_variance', 'hog_sc3_mean', 'lbpm_sc3_kurtosis',
       'sfs_sc31_mean', 'lbpm_sc3_max', 'hog_sc3_kurtosis',
       'gabor_sc3_variance', 'sfs_sc71_max_line_length',
       'mean_sc3_variance', 'lbpm_sc5_kurtosis', 'hog_sc3_skew',
       'lac_sc5_lac', 'sfs_sc51_std', 'gabor_sc7_filter_14',
       'hog_sc7_variance', 'lbpm_sc3_variance', 'lsr_sc31_line_contrast',
       'gabor_sc3_filter_4', 'lbpm_sc5_max', 'lbpm_sc5_skew',
       'gabor_sc3_filter_12', 'gabor_sc3_filter_6', 'gabor_sc3_filter_10',
       'orb_sc51_mean', 'lsr_sc71_line_length', 'orb_sc31_max',
       'orb_sc71_max', 'hog_sc5_variance', 'hog_sc3_variance',
       'orb_sc71_mean', 'lbpm_sc3_skew', 'orb_sc31_mean',
       'gabor_sc5_variance', 'mean_sc5_variance', 'gabor_sc5_filter_14',
       'lbpm_sc5_variance', 'gabor_sc5_filter_13', 'gabor_sc7_filter_2',
       'gabor_sc7_filter_1', 'gabor_sc5_mean', 'gabor_sc5_filter_4',
       'gabor_sc5_filter_12', 'gabor_sc5_filter_2', 'gabor_sc5_filter_7',
       'ndvi_sc7_mean', 'gabor_sc3_filter_1', 'gabor_sc3_filter_13',
       'orb_sc51_skew', 'mean_sc7_variance', 'gabor_sc3_filter_3',
       'gabor_sc7_filter_11', 'gabor_sc3_filter_11', 'gabor_sc7_filter_5',
       'gabor_sc5_filter_3', 'gabor_sc5_filter_1', 'gabor_sc5_filter_9',
       'gabor_sc7_filter_3', 'gabor_sc7_filter_7', 'mean_sc3_mean',
       'ndvi_sc5_mean']
    return res
def important_136():
    res=['gabor_sc3_filter_7', 'lbpm_sc7_skew', 'ph_hzd_index_2011',
       'lsr_sc71_line_mean', 'orb_sc51_max', 'ph_dist_open_coast_2020',
       'in_dist_rd_intersect_2016', 'fs_dist_fs_2020', 'r',
       'lsr_sc51_line_length', 'orb_sc71_variance',
       'ph_dist_riv_network_2007', 'ho_impr_housing_2015',
       'uu_bld_den_2020', 'pantex_sc7_min', 'orb_sc31_kurtosis',
       'orb_sc51_kurtosis', 'hog_sc5_kurtosis', 'lbpm_sc7_mean',
       'sfs_sc51_min_line_length', 'sfs_sc31_min_line_length',
       'sfs_sc71_min_line_length', 'orb_sc71_kurtosis', 'pantex_sc3_min',
       'ph_land_c2_2020', 'ph_slope_2000', 'ndvi_sc3_variance',
       'pantex_sc5_min', 'hog_sc7_skew', 'ph_land_c1_2019',
       'ph_bio_dvst_2015', 'hog_sc7_kurtosis', 'ph_ndvi_2019',
       'sfs_sc71_w_mean', 'sfs_sc51_w_mean', 'uu_urb_bldg_2018',
       'sfs_sc31_max_line_length', 'ses_preg_2017', 'lbpm_sc3_mean',
       'sfs_sc51_max_ratio_of_orthogonal_angles', 'sfs_sc31_w_mean',
       'hog_sc7_max', 'sfs_sc71_max_ratio_of_orthogonal_angles',
       'hog_sc5_skew', 'po_wp_2020', 'sfs_sc31_std', 'n', 'lac_sc3_lac',
       'lsr_sc31_line_mean', 'sh_ethno_den_2020', 'in_night_light_2016',
       'sfs_sc51_max_line_length', 'hog_sc3_mean', 'lbpm_sc5_mean',
       'fourier_sc71_mean', 'hog_sc7_mean', 'lsr_sc71_line_contrast', 'b',
       'sfs_sc31_mean', 'lbpm_sc3_kurtosis', 'gabor_sc3_filter_14',
       'gabor_sc3_filter_12', 'lbpm_sc3_max', 'gabor_sc3_variance',
       'sfs_sc71_max_line_length', 'fourier_sc31_mean',
       'lsr_sc51_line_mean', 'fourier_sc51_mean', 'mean_sc3_variance',
       'fs_dist_school_2020', 'lbpm_sc5_kurtosis', 'hog_sc3_skew',
       'hog_sc5_mean', 'lac_sc5_lac', 'sfs_sc51_std', 'ph_max_tem_2019',
       'ph_dist_aq_veg_2015', 'g', 'lbpm_sc3_variance',
       'hog_sc7_variance', 'ph_dist_woody_tree_2015',
       'ph_dist_inland_water_2018', 'gabor_sc3_filter_4', 'lbpm_sc5_max',
       'lbpm_sc5_skew', 'in_dist_rd_2016', 'gabor_sc3_filter_6',
       'gabor_sc3_filter_2', 'gabor_sc3_filter_10', 'hog_sc5_variance',
       'orb_sc31_max', 'sfs_sc71_std', 'orb_sc51_mean', 'orb_sc71_max',
       'ph_dist_cultivated_2015', 'hog_sc3_variance', 'ses_w_lit_2016',
       'orb_sc71_mean', 'orb_sc31_mean', 'lbpm_sc3_skew',
       'gabor_sc5_variance', 'gabor_sc5_filter_4', 'gabor_sc5_filter_14',
       'lsr_sc71_line_length', 'lbpm_sc5_variance', 'ses_odef_2014',
       'gabor_sc7_filter_6', 'gabor_sc7_filter_2', 'gabor_sc7_filter_4',
       'gabor_sc3_filter_9', 'sh_dist_mnr_pofw_2019', 'ses_measles_2014',
       'gabor_sc5_filter_8', 'gabor_sc5_filter_2', 'gabor_sc5_filter_12',
       'ndvi_sc5_variance', 'gabor_sc3_filter_3', 'gabor_sc3_filter_13',
       'orb_sc51_skew', 'gabor_sc3_mean', 'gabor_sc3_filter_5',
       'gabor_sc3_filter_1', 'gabor_sc3_filter_11', 'gabor_sc7_filter_5',
       'mean_sc5_variance', 'gabor_sc5_filter_13', 'gabor_sc5_filter_3',
       'gabor_sc5_filter_1', 'ndvi_sc7_mean', 'gabor_sc5_filter_9',
       'gabor_sc7_filter_3', 'gabor_sc7_filter_7', 'mean_sc3_mean',
       'ph_dist_sparse_veg_2015', 'ndvi_sc5_mean', 'orb_sc31_skew']
    return res
def important_55():
    res=['uu_bld_count_2020',
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
    return res
def tif_to_df3(feature_path,feature_names):
 """
 Get tif files then return dataframe
 feature_path: sorted paths of tif files (features)
 label_path: sorted paths of tif files (label)
 """
 import numpy as np
 import pandas as pd
 import rasterio
 from rasterio.merge import merge

 RasterTiles = sorted(feature_path) #features

 # Preparing training
 raster_to_mosiac = []
 # Read the tif files with rasterio then parse into list
 for p in RasterTiles:
     raster = rasterio.open(p)
     raster_to_mosiac.append(raster)

 # Get combined numpy array from the populated list
 X, out_transform = merge(raster_to_mosiac)


 # Dataframe Setup
 #Training Data
 X_train = X[:, X[0,...]!=np.nan] #cleaning the numpy array from invalid inputs (-9999)
 X_train = np.transpose(X_train) #flip the array
 # To DataFrames- Train, Validation & Test
 df_train = pd.DataFrame(X_train,columns=feature_names)

 return df_train
