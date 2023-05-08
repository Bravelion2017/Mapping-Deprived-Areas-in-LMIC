import os
from glob import glob
import rasterio
import numpy as np
import pandas as pd
from rasterio.merge import merge
from toolbox import covariate_features, tif_to_df, contextual_features
covariate_features= covariate_features()
cf= contextual_features()
rgbn=['r','g','b','n']
test_areas= [11, 13, 20, 23, 25, 28]
areas= list(range(30))
for i in test_areas:
    areas.remove(i)
#%%
# Lagos Contextual 100m
foo=[]
for i in areas:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/spfea_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,cf)
    foo.append(area0_df)
    # df= pd.concat([df_train, df_val, df_test])
    # print(area0_df.shape)
lagos_ctx= pd.concat(foo)
lagos_ctx['type']= 'train'

foo=[]
for i in test_areas:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/spfea_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_contextual_100m/mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,cf)
    foo.append(area0_df)
lagos_ctx_test= pd.concat(foo)
lagos_ctx_test['type']= 'test'

lagos_ctx= pd.concat([lagos_ctx,lagos_ctx_test])
lagos_ctx= lagos_ctx[~(lagos_ctx==-9999.000000)]
# Check for Nulls $ replace with mean
print(lagos_ctx.isna().sum().sort_values(ascending=False))
values= {i:lagos_ctx[i].mean() for i in lagos_ctx.columns[:-1]}
lagos_ctx.loc[:,lagos_ctx.columns[:-1]]= lagos_ctx.loc[:,lagos_ctx.columns[:-1]].fillna(value=values)
print(f'Null values: {lagos_ctx.isna().sum().any()}')

# Save df
lagos_ctx.to_parquet('lagos_ctx100m_df.parquet.gzip',compression='gzip')

#%%
# Lagos Covariate 100m
foo=[]
for i in areas:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,covariate_features)
    foo.append(area0_df)
    # df= pd.concat([df_train, df_val, df_test])
    # print(area0_df.shape)
lagos_cv= pd.concat(foo)
lagos_cv['type']= 'train'

foo=[]
for i in test_areas:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/covariate_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_covariate_feature_53/mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,covariate_features)
    foo.append(area0_df)
lagos_cv_test= pd.concat(foo)
lagos_cv_test['type']= 'test'

lagos_cv= pd.concat([lagos_cv,lagos_cv_test])
lagos_cv= lagos_cv[~(lagos_cv==-9999.000000)]
# Check for Nulls $ replace with mean
print(lagos_cv.isna().sum().sort_values(ascending=False))
values= {i:lagos_cv[i].mean() for i in lagos_cv.columns[:-1]}
lagos_cv.loc[:,lagos_cv.columns[:-1]]= lagos_cv.loc[:,lagos_cv.columns[:-1]].fillna(value=values)
print(f'Null values: {lagos_cv.isna().sum().any()}')

# Save df
lagos_cv.to_parquet('lagos_cv100m_df.parquet.gzip',compression='gzip')

#%%
# Lagos RGBN 100m
foo=[]
for i in areas:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    foo.append(area0_df)
    # df= pd.concat([df_train, df_val, df_test])
    # print(area0_df.shape)
lagos_rgbn= pd.concat(foo)
lagos_rgbn['type']= 'train'

foo=[]
for i in test_areas:
    area0 = [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_100m/lag_area{i}.tif"]
    area0_mask = [f"/home/ubuntu/capstone/Lagos/lagos_rgb_100m/rgb_mask_100m/lag_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    foo.append(area0_df)
lagos_rgbn_test= pd.concat(foo)
lagos_rgbn_test['type']= 'test'

lagos_rgbn= pd.concat([lagos_rgbn,lagos_rgbn_test])
lagos_rgbn= lagos_rgbn[~(lagos_rgbn==-9999.000000)]
# Check for Nulls $ replace with mean
print(lagos_rgbn.isna().sum().sort_values(ascending=False))
values= {i:lagos_rgbn[i].mean() for i in lagos_rgbn.columns[:-1]}
lagos_rgbn.loc[:,lagos_rgbn.columns[:-1]]= lagos_rgbn.loc[:,lagos_rgbn.columns[:-1]].fillna(value=values)
print(f'Null values: {lagos_rgbn.isna().sum().any()}')

# Save df
lagos_rgbn.to_parquet('lagos_rgbn100m_df.parquet.gzip',compression='gzip')