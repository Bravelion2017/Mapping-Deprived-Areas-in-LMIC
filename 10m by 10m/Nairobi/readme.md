#### Converting GeoTif files into Dataframe
The tif_to_df_nairobi_contextual.py and tif_to_df_nairobi_covariate.py script converts the geo-encoded tif images into pandas dataframe
and stores it as a parquet file for reusability for contextual and covariate features respectively.

#### modeling_nairobi_contextual.py
This script looks at collinearity/multicollinearity present amongst the 144
contextual features using SVD and VIF.
Dimensionality reduction using PCA.
It also performs K-Means clustering, Random Forest classification, Logistic Regression and XGBoost Classifier on the contextual features.
