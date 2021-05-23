from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import argparse
import requests
import tempfile
import logging
import sklearn
import os


logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

logger.info(f'Using Sklearn version: {sklearn.__version__}')


if __name__ == '__main__':
    logger.info('Sklearn Preprocessing Job [Start]')
    base_dir = '/opt/ml/processing'

    df = pd.read_csv(f'{base_dir}/input/abalone.csv')
    y = df.pop('rings')
    cols = df.columns
    logger.info(f'Columns = {cols}')

    numeric_features = list(df.columns)
    numeric_features.remove('sex')
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                          ('scaler', StandardScaler())])

    categorical_features = ['sex']
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocess = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), 
                                                 ('cat', categorical_transformer, categorical_features)])

    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    pd.DataFrame(train).to_csv(f'{base_dir}/train/train.csv', header=False, index=False)
    pd.DataFrame(validation).to_csv(f'{base_dir}/validation/validation.csv', header=False, index=False)
    pd.DataFrame(test).to_csv(f'{base_dir}/test/test.csv', header=False, index=False)
    logger.info('Sklearn Preprocessing Job [End]')
