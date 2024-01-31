"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
import numpy

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from azureml.training.tabular._diagnostics import logging_utilities
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger("azureml.training.tabular")

TARGET_COL = "VISIT_TIME"

NUMERIC_COLS = [
    "distance", "dropoff_latitude", "dropoff_longitude", "passengers", "pickup_latitude",
    "pickup_longitude", "pickup_weekday", "pickup_month", "pickup_monthday", "pickup_hour",
    "pickup_minute", "pickup_second", "dropoff_weekday", "dropoff_month", "dropoff_monthday",
    "dropoff_hour", "dropoff_minute", "dropoff_second"
]

CAT_NOM_COLS = [
    "store_forward", "vendor"
]

CAT_ORD_COLS = [
]


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    args = parser.parse_args()

    return args

def get_mapper_0(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from azureml.training.tabular.featurization.utilities import wrap_in_list
    from numpy import uint8
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': CountVectorizer,
                'analyzer': 'word',
                'binary': True,
                'decode_error': 'strict',
                'dtype': numpy.uint8,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 1.0,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (1, 1),
                'preprocessor': None,
                'stop_words': None,
                'strip_accents': None,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': wrap_in_list,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_1(column_names):
    from azureml.training.tabular.featurization.categorical.cat_imputer import CatImputer
    from azureml.training.tabular.featurization.datetime.datetime_transformer import DateTimeFeaturesTransformer
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': CatImputer,
                'copy': True,
            },
            {
                'class': StringCastTransformer,
            },
            {
                'class': DateTimeFeaturesTransformer,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_2(column_names):
    from sklearn.impute import SimpleImputer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': SimpleImputer,
                'add_indicator': False,
                'copy': True,
                'fill_value': None,
                'missing_values': numpy.nan,
                'strategy': 'mean',
                'verbose': 0,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_3(column_names):
    from azureml.training.tabular.featurization.generic.imputation_marker import ImputationMarker
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': ImputationMarker,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper

def generate_data_transformation_config():
    '''
    Specifies the featurization step in the final scikit-learn pipeline.
    
    If you have many columns that need to have the same featurization/transformation applied (for example,
    50 columns in several column groups), these columns are handled by grouping based on type. Each column
    group then has a unique mapper applied to all columns in the group.
    '''
    from sklearn.pipeline import FeatureUnion
    
    column_group_3 = [['GENDERID'], ['PROVIDERAGE'], ['TENURE'], ['PROD_DSNP'], ['PROD_CKD'], ['PROD_DEE'], ['PROD_FOBT'], ['PROD_SPIROMETRY'], ['PROD_HBA1C'], ['PROD_HHRA'], ['PROD_MHC'], ['PROD_MTM'], ['PROD_OMW'], ['PROD_PAD']]
    
    column_group_2 = [['APPT_LAT'], ['APPT_LNG'], ['GENDERID'], ['PROVIDERAGE'], ['TENURE'], ['PROD_DSNP'], ['PROD_CKD'], ['PROD_DEE'], ['PROD_FOBT'], ['PROD_SPIROMETRY'], ['PROD_HBA1C'], ['PROD_HHRA'], ['PROD_MHC'], ['PROD_MTM'], ['PROD_OMW'], ['PROD_PAD'], ['VISIT_TIME_MEAN'], ['VISIT_COUNT']]
    
    column_group_1 = ['SERVICE_DAY', 'DATEOFBIRTH', 'HIRINGDATE']
    
    column_group_0 = ['STATE', 'CLIENT', 'LOB', 'EMPLOYEETYPENAME', 'PROVIDERSTATE', 'DEGREE']
    
    feature_union = FeatureUnion([
        ('mapper_0', get_mapper_0(column_group_0)),
        ('mapper_1', get_mapper_1(column_group_1)),
        ('mapper_2', get_mapper_2(column_group_2)),
        ('mapper_3', get_mapper_3(column_group_3)),
    ])
    return feature_union

def generate_preprocessor_config():
    '''
    Specifies a preprocessing step to be done after featurization in the final scikit-learn pipeline.
    
    Normally, this preprocessing step only consists of data standardization/normalization that is
    accomplished with sklearn.preprocessing. Automated ML only specifies a preprocessing step for
    non-ensemble classification and regression models.
    '''
    from sklearn.preprocessing import Normalizer
    
    preproc = Normalizer(
        copy=True,
        norm='l1'
    )
    
    return preproc
    
    
def generate_algorithm_config():
    '''
    Specifies the actual algorithm and hyperparameters for training the model.
    
    It is the last stage of the final scikit-learn pipeline. For ensemble models, generate_preprocessor_config_N()
    (if needed) and generate_algorithm_config_N() are defined for each learner in the ensemble model,
    where N represents the placement of each learner in the ensemble model's list. For stack ensemble
    models, the meta learner generate_algorithm_config_meta() is defined.
    '''
    from xgboost.sklearn import XGBRegressor
    
    algorithm = XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.5,
        eta=0.2,
        gamma=0,
        gpu_id=-1,
        grow_policy='lossguide',
        importance_type='gain',
        interaction_constraints='',
        learning_rate=0.200000003,
        max_bin=63,
        max_delta_step=0,
        max_depth=2,
        max_leaves=0,
        min_child_weight=1,
        missing=numpy.nan,
        monotone_constraints='()',
        n_estimators=200,
        n_jobs=0,
        num_parallel_tree=1,
        objective='reg:squarederror',
        random_state=0,
        reg_alpha=2.3958333333333335,
        reg_lambda=0.9375,
        scale_pos_weight=1,
        subsample=0.9,
        tree_method='hist',
        validate_parameters=1,
        verbose=-10,
        verbosity=0
    )
    
    return algorithm

def build_model_pipeline():
    '''
    Defines the scikit-learn pipeline steps.
    '''
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('featurization', generate_data_transformation_config()),
            ('preproc', generate_preprocessor_config()),
            ('model', generate_algorithm_config()),
        ]
    )
    
    return pipeline


def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data.drop(columns=[TARGET_COL])
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    model = model_pipeline.fit(X_train, y_train)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    main(args)

    mlflow.end_run()
    