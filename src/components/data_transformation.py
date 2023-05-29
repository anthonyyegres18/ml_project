import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from exception import CustomException
from logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  
from src.utils import save_object 
from imblearn.over_sampling import SMOTE


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
     self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['amount', 'oldbalanceOrg',
                                  'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            categorical_columns = ['type']

            num_pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
            steps=[
                ('one_hot_encoder', OneHotEncoder())
            ]
            )

            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            preprocessor = ColumnTransformer(
            [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ]
            )

            return preprocessor
       
        except Exception as error:
            raise CustomException(error, sys)
        

    def balance_data(self, X_train, y_train):

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        return X_res, y_res
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column = 'isFraud'

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_train_df_preprocessing = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_preprocessing = preprocessing_obj.transform(input_feature_test_df) 

            X_train_balance, y_train_balance = self.balance_data(input_feature_train_df_preprocessing, target_feature_train_df)

            print(X_train_balance)
            print(type(X_train_balance))
            print(len(y_train_balance))

            train_arr = [X_train_balance, np.array(y_train_balance)]
            test_arr = [input_feature_test_df_preprocessing, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as error:

            raise CustomException(error, sys)
        

