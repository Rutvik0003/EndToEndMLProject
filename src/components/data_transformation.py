import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = 'artifact/preprocessor.pkl'

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self):

        try:
            numerical_features = ['reading score', 'writing score']
            catagorical_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            cat_pipeline = Pipeline(

                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder())
                ]

            )

            logging.info("Cat_Pipeline Initiated")

            num_pipeline = Pipeline(

                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]

            )

            logging.info('Num_Pipeline Initiated')

            preprocessor = ColumnTransformer(
                [
                    ("Catagorical_Pipeline",cat_pipeline,catagorical_features),
                    ("Numerical_Pipeline",num_pipeline,numerical_features)
                ]
            )

            logging.info("Column Pipeline Initiated")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys) from None
    
    def transform_data(self,train_data_path,test_data_path):
        try:

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            preprocessor = self.get_preprocessor()
            target_column = 'math score'

            input_train_data_df = train_data.drop(columns=[target_column],axis=1)
            input_test_data_df = test_data.drop(columns=[target_column],axis=1)

            target_train_data_df = train_data[target_column]
            target_test_data_df = test_data[target_column]

            logging.info("Data Split Completed")

            input_train_data_df = preprocessor.fit_transform(input_train_data_df)
            input_test_data_df = preprocessor.transform(input_test_data_df)

            logging.info("Data Transformation Completed")

            train_arr = np.c_[input_train_data_df, np.array(target_train_data_df)]
            test_arr = np.c_[input_test_data_df, np.array(target_test_data_df)]

            save_obj(
                file_path = self.transformation_config.preprocessor_path,
                obj = preprocessor

            )

            logging.info("Pickel File Created")

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e,sys) from None





