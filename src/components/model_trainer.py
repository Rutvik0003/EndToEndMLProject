import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig:
    model_trainer_path : str = os.path.join('artifact', 'model_trainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def train_model(self,train_arr, test_arr):

        try:
            models = {
                "LinearRegression" : LinearRegression(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForest" : RandomForestRegressor(),
                "GredientBoosting" : GradientBoostingRegressor(),
                "AdaBoosting" : AdaBoostRegressor(),
                "XGBoost" : XGBRegressor(),
                "K-Nearest-Neighbour" : KNeighborsRegressor()
            }

            logging.info('train and test array division initiated')

            X_train,X_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            logging.info("Array Split Completed")

            logging.info("Models Report Initated")

            model_report : dict = evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test, models = models)

            logging.info("Models Report Completed")

            best_model_score = max(sorted(list(model_report.values())))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info("Evaluated Best Model")

            save_obj(
                file_path=self.trainer_config.model_trainer_path,
                obj= best_model
            )

            logging.info("Model Trainer Pickel file made")



        except Exception as e:
            raise CustomException(e,sys) from None 
