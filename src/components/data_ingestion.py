import os
import sys
import pandas as pd


from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data : str = 'artifact/train_data.csv'
    test_data : str = 'artifact/test_data.csv'
    raw_data : str = 'artifact/raw_data.csv'

   

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        path_name = "artifact"
        os.makedirs(path_name, exist_ok = True)

    def get_data_ingestion(self):

        try:

            logging.info('Entered get_data_ingestion method')
            df = pd.read_csv('notebook\\StudentsPerformance.csv')


            logging.info('data ingested sucessfully')
            df.to_csv(self.ingestion_config.raw_data,index=False,header=True)

            logging.info('Initiating Train-Test Split')
            train_data,test_data = train_test_split(df,test_size=0.3,random_state=46)

            logging.info("train-test split initiated")
            train_data.to_csv(self.ingestion_config.train_data,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data,index=False,header=True)

            logging.info('train-test split completed')

            return(
                self.ingestion_config.train_data,
                self.ingestion_config.test_data
            )

        except Exception as e:
            raise CustomException(e,sys) from None
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.get_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.transform_data(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.train_model(train_arr,test_arr)


