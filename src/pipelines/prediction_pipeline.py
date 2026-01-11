import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

class PredictionPipeline:
    def __init__(self):
        pass

    def predict_data(self,data):

        model_path = 'artifact/model_trainer.pkl'
        preprocessor_path = 'artifact/preprocessor.pkl'

        model = load_obj(
            file_path = model_path,
        )

        preprocessor = load_obj(
            file_path=preprocessor_path
        )

        logging.info("Models are loaded")
        data_processed = preprocessor.transform(data)

        result = model.predict(data_processed)

        logging.info("Predicted New Result")

        return result[0]

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        input_data = {
            "gender": self.gender,
            "race/ethnicity": self.race_ethnicity,
            "parental level of education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test preparation course": self.test_preparation_course,
            "reading score": self.reading_score,
            "writing score": self.writing_score
        }

        return pd.DataFrame([input_data])
