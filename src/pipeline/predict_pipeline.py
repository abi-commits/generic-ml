import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.exception import customException
from src.utils import load_object


class PredictPipeline:
    def __init__(self,model_path="artifacts/model.pkl",preprocessor_path="artifacts/preprocessor.pkl"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def predict(self,features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise customException(e, sys)   

@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {"gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise customException(e, sys)