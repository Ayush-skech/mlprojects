import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

# Prediction pipeline for loading model and preprocessing objects, transforming input features, and generating predictions
class PredictPipeline:
    def __init__(self):
        pass  

    def predict(self, features):
        """
        Main Features:
        - Loads the trained model and preprocessor from the saved files.
        - Transforms the input features using the preprocessor.
        - Generates predictions using the loaded model.
        """
        try:
            # Paths to the saved model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")

            # Load the trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)

            # Generate predictions
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            # Handle exceptions using the custom exception class
            raise CustomException(e, sys)


# Class to structure custom input data for prediction
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        # Initialize all input features
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Main Features:
        - Converts user input into a pandas DataFrame.
        - This structured DataFrame can be directly fed into the prediction pipeline.
        """
        try:
            # Create a dictionary with the input feature values
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Handle exceptions using the custom exception class
            raise CustomException(e, sys)
