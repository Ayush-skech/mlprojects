import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# Function to save any Python object (like a trained model) to disk using pickle
def save_object(file_path, obj):
    try:
        # Create directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save object in binary write mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Custom exception handling
        raise CustomException(e, sys)
    
# Function to train, tune, and evaluate multiple machine learning models
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Main Features:
    - Iterates over multiple models and their parameter grids.
    - Performs hyperparameter tuning using GridSearchCV.
    - Fits the best model for each algorithm.
    - Calculates and stores R² scores for all models.
    - Returns a dictionary of model names and their test scores.
    """
    try:
        report = {}

        # Iterate over each model provided
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Hyperparameter tuning using GridSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Update the model with best found parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on train and test datasets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Custom exception handling
        raise CustomException(e, sys)
    
# Function to load a saved object (like a trained model) from disk
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Custom exception handling
        raise CustomException(e, sys)
