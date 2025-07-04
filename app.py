# Import necessary libraries for web application and data processing
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)

# Assign the Flask app to a variable for routing
app = application

# ============================
# Route for the Home Page
# ============================
@app.route('/')
def index():
    """
    Home page route. It renders the landing page (index.html).
    """
    return render_template('index.html')

# ============================
# Route for Prediction
# ============================
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Prediction route:
    - GET request: Loads the form page for user input.
    - POST request: Takes user input, processes it, predicts the result, and returns the prediction to the form.
    """
    if request.method == 'GET':
        # If the request is GET, simply render the input form page
        return render_template('home.html')
    else:
        # If the request is POST, collect user input from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            # Note: Inputs for reading and writing scores seem to be swapped here (possible correction needed)
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        # Convert input data into a pandas DataFrame (required format for the pipeline)
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Create an instance of the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        # Generate predictions using the pipeline
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Return the result to the form page to display the prediction
        return render_template('home.html', results=results[0])

# ============================
# Entry point of the application
# ============================
if __name__ == "__main__":
    # Run the Flask app on host 0.0.0.0 to make it accessible across networks
    app.run(host="0.0.0.0")
