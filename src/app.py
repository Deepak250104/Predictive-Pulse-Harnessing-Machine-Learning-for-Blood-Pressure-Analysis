import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# Initialize the Flask application
app = Flask(__name__, template_folder='templates')

# Define the path to the model file
# Assuming app.py is in project-root/src/ and model is in project-root/models/
# This calculates the path relative to the current script's location.
current_dir = os.path.dirname(__file__) # This is project-root/src/
project_root_dir = os.path.dirname(current_dir) # This is project-root/
MODEL_PATH = os.path.join(project_root_dir, 'models', 'random_forest_model.pkl')

# Load the trained model
model = None # Initialize model to None
try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file DOES NOT EXIST at {MODEL_PATH}.")
        print("Please ensure 'random_forest_model.pkl' is located in 'project-root/models/'.")
    else:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error (FileNotFoundError): Model file not found at {MODEL_PATH}.")
    print("Double-check the path and file name.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    # This error (e.g., "STACK_GLOBAL requires str") often indicates a version mismatch
    # between the Python/library environment where the model was saved and where it's being loaded.
    # To resolve this:
    # 1. Ensure your scikit-learn, pandas, and numpy versions in this environment
    #    match the versions used when the model was originally trained and saved.
    #    You might need to create a virtual environment and install specific versions.
    # 2. If version matching is difficult, try re-training and re-saving your model
    #    in the same environment where you intend to run this Flask app.
    print("This might be due to an incompatible pickle version or a corrupted file.")


# Define all possible one-hot encoded columns in the exact order the model expects.
# This order is crucial and must match the training data's column order.
# Based on the unique values provided in the prompt.
# If your model was trained with a different order or different one-hot encoding,
# you will need to adjust this list.
EXPECTED_COLUMNS = [
    'Age_18-34', 'Age_35-50', 'Age_51-64', 'Age_65+',
    'Severity_Mild', 'Severity_Moderate', 'Severity_Severe',
    'Whendiagnoused_1 - 5 Years', 'Whendiagnoused_<1 Year', 'Whendiagnoused_>5 Years',
    'Systolic_100+', 'Systolic_111 - 120', 'Systolic_121 - 130', 'Systolic_130+',
    'Diastolic_100+', 'Diastolic_130+', 'Diastolic_70 - 80', 'Diastolic_81 - 90', 'Diastolic_91 - 100',
    'Gender_Female', 'Gender_Male',
    'History_No', 'History_Yes',
    'Patient_No', 'Patient_Yes',
    'TakeMedication_No', 'TakeMedication_Yes',
    'BreathShortness_No', 'BreathShortness_Yes',
    'VisualChanges_No', 'VisualChanges_Yes',
    'NoseBleeding_No', 'NoseBleeding_Yes',
    'ControlledDiet_No', 'ControlledDiet_Yes'
]

def preprocess_input(data):
    """
    Preprocesses the raw input data from the form into a one-hot encoded DataFrame
    that the model expects.

    Args:
        data (dict): A dictionary containing input values from the HTML form.

    Returns:
        pandas.DataFrame: A DataFrame with one-hot encoded features, ready for prediction.
    """
    # Create an empty DataFrame with all expected columns, initialized to 0
    processed_df = pd.DataFrame(0, index=[0], columns=EXPECTED_COLUMNS)

    # Map form data to one-hot encoded columns
    for feature, value in data.items():
        # Construct the column name for one-hot encoding
        col_name = f"{feature}_{value}"
        # For values like '<1 Year' or '>5 Years', ensure the column name matches
        if value == '<1 Year':
            col_name = f"{feature}_<1 Year"
        elif value == '>5 Years':
            col_name = f"{feature}_>5 Years"
        elif value == '1 - 5 Years':
            col_name = f"{feature}_1 - 5 Years"
        elif value == '111 - 120':
            col_name = f"{feature}_111 - 120"
        elif value == '121 - 130':
            col_name = f"{feature}_121 - 130"
        elif value == '70 - 80':
            col_name = f"{feature}_70 - 80"
        elif value == '81 - 90':
            col_name = f"{feature}_81 - 90"
        elif value == '91 - 100':
            col_name = f"{feature}_91 - 100"


        # Set the corresponding column to 1 if it exists in our expected columns
        if col_name in processed_df.columns:
            processed_df[col_name] = 1
        else:
            print(f"Warning: Column '{col_name}' generated from input '{feature}:{value}' not found in expected columns. This might indicate a mismatch in input data or expected features.")

    return processed_df

@app.route('/')
def index():
    """Renders the main index.html page."""
    return render_template('index.html')

@app.route('/details')
def details():
    """Renders the details.html page where users input data."""
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request.
    Retrieves form data, preprocesses it, makes a prediction, and renders the result.
    """
    if model is None:
        # If model failed to load, display an informative error on the page
        return render_template('prediction.html', prediction="Error: Model could not be loaded. Please check the server logs for details.")

    try:
        # Get all form data
        form_data = request.form.to_dict()
        print(f"Received form data: {form_data}")

        # Preprocess the input data
        processed_input = preprocess_input(form_data)
        print(f"Processed input for model: {processed_input.to_dict()}")

        # Make prediction
        prediction_result = model.predict(processed_input)[0]
        print(f"Prediction result: {prediction_result}")

        # Render the prediction.html template with the result
        return render_template('prediction.html', prediction=prediction_result)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Display a more user-friendly error message on the page
        return render_template('prediction.html', prediction=f"An error occurred during prediction: {e}. Please ensure all fields are filled correctly.")

# Main function to run the Flask application
if __name__ == '__main__':
    # Run the app in debug mode. Set debug=False for production.
    app.run(debug=True)
