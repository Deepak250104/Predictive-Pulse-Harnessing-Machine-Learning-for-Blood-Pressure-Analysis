import os
import joblib  # Changed from pickle to joblib
import pickle  # Need pickle for loading the preprocessor
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# Initialize the Flask application
app = Flask(__name__, template_folder='templates')

# Define the path to the model file and preprocessor
# Assuming app.py is in project-root/src/ and model is in project-root/models/
# This calculates the path relative to the current script's location.
current_dir = os.path.dirname(__file__) # This is project-root/src/
project_root_dir = os.path.dirname(current_dir) # This is project-root/
MODEL_PATH = os.path.join(project_root_dir, 'models', 'random_forest_model.pkl')
PREPROCESSOR_PATH = os.path.join(project_root_dir, 'models', 'fitted_preprocessor.pkl')

# Load the trained model and preprocessor
model = None # Initialize model to None
preprocessor = None # Initialize preprocessor to None
# Load the trained model
try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file DOES NOT EXIST at {MODEL_PATH}.")
        print("Please ensure 'random_forest_model.pkl' is located in 'project-root/models/'.")
    else:
        # Changed from pickle.load to joblib.load
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error (FileNotFoundError): Model file not found at {MODEL_PATH}.")
    print("Double-check the path and file name.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    # Updated error message to reflect joblib usage
    print("This might be due to incompatible library versions or a corrupted file.")
    print("Ensure scikit-learn, pandas, and numpy versions match between training and deployment.")

# Load the fitted preprocessor
try:
    print(f"Attempting to load preprocessor from: {PREPROCESSOR_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Error: Preprocessor file DOES NOT EXIST at {PREPROCESSOR_PATH}.")
        print("Please ensure 'fitted_preprocessor.pkl' is located in 'project-root/models/'.")
    else:
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded successfully from {PREPROCESSOR_PATH}")
        
        # Debug: Check what the preprocessor expects
        if hasattr(preprocessor, 'feature_names_in_'):
            print(f"Preprocessor was fitted on features: {preprocessor.feature_names_in_}")
        
        # Debug: Check what the model expects
        if model is not None and hasattr(model, 'feature_names_in_'):
            print(f"Model expects features: {model.feature_names_in_}")
            
except FileNotFoundError:
    print(f"Error (FileNotFoundError): Preprocessor file not found at {PREPROCESSOR_PATH}.")
    print("Double-check the path and file name.")
except Exception as e:
    print(f"An unexpected error occurred while loading the preprocessor: {e}")
    print("This might be due to incompatible library versions or a corrupted file.")


# Define the mapping from numerical predictions to human-readable labels
# Based on your unique values: ['HYPERTENSION (Stage-1)' 'HYPERTENSION (Stage-2)' 'HYPERTENSIVE CRISIS' 'NORMAL']
# You might need to adjust this mapping based on how your target was encoded
PREDICTION_LABELS = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)', 
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

def convert_prediction_to_label(numerical_prediction):
    """
    Converts numerical prediction to human-readable label.
    
    Args:
        numerical_prediction: The numerical output from the model
        
    Returns:
        str: Human-readable prediction label
    """
    if numerical_prediction in PREDICTION_LABELS:
        return PREDICTION_LABELS[numerical_prediction]
    else:
        # If prediction doesn't match expected values, return as is with warning
        print(f"Warning: Unexpected prediction value: {numerical_prediction}")
        return f"Unknown prediction: {numerical_prediction}"

def preprocess_input(data):
    """
    Preprocesses the raw input data from the form using the fitted preprocessor
    that was used during training.

    Args:
        data (dict): A dictionary containing input values from the HTML form.

    Returns:
        pandas.DataFrame: A DataFrame with numerical features, ready for prediction.
    """
    # Create a DataFrame with the original categorical features
    # Make sure the column order matches what the preprocessor expects
    feature_order = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity', 
                    'BreathShortness', 'VisualChanges', 'NoseBleeding', 'Whendiagnoused', 
                    'Systolic', 'Diastolic', 'ControlledDiet']
    
    processed_data = {}
    
    # Map form data to the expected feature names in the correct order
    for feature in feature_order:
        if feature in data:
            processed_data[feature] = data[feature]
        else:
            print(f"Warning: Expected feature '{feature}' not found in form data")
            # Handle missing data - you might want to set a default value
            processed_data[feature] = None
    
    # Add the target column 'Stages' with a dummy value since preprocessor expects it
    # This will be ignored during transformation of features
    processed_data['Stages'] = 'NORMAL'  # dummy value
    
    # Create DataFrame with single row
    input_df = pd.DataFrame([processed_data])
    
    print(f"Input DataFrame before preprocessing:")
    print(input_df)
    print(f"Input DataFrame columns: {list(input_df.columns)}")
    
    # Apply the fitted preprocessor to transform categorical to numerical
    if preprocessor is not None:
        try:
            # Transform the input using the fitted preprocessor
            processed_array = preprocessor.transform(input_df)
            
            # Get feature names from the preprocessor
            if hasattr(preprocessor, 'get_feature_names_out'):
                all_feature_names = preprocessor.get_feature_names_out()
                print(f"All feature names from preprocessor: {all_feature_names}")
                
                # Convert to DataFrame with all feature names first
                processed_df_full = pd.DataFrame(processed_array, columns=all_feature_names)
                
                # Now map to model's expected feature names
                # The model expects features without prefixes
                model_feature_names = []
                processed_data_for_model = {}
                
                for original_name in all_feature_names:
                    # Remove prefixes to match model expectations
                    if original_name.startswith('ord__'):
                        # For ordinal features, remove the 'ord__' prefix
                        clean_name = original_name.replace('ord__', '')
                    elif original_name.startswith('ohe__'):
                        # For one-hot encoded features, remove the 'ohe__' prefix
                        clean_name = original_name.replace('ohe__', '')
                    else:
                        clean_name = original_name
                    
                    # Skip 'Stages' related features as they're not needed for prediction
                    if not clean_name.startswith('Stages'):
                        model_feature_names.append(clean_name)
                        processed_data_for_model[clean_name] = processed_df_full[original_name].iloc[0]
                
                print(f"Model feature names after cleaning: {model_feature_names}")
                
                # Create final DataFrame with model's expected feature names
                final_df = pd.DataFrame([processed_data_for_model])
                
                # Ensure column order matches what model expects (if we have that info)
                if hasattr(model, 'feature_names_in_'):
                    expected_features = list(model.feature_names_in_)
                    print(f"Model expects these features in this order: {expected_features}")
                    
                    # Reorder columns to match model expectations
                    try:
                        final_df = final_df[expected_features]
                    except KeyError as e:
                        print(f"Warning: Some expected features missing: {e}")
                        # Keep available features in the expected order
                        available_features = [f for f in expected_features if f in final_df.columns]
                        final_df = final_df[available_features]
                
                print(f"Final processed DataFrame:")
                print(final_df)
                print(f"Final DataFrame shape: {final_df.shape}")
                print(f"Final feature names: {list(final_df.columns)}")
                
                return final_df
                
            else:
                # Fallback method if get_feature_names_out is not available
                print("Warning: Using fallback preprocessing method")
                
                # Based on your training setup, create feature names manually
                ordinal_features = ['Age', 'Severity', 'Whendiagnoused', 'Systolic', 'Diastolic']
                nominal_features = ['Gender', 'History', 'Patient', 'TakeMedication', 
                                  'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet']
                
                # Create column names as they would appear after preprocessing
                feature_names = ordinal_features.copy()
                
                # Add one-hot encoded feature names
                for feature in nominal_features:
                    if feature == 'Gender':
                        feature_names.extend(['Gender_Female', 'Gender_Male'])
                    else:
                        feature_names.extend([f'{feature}_No', f'{feature}_Yes'])
                
                # Exclude 'Stages' columns
                processed_array_features = processed_array[:, :len(feature_names)]
                processed_df = pd.DataFrame(processed_array_features, columns=feature_names)
                
                return processed_df
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise e
    else:
        raise ValueError("Preprocessor not loaded. Cannot transform input data.")

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
    Retrieves form data, preprocesses it using the fitted preprocessor, makes a prediction, and renders the result.
    """
    if model is None:
        # If model failed to load, display an informative error on the page
        return render_template('prediction.html', prediction="Error: Model could not be loaded. Please check the server logs for details.")
    
    if preprocessor is None:
        # If preprocessor failed to load, display an informative error on the page
        return render_template('prediction.html', prediction="Error: Preprocessor could not be loaded. Please check the server logs for details.")

    try:
        # Get all form data
        form_data = request.form.to_dict()
        print(f"Received form data: {form_data}")

        # Preprocess the input data using the fitted preprocessor
        processed_input = preprocess_input(form_data)
        print(f"Processed input for model: {processed_input.to_dict()}")

        # Make prediction (numerical)
        numerical_prediction = model.predict(processed_input)[0]
        print(f"Numerical prediction result: {numerical_prediction}")
        
        # Convert numerical prediction to human-readable label
        prediction_result = convert_prediction_to_label(numerical_prediction)
        print(f"Final prediction result: {prediction_result}")

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