import joblib
import json
import numpy as np

def export_scalers_to_json(scaler_x_path, scaler_y_path, output_path):
    """
    Loads StandardScaler objects and saves their mean and scale (std)
    to a single JSON file for C++ to read.
    """
    try:
        # Load the fitted scalers
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        # Extract the parameters
        # .mean_ is the mean vector
        # .scale_ is the standard deviation vector
        scaler_params = {
            "x_mean": scaler_X.mean_.tolist(),
            "x_std": scaler_X.scale_.tolist(),
            "y_mean": scaler_y.mean_.tolist(),
            "y_std": scaler_y.scale_.tolist()
        }

        # Save to a JSON file
        with open(output_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
            
        print(f"Successfully exported scaler parameters to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define the paths to your saved scalers
    SCALER_X_FILE = 'scaler_X.gz'
    SCALER_Y_FILE = 'scaler_y.gz'
    
    # Define the output path for the JSON file
    OUTPUT_JSON_FILE = 'scalers.json'
    
    export_scalers_to_json(SCALER_X_FILE, SCALER_Y_FILE, OUTPUT_JSON_FILE)