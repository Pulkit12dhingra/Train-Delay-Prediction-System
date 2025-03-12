import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "model/train_delay_model.pkl"

def get_model_features():
    """Load the trained model and return feature names used during training."""
    model = joblib.load(MODEL_PATH)
    return model.feature_names_in_  # Fetch feature names

def predict_delay(input_data):
    """Predict train delay based on input features."""
    model = joblib.load(MODEL_PATH)
    
    # Load correct feature names from the trained model
    expected_features = get_model_features()

    # Convert input data into DataFrame with correct feature names
    input_df = pd.DataFrame([input_data])

    # Ensure correct column order and missing features are filled with 0
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Predict delay
    predicted_delay = model.predict(input_df)[0]
    
    return round(predicted_delay, 2)

if __name__ == "__main__":
    # Example input (should match feature structure)
    sample_input = {
        "scheduled_time": 900,  # 15:00 (3:00 PM)
        "actual_time": 915,  # 15:15 (3:15 PM)
        "Congestion Level": 5,
        "Weather_Cloudy": 0,
        "Weather_Rain": 1,
        "Weather_Fog": 0,
        "Day of Week_Monday": 0,
        "Day of Week_Tuesday": 1,
        "Day of Week_Wednesday": 0,
        "Day of Week_Thursday": 0,
        "Day of Week_Friday": 0,
        "Day of Week_Saturday": 0,
        "Day of Week_Sunday": 0
    }
    
    delay_prediction = predict_delay(sample_input)
    print(f"🚆 Predicted Delay: {delay_prediction} minutes")
