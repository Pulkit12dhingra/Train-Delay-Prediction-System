from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load the best trained model dynamically
MODEL_DIR = "model"
MODEL_FILE = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_model_")][0]  # Get the best saved model
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
model = joblib.load(MODEL_PATH)

# Get expected features from the trained model
EXPECTED_FEATURES = model.feature_names_in_

# Initialize Flask App
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"message": "🚆 Train Delay Prediction API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """Predict train delay based on input features."""
    try:
        # Get JSON data
        data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Validate and reformat features to match the trained model
        input_df = input_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        # Make Prediction
        predicted_delay = model.predict(input_df)[0]
        
        return jsonify({
            "predicted_delay_minutes": round(predicted_delay, 2),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failure"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
