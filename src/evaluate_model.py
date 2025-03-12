import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

DATA_PATH = "data/processed/train_delays_preprocessed.csv"
MODEL_PATH = "model/train_delay_model.pkl"

def load_data():
    """Load dataset for evaluation."""
    df = pd.read_csv(DATA_PATH)
    return df

def evaluate_model():
    """Evaluate model performance using test data."""
    df = load_data()
    
    # Load trained model
    model = joblib.load(MODEL_PATH)
    
    # Split features & target variable
    X = df.drop(columns=["Delay (min)"])
    y = df["Delay (min)"]
    
    # Predict
    y_pred = model.predict(X)

    # Compute metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"📊 Model Evaluation Results:")
    print(f"➡️ MAE: {mae:.2f} minutes")
    print(f"➡️ RMSE: {rmse:.2f} minutes")
    print(f"➡️ R² Score: {r2:.2f}")

if __name__ == "__main__":
    evaluate_model()
