import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load preprocessed data
DATA_PATH = "data/processed/train_delays_preprocessed.csv"
MODEL_PATH = "model/train_delay_model.pkl"

def load_data():
    """Load preprocessed dataset for training."""
    df = pd.read_csv(DATA_PATH)
    return df

def train_model():
    """Train a regression model to predict train delays."""
    df = load_data()

    # Split features & target variable
    X = df.drop(columns=["Delay (min)"])  # Features
    y = df["Delay (min)"]  # Target

    # Split into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline (Scaling + Model)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train model
    print("🚆 Training model...")
    pipeline.fit(X_train, y_train)

    # Save trained model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
