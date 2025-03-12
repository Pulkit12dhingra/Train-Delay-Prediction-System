import pandas as pd
import sqlite3
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Connect to SQLite database
DB_PATH = "data/train_delays.db"

def load_data():
    """Load data from SQLite tables and merge them into a single DataFrame."""
    conn = sqlite3.connect(DB_PATH)

    # Load tables
    df_schedule = pd.read_sql("SELECT * FROM train_schedule", conn)
    df_weather = pd.read_sql("SELECT * FROM weather_conditions", conn)
    df_congestion = pd.read_sql("SELECT * FROM congestion_levels", conn)
    df_delays = pd.read_sql("SELECT * FROM train_delays", conn)

    conn.close()

    # Merge data on 'Train ID'
    df = (df_schedule
          .merge(df_weather, on="Train ID")
          .merge(df_congestion, on="Train ID")
          .merge(df_delays, on="Train ID"))
    
    return df

def preprocess_data(df):
    """Preprocess the data: feature engineering, encoding, and handling missing values."""
    
    # Convert time features to minutes
    def time_to_minutes(time_str):
        """Convert HH:MM time format to minutes since midnight."""
        h, m = map(int, time_str.split(":"))
        return h * 60 + m

    df["Scheduled Time"] = df["Scheduled Time"].apply(time_to_minutes)
    df["Actual Time"] = df["Actual Time"].apply(time_to_minutes)

    # Rename time columns to lowercase for consistency
    df.rename(columns={"Scheduled Time": "scheduled_time", "Actual Time": "actual_time"}, inplace=True)

    # Encode categorical features (Weather, Day of Week)
    categorical_features = ["Weather", "Day of Week"]
    
    encoder = OneHotEncoder(sparse_output=False, drop="first")  # ✅ Fixed 'sparse' issue
    encoded_data = encoder.fit_transform(df[categorical_features])
    
    # Convert to DataFrame and merge back
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(columns=categorical_features).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    # Drop non-useful columns
    df = df.drop(columns=["Train ID"])  # Unique ID, not useful for ML

    return df

def save_processed_data(df):
    """Save the cleaned and preprocessed data for model training."""
    df.to_csv("./data/processed/train_delays_preprocessed.csv", index=False)
    print("✅ Preprocessed data saved to './data/processed/train_delays_preprocessed.csv'")

if __name__ == "__main__":
    print("🔄 Loading data from database...")
    df_raw = load_data()
    
    print("⚙️ Preprocessing data...")
    df_processed = preprocess_data(df_raw)

    print("💾 Saving preprocessed data...")
    save_processed_data(df_processed)

    print("✅ Data preprocessing complete!")
