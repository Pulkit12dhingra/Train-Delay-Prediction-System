import pandas as pd
import numpy as np
import random
import sqlite3
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of synthetic records
NUM_SAMPLES = 5000

# Possible weather conditions
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"]

# Function to generate random timestamps
def random_time():
    return datetime.strptime(f"{random.randint(0, 23)}:{random.randint(0, 59)}", "%H:%M")

# Function to simulate delays based on weather and congestion
def simulate_delay(weather, congestion):
    base_delay = np.random.normal(5, 10)  # Base delay (5 min avg, std dev 10 min)
    
    # Add delay based on weather
    weather_delay = {
        "Clear": 0,
        "Cloudy": np.random.randint(0, 3),
        "Rain": np.random.randint(5, 15),
        "Fog": np.random.randint(10, 20),
        "Storm": np.random.randint(15, 30),
        "Snow": np.random.randint(20, 40)
    }.get(weather, 0)
    
    # Add delay based on congestion
    congestion_delay = congestion * np.random.randint(2, 10)  # Higher congestion → more delay
    
    total_delay = max(0, base_delay + weather_delay + congestion_delay)  # Avoid negative delay
    return round(total_delay, 2)

# Lists to store separate data
train_schedule_data = []
weather_data = []
congestion_data = []
train_delays_data = []

# Generate synthetic dataset
for i in range(NUM_SAMPLES):
    train_id = f"T{i+1:05d}"  # Unique train ID
    scheduled_time = random_time()
    weather = random.choice(WEATHER_CONDITIONS)
    congestion = np.random.randint(0, 10)  # 0 (low congestion) to 10 (high congestion)
    day_of_week = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    delay = simulate_delay(weather, congestion)
    actual_time = scheduled_time + timedelta(minutes=delay)

    # Append data to respective lists
    train_schedule_data.append([train_id, scheduled_time.strftime("%H:%M"), day_of_week])
    weather_data.append([train_id, weather])
    congestion_data.append([train_id, congestion])
    train_delays_data.append([train_id, actual_time.strftime("%H:%M"), delay])

# Convert to DataFrames
df_train_schedule = pd.DataFrame(train_schedule_data, columns=["Train ID", "Scheduled Time", "Day of Week"])
df_weather = pd.DataFrame(weather_data, columns=["Train ID", "Weather"])
df_congestion = pd.DataFrame(congestion_data, columns=["Train ID", "Congestion Level"])
df_train_delays = pd.DataFrame(train_delays_data, columns=["Train ID", "Actual Time", "Delay (min)"])

# Save as CSV
df_train_schedule.to_csv("./data/generated/train_schedule.csv", index=False)
df_weather.to_csv("./data/generated/weather_conditions.csv", index=False)
df_congestion.to_csv("./data/generated/congestion_levels.csv", index=False)
df_train_delays.to_csv("./data/generated/train_delays.csv", index=False)

print("✅ Synthetic CSV files saved successfully!")

# -------- Store Data in SQLite Database --------
conn = sqlite3.connect("data/train_delays.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS train_schedule (
    train_id TEXT PRIMARY KEY,
    scheduled_time TEXT,
    day_of_week TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS weather_conditions (
    train_id TEXT PRIMARY KEY,
    weather TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS congestion_levels (
    train_id TEXT PRIMARY KEY,
    congestion_level INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS train_delays (
    train_id TEXT PRIMARY KEY,
    actual_time TEXT,
    delay_min REAL
)
""")

# Insert data into tables
df_train_schedule.to_sql("train_schedule", conn, if_exists="replace", index=False)
df_weather.to_sql("weather_conditions", conn, if_exists="replace", index=False)
df_congestion.to_sql("congestion_levels", conn, if_exists="replace", index=False)
df_train_delays.to_sql("train_delays", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print("✅ Data successfully stored in 'train_delays.db' SQLite database!")
