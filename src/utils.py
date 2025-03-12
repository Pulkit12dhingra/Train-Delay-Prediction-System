import sqlite3
import pandas as pd

DB_PATH = "data/train_delays.db"

def fetch_data(query):
    """Fetch data from SQLite database using a custom query."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def time_to_minutes(time_str):
    """Convert HH:MM time format to minutes since midnight."""
    h, m = map(int, time_str.split(":"))
    return h * 60 + m

def save_to_csv(df, filename):
    """Save DataFrame to CSV."""
    df.to_csv(f"data/processed/{filename}", index=False)
    print(f"✅ Saved {filename}")

if __name__ == "__main__":
    # Example usage
    query = "SELECT * FROM train_schedule LIMIT 5"
    df_sample = fetch_data(query)
    print(df_sample)
