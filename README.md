# 🚆 Train Delay Prediction System

A machine learning model to predict train delays using historical and real-time data.

## 📌 Features
- **Predicts train delays** based on schedule, weather, and congestion.
- **Trains multiple models** (Random Forest, XGBoost, Linear Regression).
- **Exposes a REST API** for real-time inference.
- **Stores data in SQLite** for easy management.

## 📂 Project Structure

```bash
train-delay-prediction/ 
│── data/ # Raw & processed datasets 
│── model/ # Saved ML models 
│── src/ # Core scripts 
│ ├── generate_data.py # Synthetic data generation 
│ ├── data_preprocessing.py # Prepares data for training 
│ ├── train_model.py # Trains ML models 
│ ├── evaluate_model.py # Evaluates performance 
│ ├── predict.py # Runs predictions 
│── notebooks/ # Jupyter notebooks for EDA & model testing 
│── requirements.txt # Dependencies 
│── README.md # Project documentation
├── app.py # REST API using Flask 
```

## 🛠 Setup & Installation
1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/train-delay-prediction.git
cd train-delay-prediction
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the API
```bash
python src/app.py
```

Predict Train Delay
```bash
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"scheduled_time":900,"actual_time":915,"Congestion Level":5,"Weather_Cloudy":0,"Weather_Rain":1}'
```

Response
```bash
{ "predicted_delay_minutes": 7.85, "status": "success" }
```

📊 Notebooks
eda.ipynb → Exploratory Data Analysis
model_testing.ipynb → Train & compare models


📌 Next Steps
✅ Improve model accuracy
✅ Deploy API on AWS/GCP
✅ Build a frontend dashboard