import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# --- Simulated dataset generation ---
def generate_dataset(num_samples=1000):
    np.random.seed(42)
    data = {
        'soil_moisture': np.random.uniform(20, 90, num_samples),
        'temperature': np.random.uniform(10, 40, num_samples),
        'humidity': np.random.uniform(30, 90, num_samples),
        'light_intensity': np.random.uniform(200, 1000, num_samples),
        'rain_detected': np.random.choice([0, 1], size=num_samples),
        'crop_yield': np.random.uniform(1000, 5000, num_samples)  # kg per hectare
    }
    df = pd.DataFrame(data)
    return df

# --- Train AI Model ---
df = generate_dataset()

X = df.drop('crop_yield', axis=1)
y = df['crop_yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate Model ---
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"✅ Model trained with MAE: {mae:.2f} kg/hectare")

# --- Save Model ---
joblib.dump(model, 'crop_yield_predictor.pkl')

# --- Predict Sample Input ---
sample_input = pd.DataFrame([{
    'soil_moisture': 65,
    'temperature': 28,
    'humidity': 60,
    'light_intensity': 700,
    'rain_detected': 1
}])
predicted_yield = model.predict(sample_input)[0]
print(f"\n🌱 Predicted Crop Yield: {predicted_yield:.2f} kg/hectare")