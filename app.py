import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# Step 1: Feature extraction function for x,y,z data (1-min windows)
def extract_features(x, y, z, window_size=60, fs=1):  # fs=1 Hz assumed
    """
    Extract features from accelerometer data.
    Assumes data is time-series with timestamps.
    """
    features = []
    n_samples = len(x)
    for i in range(0, n_samples - window_size + 1, window_size // 2):  # 50% overlap
        window_x = x[i:i+window_size]
        window_y = y[i:i+window_size]
        window_z = z[i:i+window_size]
        
        # Time-domain features
        feat = [
            np.mean(window_x), np.std(window_x), np.max(window_x), np.min(window_x),
            np.mean(window_y), np.std(window_y), np.max(window_y), np.min(window_y),
            np.mean(window_z), np.std(window_z), np.max(window_z), np.min(window_z),
            np.mean(np.abs(np.diff(window_x))), np.mean(np.abs(np.diff(window_y))), np.mean(np.abs(np.diff(window_z)))
        ]
        
        # Magnitude
        mag = np.sqrt(window_x**2 + window_y**2 + window_z**2)
        feat += [np.mean(mag), np.std(mag)]
        
        features.append(feat)
    return np.array(features)

# Step 2: Generate synthetic training data (replace with real labeled data)
np.random.seed(42)
def generate_synthetic_data(n_samples=10000, window_size=60):
    data = []
    labels = []
    for _ in range(n_samples):
        if np.random.rand() > 0.5:  # Rumination: periodic chewing ~1-2 Hz
            label = 1
            t = np.arange(window_size)
            x = 0.5 * np.sin(2 * np.pi * 1.5 * t / fs) + 0.1 * np.random.randn(window_size)
            y = 0.3 * np.sin(2 * np.pi * 1.8 * t / fs) + 0.1 * np.random.randn(window_size)
            z = 0.8 + 0.2 * np.random.randn(window_size)  # Mostly vertical
        else:  # Non-rumination: random walk/walking
            label = 0
            t = np.arange(window_size)
            x = np.cumsum(0.1 * np.random.randn(window_size))
            y = np.cumsum(0.1 * np.random.randn(window_size))
            z = 0.9 + 0.3 * np.sin(2 * np.pi * 0.5 * t / fs) + 0.1 * np.random.randn(window_size)
        data.append(np.concatenate([x, y, z]))
        labels.append(label)
    return np.array(data), np.array(labels)

# Train model
X_windows, y = generate_synthetic_data()
X_features = np.array([extract_features(x[:360], x[360:720], x[720:]) for x in X_windows])  # Simulate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print("Model F1-score:", f1_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# Step 3: Real-time prediction and daily aggregation
class RuminationMonitor:
    def __init__(self, model, scaler, healthy_min=400):  # minutes/day
        self.model = model
        self.scaler = scaler
        self.healthy_min = healthy_min
        self.daily_rumination = 0
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0)
    
    def predict_window(self, x, y, z):
        feats = extract_features(x, y, z)
        feats_scaled = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats_scaled)[:, 1]
        return np.mean(probs > 0.5)  # Fraction ruminating
    
    def update_daily(self, is_ruminating, duration_min=1):
        if is_ruminating:
            self.daily_rumination += duration_min
        if (datetime.now() - self.start_time).total_seconds() / 3600 >= 24:
            self.check_alert()
            self.reset_daily()
    
    def check_alert(self):
        if self.daily_rumination < self.healthy_min:
            print(f"ALERT: Low rumination! {self.daily_rumination} min (healthy: {self.healthy_min}+)")
        else:
            print(f"Healthy: {self.daily_rumination} min")
    
    def reset_daily(self):
        self.daily_rumination = 0
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0)

# Example usage
monitor = RuminationMonitor(model, scaler)
# Simulate stream: replace with sensor data loop
for _ in range(1440):  # 1440 min/day
    # Mock x,y,z (replace with real sensor read)
    x = np.random.randn(60) * 0.5 + 0.5 * np.sin(np.arange(60) * 0.1)
    y = np.random.randn(60) * 0.3
    z = np.ones(60) * 0.9 + np.random.randn(60) * 0.1
    is_rum = monitor.predict_window(x, y, z)
    monitor.update_daily(is_rum > 0.5)

# Dashboard with Streamlit
st.title("Cattle Rumination Dashboard")
daily_time = st.metric("Daily Rumination (min)", monitor.daily_rumination, "400-550")
if daily_time.value < 400:
    st.error("ALERT: Below healthy range!")
else:
    st.success("Healthy rumination!")

# Chart
fig, ax = plt.subplots()
ax.bar(['Rumination', 'Healthy Range'], [monitor.daily_rumination, 475])
st.pyplot(fig)
