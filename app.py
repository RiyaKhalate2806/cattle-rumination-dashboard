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
def extract_features(window_x, window_y, window_z):
    # Safe array operations - handle empty arrays
    mean_x = np.mean(window_x) if len(window_x) > 0 else 0
    std_x = np.std(window_x) if len(window_x) > 0 else 0
    max_x = np.max(window_x) if len(window_x) > 0 else 0
    min_x = np.min(window_x) if len(window_x) > 0 else 0
    
    mean_y = np.mean(window_y) if len(window_y) > 0 else 0
    std_y = np.std(window_y) if len(window_y) > 0 else 0
    max_y = np.max(window_y) if len(window_y) > 0 else 0
    min_y = np.min(window_y) if len(window_y) > 0 else 0
    
    mean_z = np.mean(window_z) if len(window_z) > 0 else 0
    std_z = np.std(window_z) if len(window_z) > 0 else 0
    max_z = np.max(window_z) if len(window_z) > 0 else 0
    min_z = np.min(window_z) if len(window_z) > 0 else 0
    
    return [mean_x, std_x, max_x, min_x, mean_y, std_y, max_y, min_y, 
            mean_z, std_z, max_z, min_z]

# Step 2: Generate synthetic training data (replace with real labeled data)
np.random.seed(42)
def generate_synthetic_data(n_samples=10000, window_size=60):
    fs = 1 
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
X_windows, y = generate_synthetic_data(n_samples=500, window_size=1080)
X_windows_short = X_windows[:1000]  # Limit for Cloud
X_features = []
for x in X_windows_short:
    if len(x) >= 1080:  # Ensure enough data
        feats = extract_features(x[:360], x[360:720], x[720:1080])
        X_features.append(feats)
X_features = np.array(X_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print("Model F1-score:", f1_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# Step 3: Real-time prediction and daily aggregation
class RuminationMonitor:
    def __init__(self):
        self.daily_rumination = 0  # ADD THIS - tracks daily total
        X_windows, y = generate_synthetic_data(n_samples=500, window_size=1080)
        X_features = []
        y_short = []
        for i, x in enumerate(X_windows):
            if len(x) >= 1080:
                try:
                    feats = extract_features(x[:360], x[360:720], x[720:1080])
                    X_features.append(feats)
                    y_short.append(y[i])
                except:
                    continue
        
        if len(X_features) == 0:
            print("Warning: No valid features extracted, using dummy data")
            X_features = np.random.randn(100, 12)
            y_short = np.random.randint(0, 2, 100)
        else:
            X_features = np.nan_to_num(np.array(X_features), nan=0.0, posinf=0.0, neginf=0.0)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_features)
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_scaled, y_short)
    
    def predict_window(self, x, y, z):
        feats = extract_features(x[:360], y[:360], z[:360])
        feats = np.nan_to_num(np.array(feats), nan=0.0, posinf=0.0, neginf=0.0)
        feats_scaled = self.scaler.transform(feats.reshape(1, -1))
        prediction = self.model.predict(feats_scaled)[0]
        probability = self.model.predict_proba(feats_scaled)[0]
        return prediction, probability[1]
        
    def update_daily(self, is_rum):
    """Update daily rumination count using session state"""
    if is_rum:
        st.session_state.daily_rumination_total += 1
    st.session_state.rumination_counter += 1

# Example usage
# Example usage - NON-BLOCKING simulation
monitor = RuminationMonitor()  # CORRECT - no arguments

# REAL-TIME SIMULATION (runs 1 iteration per page refresh)
if st.session_state.is_monitoring:
    # Mock x,y,z accelerometer data (60 seconds @ 1Hz)
    x = np.random.randn(60) * 0.5 + 0.5 * np.sin(np.arange(60) * 0.1)
    y = np.random.randn(60) * 0.3
    z = np.ones(60) * 0.9 + np.random.randn(60) * 0.1
    
    # ML Prediction (returns prediction, confidence)
    is_rum, confidence = monitor.predict_window(x, y, z)
    
    # UPDATE SESSION STATE DIRECTLY (no daily_rumination list needed)
    if is_rum > 0.5:  # Rumination detected
        st.session_state.daily_rumination_total += 1
    st.session_state.rumination_counter += 1
    
    # Refresh every 1 second
    time.sleep(1)
    st.rerun()
    
# ========================================
# BARAMATI CATTLE RUMINATION DASHBOARD
# ========================================
st.set_page_config(page_title="🐄 Baramati Cattle Health", layout="wide")

# ONLY RUN ONCE - Initialize state
if 'initialized' not in st.session_state:
    st.session_state.daily_rumination_total = 0
    st.session_state.rumination_counter = 0
    st.session_state.is_monitoring = False
    st.session_state.initialized = True

st.title("🐄 Baramati Cattle Rumination Dashboard")
st.markdown("---")

# ========================================
# CONTROL BUTTONS - UNIQUE KEYS
# ========================================
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🚀 Start Monitoring", key="start_btn", type="primary"):
        st.session_state.is_monitoring = True
        st.rerun()

with col2:
    if st.button("🛑 Stop Monitoring", key="stop_btn"):
        st.session_state.is_monitoring = False
        st.rerun()

with col3:
    monitoring_status = "🟢 LIVE" if st.session_state.is_monitoring else "🔴 STOPPED"
    st.info(f"📊 Status: {monitoring_status} | "
            f"📈 Samples: {st.session_state.rumination_counter:,} | "
            f"⏱️ Rumination: {st.session_state.daily_rumination_total} min")

# ========================================
# ONE SINGLE METRIC - KEY PREVENTS DUPLICATES
# ========================================
st.markdown("### 📊 Key Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("🐄 Daily Rumination (min)", 
              st.session_state.daily_rumination_total,
              "400-550 Healthy Range",
              key="rumination_metric")  # UNIQUE KEY = NO DUPLICATES!

with col2:
    st.metric("🔄 Samples Processed", 
              st.session_state.rumination_counter,
              "1440/day target",
              key="samples_metric")

# ========================================
# HEALTH STATUS - CRISP ALERTS
# ========================================
st.markdown("### 🏥 Cow Health Status")
rumination = st.session_state.daily_rumination_total

if rumination < 400:
    st.error("🚨 **CRITICAL VET ALERT** - Low rumination! Check cow NOW!")
elif 400 <= rumination <= 550:
    st.success("✅ **HEALTHY COW** - Normal rumination detected")
else:
    st.warning("⚠️ **MONITOR CLOSELY** - High rumination activity")

# ========================================
# SIMULATION - RUNS ONLY WHEN MONITORING
# ========================================
if st.session_state.is_monitoring:
    # Simulate 1 minute of accelerometer data
    x = np.random.randn(60) * 0.5 + 0.5 * np.sin(np.arange(60) * 0.1)
    y = np.random.randn(60) * 0.3
    z = np.ones(60) * 0.9 + np.random.randn(60) * 0.1
    
    # ML Prediction
    is_rum, confidence = monitor.predict_window(x, y, z)
    
    # Update counters
    if is_rum > 0.5:
        st.session_state.daily_rumination_total += 1
    st.session_state.rumination_counter += 1
    
    # Refresh every second
    time.sleep(1)
    st.rerun()
else:
    st.info("👆 **Click START MONITORING** to begin live cattle tracking")
    st.stop()  # STOP EXECUTION - PREVENTS DUPLICATES!

st.markdown("---")
st.caption("🌍 24/7 Global Access | 📱 Farm Manager Mobile | 🐄 Baramati Cattle Health System v2.0")

# Chart
fig, ax = plt.subplots()
ax.bar(['Rumination', 'Healthy Range'], [monitor.daily_rumination, 475])
st.pyplot(fig)
