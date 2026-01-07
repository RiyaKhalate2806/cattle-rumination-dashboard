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
monitor = RuminationMonitor()  # CORRECT - no arguments
# Simulate stream: replace with sensor data loop
daily_rumination = []  # ADD THIS - initialize list
for _ in range(1440):  # 1440 min/day
    # Mock x,y,z (replace with real sensor read)
    x = np.random.randn(60) * 0.5 + 0.5 * np.sin(np.arange(60) * 0.1)
    y = np.random.randn(60) * 0.3
    z = np.ones(60) * 0.9 + np.random.randn(60) * 0.1
    
    is_rum, confidence = monitor.predict_window(x, y, z)  # Returns tuple
    daily_rumination.append(is_rum)  # COLLECT rumination data
    
    # Update every 60 minutes (not every minute)
    if len(daily_rumination) % 60 == 0:
        avg_rum = np.mean(daily_rumination[-60:])  # Last hour average
        monitor.update_daily(avg_rum > 0.5)


# ========================================
# BARAMATI CATTLE RUMINATION MONITORING
# ========================================
st.set_page_config(page_title="🐄 Cattle Health", layout="wide")
st.title("🐄 Cattle Rumination Dashboard")
st.markdown("---")

# ========================================
# PERSISTENT STATE - Survives Streamlit reruns
# ========================================
if 'daily_rumination_total' not in st.session_state:
    st.session_state.daily_rumination_total = 0
if 'rumination_counter' not in st.session_state:
    st.session_state.rumination_counter = 0
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = st.time()

# Current values
daily_rumination_min = st.session_state.daily_rumination_total
sample_count = st.session_state.rumination_counter

# ========================================
# CONTROL PANEL
# ========================================
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🚀 Start Monitoring", type="primary"):
        st.session_state.is_monitoring = True
        st.session_state.start_time = st.time()
        st.rerun()
    
with col2:
    if st.button("🛑 Stop Monitoring"):
        st.session_state.is_monitoring = False
        st.rerun()

with col3:
    st.info(f"📊 Monitoring: {'🟢 LIVE' if st.session_state.is_monitoring else '🔴 STOPPED'} | "
            f"⏱️ Uptime: {(st.time() - st.session_state.start_time).total_seconds()//60:.0f} min")

# ========================================
# KEY METRICS - SINGLE CLEAN DISPLAY
# ========================================
col1, col2 = st.columns(2)
with col1:
    st.metric("🐄 Daily Rumination (min)", 
              f"{daily_rumination_min}",
              "400-550",
              delta=None)
    
with col2:
    st.metric("📈 Samples Processed", 
              f"{sample_count:,}",
              "1440 target")

# ========================================
# HEALTH STATUS - Clear Alerts
# ========================================
st.markdown("### 🏥 Health Status")
if daily_rumination_min < 400:
    st.error("🚨 **VET ALERT**: Low rumination detected! Check cow immediately!")
elif daily_rumination_min >= 400 and daily_rumination_min <= 550:
    st.success("✅ **HEALTHY**: Normal rumination levels")
else:
    st.warning("⚠️ **MONITOR**: High rumination detected")

# ========================================
# PROGRESS BAR - Visual Feedback
# ========================================
progress_col1, progress_col2 = st.columns([3,1])
with progress_col1:
    st.progress(min(daily_rumination_min / 550, 1.0))
with progress_col2:
    st.caption("Normal range")

# ========================================
# LIVE SIMULATION STATUS
# ========================================
if st.session_state.is_monitoring:
    # Simulate 1 minute of data every 0.5s (1440 min/day simulation)
    x = np.random.randn(60) * 0.5 + 0.5 * np.sin(np.arange(60) * 0.1)
    y = np.random.randn(60) * 0.3
    z = np.ones(60) * 0.9 + np.random.randn(60) * 0.1
    
    is_rum, confidence = monitor.predict_window(x, y, z)
    
    if is_rum > 0.5:  # Rumination detected
        st.session_state.daily_rumination_total += 1
    st.session_state.rumination_counter += 1
    
    # Auto-rerun every 0.5 seconds during monitoring
    time.sleep(0.5)
    st.rerun()
else:
    st.info("👆 Click **Start Monitoring** to begin real-time rumination tracking")

st.markdown("---")
st.caption("🌍 Live worldwide access | 📱 Mobile-ready for farm team | 🐄 Cattle Health System")

# Chart
fig, ax = plt.subplots()
ax.bar(['Rumination', 'Healthy Range'], [monitor.daily_rumination, 475])
st.pyplot(fig)
