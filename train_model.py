import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

# Path to your training data
DATA_PATH = 'data/training_data.csv'  # <-- Make sure this file exists and is correct!

# List of features per task (17, matching your Flask app)
feature_names = [
    'air_time', 'paper_time', 'total_time', 'dispersion_index', 'mean_speed', 'gmrt',
    'max_x_extension', 'max_y_extension', 'num_of_pendown', 'mean_speed_in_air',
    'mean_speed_on_paper', 'mean_acc_in_air', 'mean_acc_on_paper',
    'mean_jerk_in_air', 'mean_jerk_on_paper', 'gmrt_in_air', 'gmrt_on_paper'
]

# Build the full feature list for all 19 tasks
all_features = []
for i in range(1, 20):
    all_features.extend([f"{name}{i}" for name in feature_names])

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found!")
        return

    df = pd.read_csv(DATA_PATH)
    if 'class' not in df.columns:
        print("ERROR: 'class' column not found in data!")
        return

    # Convert class labels to binary (P -> 1, others -> 0)
    df['class'] = (df['class'] == 'P').astype(int)

    # Check for missing features
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print("ERROR: Missing features in data:", missing)
        return

    X = df[all_features]
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to 'models/'.")

    # Print test accuracy
    acc = model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {acc:.3f}")

if __name__ == "__main__":
    main() 