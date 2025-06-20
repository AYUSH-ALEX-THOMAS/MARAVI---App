import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Load data
print("Loading data...")
data_file = "d:/AlzheimInk-App-main/AlzheimInk-App-main/dataframe_clean.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Could not find {data_file}. Please make sure the file exists in the specified path.")

data = pd.read_csv(data_file)

# Prepare features and target
X = data.drop(['ID', 'class'], axis=1)
y = data['class']

print(f"\nDataset shape: {data.shape}")
print(f"Features: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("\nSaved scaler to scaler.pkl")

# Train models
print("\nTraining models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    results[name] = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cv_scores': cross_val_score(model, X_train_scaled, y_train, cv=5)
    }
    
    print(f"\n{name} Results:")
    print("Classification Report:")
    print(results[name]['classification_report'])
    print("\nCross-validation scores:", results[name]['cv_scores'])
    print("Mean CV Score:", results[name]['cv_scores'].mean())
    
    # Plot feature importance for Random Forest
    if name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

# Save models
print("\nSaving models...")
for name, model in models.items():
    model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved {name} model as {model_filename}")

print("\nTraining and analysis complete!") 