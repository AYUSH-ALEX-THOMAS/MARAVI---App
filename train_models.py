import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import os

def train_models(data_path='data.csv'):
    """
    Train the models using the provided dataset
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert class labels to binary (P -> 1, others -> 0)
    df['class'] = (df['class'] == 'P').astype(int)
    
    # Extract features for each task
    features = []
    for i in range(1, 20):  # 19 tasks
        task_features = [
            f'air_time{i}',
            f'disp_index{i}',
            f'mean_gmrt{i}',  # This is what we calculate as gmrt in the app
            f'max_x_extension{i}',
            f'max_y_extension{i}',
            f'mean_speed_on_paper{i}',  # This is what we calculate as mean_speed in the app
            f'num_of_pendown{i}',
            f'paper_time{i}',
            f'gmrt_in_air{i}',
            f'gmrt_on_paper{i}',
            f'mean_acc_in_air{i}',
            f'mean_acc_on_paper{i}',
            f'mean_jerk_in_air{i}',
            f'mean_jerk_on_paper{i}',
            f'mean_speed_in_air{i}',
            f'pressure_mean{i}',
            f'pressure_var{i}',
            f'total_time{i}'
        ]
        features.extend(task_features)
    
    # Separate features and target
    X = df[features]
    y = df['class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'xgboost': XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"{name} - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        # Save model
        joblib.dump(model, f'models/{name}_model.pkl')
        
        # Track best model
        if test_score > best_score:
            best_score = test_score
            best_model = model
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save best model as default model
    joblib.dump(best_model, 'random_forest_model.pkl')
    
    print(f"\nBest model accuracy: {best_score:.4f}")
    print("All models and scaler saved to 'models' directory")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('data.csv'):
        print("Error: data.csv not found!")
        print("Please ensure you have a dataset with the following columns:")
        print("- Features for each task (air_time, paper_time, etc.)")
        print("- A 'class' column (P for Parkinson's/cognitive impairment, others for healthy)")
    else:
        train_models() 