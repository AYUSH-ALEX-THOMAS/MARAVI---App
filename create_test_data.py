import pandas as pd
import numpy as np
import os

def create_test_data():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Number of samples
    n_samples = 50
    
    # Create features similar to our model's input
    data = {
        'air_time': np.random.uniform(100, 1000, n_samples),
        'paper_time': np.random.uniform(1000, 5000, n_samples),
        'total_time': np.random.uniform(1100, 6000, n_samples),
        'dispersion_index': np.random.uniform(0.1, 1.0, n_samples),
        'mean_speed': np.random.uniform(5000, 20000, n_samples),
        'gmrt': np.random.uniform(20, 100, n_samples),
        'max_x_extension': np.random.uniform(1000, 10000, n_samples),
        'max_y_extension': np.random.uniform(1000, 10000, n_samples),
        'num_of_pendown': np.random.randint(1, 10, n_samples),
        'mean_speed_in_air': np.random.uniform(1000, 5000, n_samples),
        'mean_speed_on_paper': np.random.uniform(5000, 15000, n_samples),
        'mean_acc_in_air': np.random.uniform(-5, 5, n_samples),
        'mean_acc_on_paper': np.random.uniform(-5, 5, n_samples),
        'mean_jerk_in_air': np.random.uniform(-1, 1, n_samples),
        'mean_jerk_on_paper': np.random.uniform(-1, 1, n_samples),
        'gmrt_in_air': np.random.uniform(10, 50, n_samples),
        'gmrt_on_paper': np.random.uniform(20, 80, n_samples),
        'class': np.random.choice(['H', 'P'], n_samples, p=[0.7, 0.3])  # 70% healthy, 30% at risk
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/test_data.csv', index=False)
    print("Test data created successfully!")

if __name__ == '__main__':
    create_test_data() 