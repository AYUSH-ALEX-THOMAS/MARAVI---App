import csv
import random

# Feature names per task
feature_names = [
    'air_time', 'paper_time', 'total_time', 'dispersion_index', 'mean_speed', 'gmrt',
    'max_x_extension', 'max_y_extension', 'num_of_pendown', 'mean_speed_in_air',
    'mean_speed_on_paper', 'mean_acc_in_air', 'mean_acc_on_paper',
    'mean_jerk_in_air', 'mean_jerk_on_paper', 'gmrt_in_air', 'gmrt_on_paper'
]

# Build the full header for all 19 tasks
header = []
for i in range(1, 20):
    header.extend([f"{name}{i}" for name in feature_names])
header.append('class')

# Generate 10 rows (5 'P', 5 'H')
rows = []
for label in (['P'] * 5 + ['H'] * 5):
    row = []
    for _ in range(19):
        row.extend([
            round(random.uniform(100, 5000), 2),   # air_time
            round(random.uniform(1000, 6000), 2),   # paper_time
            round(random.uniform(1000, 7000), 2),   # total_time
            round(random.uniform(0.1, 1.0), 4),     # dispersion_index
            round(random.uniform(1000, 20000), 2),  # mean_speed
            round(random.uniform(10, 100), 2),      # gmrt
            round(random.uniform(1000, 10000), 2),  # max_x_extension
            round(random.uniform(1000, 10000), 2),  # max_y_extension
            random.randint(1, 10),                  # num_of_pendown
            round(random.uniform(1000, 5000), 2),   # mean_speed_in_air
            round(random.uniform(5000, 20000), 2),  # mean_speed_on_paper
            round(random.uniform(-5, 5), 4),        # mean_acc_in_air
            round(random.uniform(-5, 5), 4),        # mean_acc_on_paper
            round(random.uniform(-1, 1), 4),        # mean_jerk_in_air
            round(random.uniform(-1, 1), 4),        # mean_jerk_on_paper
            round(random.uniform(10, 100), 2),      # gmrt_in_air
            round(random.uniform(10, 100), 2),      # gmrt_on_paper
        ])
    row.append(label)
    rows.append(row)

with open('data/training_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print('Synthetic training_data.csv generated with 10 rows.') 