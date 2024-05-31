import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
import os 

start_time = time.time()
# TAG = 'SYD1AIG01CURR03'
base_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024'
train_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\ProcessedNormalData'
test_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\AbnormalData'

# Read TAG names from the file
tags_file = os.path.join(base_dir, 'csv_tag_list_processed.txt')
with open(tags_file, 'r') as file:
        tag_names = [line.strip() for line in file]

for TAG in tag_names:
    print(f"Processing TAG: {TAG}")

    filename = TAG + '.csv'

    df_X = pd.read_csv(os.path.join(train_dir, filename), header=None, names=['timestamp', 'value'], encoding='utf-8', low_memory=False, on_bad_lines='warn')
    df_Y = pd.read_csv(os.path.join(test_dir, filename), header=None, names=['timestamp', 'value'], encoding='utf-8', low_memory=False, on_bad_lines='warn')
    df_X = df_X[df_X['timestamp'].str.contains(':', na=False)]

    # Convert timestamp to datetime immediately after reading the files
    datetime_format = '%Y-%m-%d %p %I:%M:%S'
    df_X['timestamp'] = pd.to_datetime(df_X['timestamp'], format=datetime_format)
    df_Y['timestamp'] = pd.to_datetime(df_Y['timestamp'], format=datetime_format)

    # Combine the dataframes
    df_combined = pd.concat([df_X, df_Y])
    df_combined.set_index('timestamp', inplace=True)

    df_combined = df_combined.resample('5S').mean()  # Resampling to ____S interval

    df_combined.reset_index(inplace=True)
    df_combined = df_combined.dropna(subset=['value'])
    # df_combined.to_csv(os.path.join(train_dir, f"{TAG}_resampled.csv"), index=False, header=False)
    print("Saving into files....")

    # Ensure column indexing is correct
    X_ = df_combined[['value']]

    print(f"Total length of combined data: {len(X_)}")

    min_length = min(50000, len(X_))
    print("min_length:", min_length)

    signal = X_.iloc[:min_length].values

    signal = signal.flatten()

    # Detect peaks
    peaks, _ = find_peaks(signal, height=0.2, distance=20)  # Adjusted parameters

    # Anomalies detected
    anomalies_detected = np.zeros_like(signal)
    anomalies_detected[peaks] = 1

    # Ground truth for evaluation
    np.random.seed(0)
    ground_truth_anomalies = np.random.choice([0, 1], size=len(signal), p=[0.85, 0.15])


    # Calculate MSE and R2
    mse = mean_squared_error(ground_truth_anomalies, anomalies_detected)
    r2 = r2_score(ground_truth_anomalies, anomalies_detected)

    # Calculate classification metrics
    precision = precision_score(ground_truth_anomalies, anomalies_detected)
    recall = recall_score(ground_truth_anomalies, anomalies_detected)
    f1 = f1_score(ground_truth_anomalies, anomalies_detected)
    accuracy = accuracy_score(ground_truth_anomalies, anomalies_detected)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    save_path = os.path.join(base_dir, f"Fig_PDMS_{TAG}.png")

    print("PDMS:", "tag:", TAG, "mse:", mse, "r2: ", r2)

    plt.figure(figsize=(15, 6))
    plt.plot(signal, label='Signal')
    plt.plot(peaks, signal[peaks], "x", label='Prediction', color='red')
    plt.title(f'PDMS (MSE: {mse:.4f}, Accuracy: {accuracy:.4f})')
    plt.xlabel('Date as Number')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    end_time = time.time()
    duration = end_time - start_time

    print(f"The algorithm took {duration:.2f} seconds to complete.")

