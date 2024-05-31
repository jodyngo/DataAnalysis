import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import dask.dataframe as dd
from aakr.aakr import AAKR
from matplotlib import colormaps


def load_and_preprocess_data(tag_names, data_dir, num_samples):
    data_frames = []
    for TAG in tag_names:
        # print(f"Processing TAG: {TAG}")
        filename = TAG + '.csv'
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping TAG: {TAG}")
            continue
        try:
            df = dd.read_csv(file_path, header=None, names=['timestamp', TAG], assume_missing=True)
            df['timestamp'] = dd.to_datetime(df['timestamp'], format='%Y-%m-%d %p %I:%M:%S', errors='coerce')
            df = df.set_index('timestamp').resample('10S').mean().dropna()
            df = df.head(num_samples)  
            data_frames.append(df)
        except Exception as e:
            print(f"An error occurred while processing TAG {TAG}: {e}")
            continue

    if not data_frames:
        raise ValueError("No valid data files found.")
    
    combined_df = dd.concat(data_frames, axis=1)

    return combined_df

def compute_impute_normalize(train_data, test_data):
    train_data = train_data.compute()
    test_data = test_data.compute()
   
    imputer = SimpleImputer(strategy='mean')
    train_data_imputed = imputer.fit_transform(train_data)
    test_data_imputed = imputer.transform(test_data)

    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data_imputed)
    test_data_normalized = scaler.transform(test_data_imputed)

    if (train_data_normalized < 0).any() or (test_data_normalized < 0).any():
        print("Warning: Normalized data contains negative values.")
  
    return train_data_normalized, test_data_normalized

def fit_and_transform_aakr(train_data, test_data, chunk_size=1e6):
    model = AAKR(modified=True)
    model.fit(train_data)  
    
    predictions = []
    for start in range(0, len(test_data), int(chunk_size)):
        end = start + int(chunk_size)
        y_pred_chunk = model.transform(test_data[start:end])
        predictions.append(y_pred_chunk)
    
    return np.vstack(predictions)

def detect_anomalies(train_data_reduced, test_data_reduced, threshold_type='dynamic', fixed_threshold=None):
    y_pred = fit_and_transform_aakr(train_data_reduced, test_data_reduced)    
    residuals = np.linalg.norm(test_data_reduced - y_pred, axis=1)
    residuals_df = pd.DataFrame(residuals, columns=['Residuals'])
    
    if threshold_type == 'dynamic':
        rolling_mean = residuals_df['Residuals'].rolling(window=10).mean()
        rolling_std = residuals_df['Residuals'].rolling(window=10).std()
        dynamic_threshold = rolling_mean + 2 * rolling_std
        dynamic_threshold.fillna(method='bfill', inplace=True)
        threshold = dynamic_threshold
        anomalies = residuals_df['Residuals'] > dynamic_threshold

    elif threshold_type == 'fixed':
        if fixed_threshold is None:
            raise ValueError("Fixed threshold must be provided when threshold_type is 'fixed'")
        threshold = fixed_threshold
        anomalies = residuals_df['Residuals'] > fixed_threshold
    else:
        raise ValueError("Invalid threshold_type. Must be 'dynamic' or 'fixed'")
    
    return y_pred, residuals, threshold, anomalies

def transform_anomalies(test_data_normalized, y_pred, pca, threshold, n_features, threshold_type='dynamic'):
    reconstructed_test_data = pca.inverse_transform(y_pred)
    original_residuals = np.abs(test_data_normalized - reconstructed_test_data)

    print("reconstructed_test_data", reconstructed_test_data.shape)
    print("original_residuals", original_residuals.shape)

    if threshold_type == 'dynamic':
        dynamic_threshold_broadcast = np.tile(threshold.values[:, np.newaxis], (1, n_features))
        anomalous_features = original_residuals > (dynamic_threshold_broadcast / np.sqrt(n_features))
    elif threshold_type == 'fixed':
        anomalous_features = original_residuals > (threshold / np.sqrt(n_features))
    else:
        raise ValueError("Invalid threshold_type. Must be 'dynamic' or 'fixed'")
    print("anomalous_features", anomalous_features.shape)
    return anomalous_features

def create_output_dataframe(test_data, residuals, anomalies, anomalous_features, tag_names):
    output_data = {
        'Timestamp': test_data.index,
        'Anomaly Score': residuals,
        'Anomaly': anomalies
    }

    for i, feature in enumerate(tag_names):
        if i < anomalous_features.shape[1]:
            output_data[f'{feature}'] = anomalous_features[:, i]
        else:
            print(f"Warning: Index {i} is out of bounds for anomalous_features with shape {anomalous_features.shape}")  

    output_df = pd.DataFrame(output_data)
    return output_df

def save_results(output_df, base_dir):
    output_df.to_csv(os.path.join(base_dir, 'anomaly_detection_results.csv'), index=False)

def plot_results(output_df, threshold, base_dir, threshold_type='dynamic'):
    save_path = os.path.join(base_dir, 'Fig_AAKR_anomalies.png')

    plt.figure(figsize=(15, 6))
    plt.plot(output_df.index, output_df['Anomaly Score'], label='Anomaly Score')

    if threshold_type == 'dynamic':
        plt.plot(output_df.index, threshold, color='orange', linestyle='--', label='Dynamic Threshold')
    elif threshold_type == 'fixed':
        plt.axhline(y=threshold, color='orange', linestyle='--', label='Fixed Threshold')
    else:
        raise ValueError("Invalid threshold_type. Must be 'dynamic' or 'fixed'")
    
    plt.scatter(output_df.loc[output_df['Anomaly']].index, output_df.loc[output_df['Anomaly'], 'Anomaly Score'], color='red', label='Anomalies')
    plt.title('Anomaly Detection Results')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.legend()

    try:
        plt.savefig(save_path)
        print("Figure saved successfully.")
    except Exception as e:
        print(f"Error saving figure: {e}")
    plt.close()


def plot_results_with_features(output_df, threshold, resampled_df, base_dir, threshold_type='dynamic'):
    save_path_with_annotations = os.path.join(base_dir, 'Fig_AAKR_anomalies_with_features.png')

    features = set(feature for sublist in resampled_df['Anomalous Features'] for feature in sublist)
    colors = colormaps['tab20']
    feature_color_map = {feature: colors(i / len(features)) for i, feature in enumerate(features)}

    plt.figure(figsize=(15, 6))
    plt.plot(output_df.index, output_df['Anomaly Score'], label='Anomaly Score')

    if threshold_type == 'dynamic':
        plt.plot(output_df.index, threshold, color='orange', linestyle='--', label='Dynamic Threshold')
    elif threshold_type == 'fixed':
        plt.axhline(y=threshold, color='orange', linestyle='--', label='Fixed Threshold')
    else:
        raise ValueError("Invalid threshold_type. Must be 'dynamic' or 'fixed'")

    for feature in feature_color_map:
        feature_anomalies = resampled_df[resampled_df['Anomalous Features'].apply(lambda x: feature in x)]
        plt.scatter(feature_anomalies['Timestamp'], feature_anomalies['Anomaly Score'], 
                    color=feature_color_map[feature], label=feature)
    
    plt.title('Anomaly Detection Results with Anomalous Features')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.legend()

    try:
        plt.savefig(save_path_with_annotations)
        print("Annotated figure saved successfully.")
    except Exception as e:
        print(f"Error saving annotated figure: {e}")
    plt.close()

def resample_and_annotate(output_df, tag_names):
    output_df.set_index('Timestamp', inplace=True)
    resampled_df = output_df.resample('30S').apply({
        'Anomaly Score': 'max',
        'Anomaly': 'max'
    }).reset_index()

    resampled_df['Anomalous Features'] = resampled_df['Timestamp'].apply(lambda x: [])

    for i, row in resampled_df.iterrows():
        start_time = row['Timestamp']
        end_time = start_time + pd.Timedelta(seconds=30)
        interval_df = output_df.loc[start_time:end_time]
        anomalous_features = set()
        for idx in interval_df.index:
            if interval_df.loc[idx, 'Anomaly']:
                anomaly_features = [feature for feature in tag_names if interval_df.loc[idx].get(feature, False)]
                anomalous_features.update(anomaly_features)
        resampled_df.at[i, 'Anomalous Features'] = list(anomalous_features)

    return resampled_df

def list_anomalous_timestamps(output_df, resampled_df):
    output_df = output_df.sort_index()
    anomalous_timestamps = []

    for i, row in resampled_df.iterrows():
        timestamp = row['Timestamp']
        anomalous = row['Anomaly']
        anomalous_features = row['Anomalous Features']
        if anomalous:  # Only consider timestamps where Anomaly is True
            anomalous_timestamps.append((timestamp, anomalous_features))

    for timestamp, features in anomalous_timestamps:
        print(f"Timestamp: {timestamp}, Anomalous TAG: {features}")

    return anomalous_timestamps

if __name__ == "__main__":
    base_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024'
    train_dir = os.path.join(base_dir, 'ProcessedNormalData')
    test_dir = os.path.join(base_dir, 'AbnormalData')
    tags_file = os.path.join(base_dir, 'csv_tag_list_processed.txt')

    with open(tags_file, 'r') as file:
        tag_names = [line.strip() for line in file if line.strip()]

    num_samples = 1000
    train_data = load_and_preprocess_data(tag_names, train_dir, num_samples)
    test_data = load_and_preprocess_data(tag_names, test_dir, num_samples)

    train_data_normalized, test_data_normalized = compute_impute_normalize(train_data, test_data)

    print("train_data_normalized", train_data_normalized.shape)
    print("test_data_normalized", test_data_normalized.shape)

    n_samples, n_features = train_data_normalized.shape  
    print("n_samples", n_samples)
    print("n_features", n_features)

    n_components = 3
    pca = PCA(n_components=n_components)
    
    train_data_reduced = pca.fit_transform(train_data_normalized)
    test_data_reduced = pca.transform(test_data_normalized)

    print("train_data_reduced", train_data_reduced.shape)
    print("test_data_reduced", test_data_reduced.shape)

    threshold_type = 'fixed' 
    fixed_threshold_value = 1.5  

    if threshold_type == 'dynamic':
        y_pred, residuals, threshold, anomalies = detect_anomalies(train_data_reduced, test_data_reduced, threshold_type='dynamic')
        anomalous_features = transform_anomalies(test_data_normalized, y_pred, pca, threshold, n_features, threshold_type='dynamic')
        
    elif threshold_type == 'fixed':
        y_pred, residuals, threshold, anomalies = detect_anomalies(train_data_reduced, test_data_reduced, threshold_type='fixed', fixed_threshold=fixed_threshold_value)
        anomalous_features = transform_anomalies(test_data_normalized, y_pred, pca, threshold, n_features, threshold_type='fixed')

    output_df = create_output_dataframe(test_data, residuals, anomalies, anomalous_features, tag_names)
    save_results(output_df, base_dir)

    resampled_df = resample_and_annotate(output_df, tag_names)
    plot_results(output_df, threshold, base_dir, threshold_type=threshold_type)
    plot_results_with_features(output_df, threshold, resampled_df, base_dir, threshold_type=threshold_type)

    anomalous_timestamps = list_anomalous_timestamps(output_df, resampled_df)

