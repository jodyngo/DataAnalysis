import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from aakr.aakr import AAKR
import time
import os


base_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024'
train_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\ProcessedNormalData'
test_dir = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\AbnormalData'

tags_file = os.path.join(base_dir, 'csv_tag_list_processed.txt')
with open(tags_file, 'r') as file:
        tag_names = [line.strip() for line in file]

for TAG in tag_names:
    print(f"Processing TAG: {TAG}")
    filename = TAG + '.csv'

    try:
        df_X = pd.read_csv(os.path.join(train_dir, filename), header=None, names=['timestamp', 'value'], encoding='utf-8', low_memory=False, on_bad_lines='warn')
        df_Y = pd.read_csv(os.path.join(test_dir, filename), header=None, names=['timestamp', 'value'], encoding='utf-8', low_memory=False, on_bad_lines='warn')        

        df_X = df_X[df_X['timestamp'].str.contains(':', na=False)]

        # Convert timestamp to datetime immediately after reading the files
        datetime_format = '%Y-%m-%d %p %I:%M:%S'
        df_X['timestamp'] = pd.to_datetime(df_X['timestamp'], format=datetime_format)
        df_Y['timestamp'] = pd.to_datetime(df_Y['timestamp'], format=datetime_format)

        # print("Timestamp range in df_train:", df_X['timestamp'].min(), "to", df_X['timestamp'].max())
        # print("Timestamp range in df_test:", df_Y['timestamp'].min(), "to", df_Y['timestamp'].max())

        # Combine the dataframes
        df_combined = pd.concat([df_X, df_Y])
        df_combined.set_index('timestamp', inplace=True)

        df_combined = df_combined.resample('5S').mean()  # Resampling to ____S interval

        df_combined.reset_index(inplace=True)
        df_combined = df_combined.dropna(subset=['value'])
        df_combined.to_csv(os.path.join(train_dir, f"{TAG}_5S_resampled.csv"), index=False, header=False)
        print("Saving into files....")

        if df_combined['value'].std() == 0:
            print(f"Skipping TAG {TAG} due to zero standard deviation in 'value' column.")
            continue

        X_ = df_combined[['value']]        

        print(f"Total length of combined data: {len(X_)}")

        min_length = min(100000, len(X_))
        print("min_length:", min_length)

        X = X_.iloc[:min_length].values
        y = X_.iloc[:min_length].values

        print("Start training....")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # AAKR starting....
        start_time = time.time()
        model = AAKR(modified=True)
        model.fit(X_train)
        # Transform the test data to get the predicted values
        y_pred = model.transform(X_test)
        end_time = time.time()
        duration = end_time - start_time

        print(f"The algorithm took {duration:.2f} seconds to complete.")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("AAKR:","tag:", TAG, "mse:", mse, "r2: ", r2)

        save_path = os.path.join(base_dir, f"Fig_AAKR_{TAG}.png")

        plt.figure(figsize=(15, 6))
        plt.plot(y_test, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'AAKR Model Evaluation (MSE: {mse:.4f}, R^2: {r2:.4f})')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot for TAG {TAG}.")

    except Exception as e:
        print(f"An error occurred while processing TAG {TAG}: {e}")