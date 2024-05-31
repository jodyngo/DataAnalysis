import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

import os 
import time


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

    X = X_.iloc[:min_length].values
    y = X_.iloc[:min_length].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()

    # Define VAE architecture
    input_dim = X_train.shape[1]
    latent_dim = 2  # Dimensionality of the latent space

    # Encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(16, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # Decoder
    decoder_h = Dense(32, activation='relu')  # Increased complexity
    decoder_mean = Dense(input_dim)
    h_decoded = decoder_h(z)
    x_decoded_mean_raw = decoder_mean(h_decoded)
    x_decoded_mean = Activation('linear')(x_decoded_mean_raw)

    # Adjust the range of output from 0-1 to 0-40
    x_decoded_mean = Lambda(lambda x: x * 40)(x_decoded_mean)

    # VAE Model
    vae = Model(inputs, x_decoded_mean)

    # Loss
    reconstruction_loss = mse(inputs, x_decoded_mean) * input_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1) * -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    # Train VAE
    vae.fit(X_train, epochs=100, batch_size=32, validation_data=(X_test, None))

    # Use VAE to reconstruct test data
    y_pred = vae.predict(X_test)

    end_time = time.time()
    duration = end_time - start_time

    print(f"The algorithm took {duration:.2f} seconds to complete.")


    # Measure performance
    mse_value = mean_squared_error(y_test, y_pred)
    r2_value = r2_score(y_test, y_pred)

    save_path = os.path.join(base_dir, f"Fig_VAE_{TAG}.png")

    print("VAE:", "tag:", TAG, "mse:", mse_value, "r2: ", r2_value)

    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'VAE Model (MSE: {mse_value:.4f}, R^2: {r2_value:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path)
    plt.close()