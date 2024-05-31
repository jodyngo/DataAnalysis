import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

TAG = 'DCD1AITL2VOLT01'

# Read CSV file (header=None specifies no column name)
df = pd.read_csv(f'./hourData/{TAG}_HAN.csv', header=None)

# data normalization
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[[1]])

# Data preparation
X, y = data_scaled[:-1], data_scaled[1:]  # same data vlues


# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DNN Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=200, batch_size=16)

# 예측
y_pred = model.predict(X_test)

# 역정규화
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# 성능 측정
mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

# 그래프 그리기
plt.figure(figsize=(15, 6))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.title(f'DNN Model (MSE: {mse:.4f}, R^2: {r2:.4f})')
plt.xlabel('Date Index')
plt.ylabel('Value')
plt.legend()
plt.savefig('Fig_DNN.png')
plt.show()

