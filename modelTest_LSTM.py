import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


TAG = 'DCD1AIG01ACTI02'

# CSV 파일 읽기 (header=None으로 컬럼 이름 없음을 지정)
df = pd.read_csv(f'./hourData/{TAG}_HAN.csv', header=None)
# 데이터 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[[1]])

# 시계열 데이터로 변환
X, y = [], []
for i in range(len(df) - 10):
    X.append(data_scaled[i:i+10])
    y.append(data_scaled[i+10])

X, y = np.array(X), np.array(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

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
plt.title(f'LSTM Model (MSE: {mse:.4f}, R^2: {r2:.4f})')
plt.xlabel('Date Index')
plt.ylabel('Value')
plt.legend()
plt.savefig('Fig_LSTM.png')
plt.show()
