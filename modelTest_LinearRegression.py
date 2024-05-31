import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

TAG = 'DCD1AIG01ACTI02'
# Read CSV file (header=None specifies no column name)
df = pd.read_csv(f'./hourData/{TAG}_HAN.csv', header=None)

# Convert the date to a number (use a simple index here)
df['date_num'] = np.arange(len(df))

# Data preparation
X = df[['date_num']]
y = df[1]  #  selects the second column of the DataFrame df and assigns it to y.

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 성능 측정
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 그래프 그리기
plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.3)
plt.scatter(X_test, y_test, color='green', label='Test', alpha=0.3)

# 예측 결과 그래프
x_range = np.linspace(X['date_num'].min(), X['date_num'].max(), 400).reshape(-1, 1)
y_range_pred = model.predict(x_range)
plt.plot(x_range, y_range_pred, color='red', label='Prediction')

plt.legend()
plt.xlabel('Date as Number')
plt.ylabel('Value')
plt.title(f'Linear Regression (MSE: {mse:.4f}, R^2: {r2:.4f})')
plt.savefig('Fig_Linear.png')
plt.show()
