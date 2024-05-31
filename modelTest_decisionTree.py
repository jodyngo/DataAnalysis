import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

TAG = 'DCD1AIG01ACTI02'

# Read CSV file (header=None specifies no column name)
df = pd.read_csv(f'./hourData/{TAG}_HAN.csv', header=None)

# Convert the date to a number (use a simple index here)
df['date_num'] = np.arange(len(df))

# Data preparation
X = df[['date_num']]
y = df[1]

print(y)

# 데이터 분할
# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Model Training
model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)

# Predictions and Performance Measurements
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Draw a graph
plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.3)
plt.scatter(X_test, y_test, color='green', label='Test', alpha=0.3)

# Prediction Results Graph
x_range = np.linspace(X['date_num'].min(), X['date_num'].max(), 400).reshape(-1, 1)
y_range_pred = model.predict(x_range)
plt.plot(x_range, y_range_pred, color='red', label='Prediction')

plt.legend()
plt.xlabel('Date as Number')
plt.ylabel('Value')
plt.title(f'Decision Tree Regression (MSE: {mse:.4f}, R^2: {r2:.4f})')
plt.savefig('Fig_DecisionTree.png')
plt.show()
