import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

'''
#######################################
Data Analysis Model Post-processing of the Power Generation Integrated Operating System
#######################################

### Use variable ###

▶ EPOCH = Learning Capacity
▶ tag1 = Analytical tag name 1
▶ tag2 = Analytical tag name 2

### How to use ###

1. Function loadCsvData (tag name)
    ▶ Finds the CSV that corresponds to the tag name and returns it to the data frame.
    
2. Global Variables

    EPOCH = Adjust the amount of learning, usually increasing the amount of variation in the loss function until it is insignificant.
    tag1 = tag corresponding to the X-axis (independent variable).
    tag2 = tag corresponding to the Y-axis (dependent variable).
    NONCHANGED = Default is False. Checks for no variation data when changed to True.

'''

# Specify global variables
EPOCH = 10
tag1 = 'DCD1AIG01ACTI01' # X-axis corresponding tag (active power)
tag2 = 'DCD1AIG01CURR01' # Y-axis corresponding tack (current)
NONCHANGED = True



start_time = time.time()


def loadCsvData(tagName):
    df = pd.read_csv(f"./hourData/{tagName}_HAN.csv")
    return df.iloc[:,1]

# Create Data
np.random.seed(42)
X = loadCsvData(tag1)
y = loadCsvData(tag2)  

print(X.shape)
X = np.array(X)
y = np.array(y)

# Data Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Model Prepare
model = Sequential([
    Dense(10, activation='relu',input_dim=1),
    Dense(10, activation='relu'),
    Dense(1)
])


# Compile
model.compile(loss='mean_squared_error', optimizer='Adam')

# Learning
model.fit(X, y, epochs=EPOCH, batch_size=32, verbose=1)


# Generating a prediction value
y_pred = model.predict(X).flatten()  

# Compute residuals
residuals = y - y_pred

# Calculate the standard deviation of the residuals
stdResiduals = np.std(residuals)

# Calculate 95% confidence intervals
confidence_interval = stdResiduals * stats.t.ppf((1 + 0.95) / 2, len(y) - 1)

# Experimental Data
X2 = X
if NONCHANGED == True:
    y2 = pd.read_csv(f'./hourData/{tag2}_HAN_NonChanging.csv').iloc[:,1]
else:
    y2 = pd.read_csv(f'./hourData/{tag2}_HAN_Error.csv').iloc[:, 1]
X2 = np.array(X2)
y2 = np.array(y2)

# Setting the criteria for error determination (e.g. 3 * standard deviation)
threshold = 3 * np.std(residuals)

# Error detection
residuals2 = y2 - y_pred
outliers = np.where(np.abs(residuals2) > threshold)[0]


# Sort x and y by x
sorted_pairs = sorted(zip(X2, y2))
x_sorted, y_sorted = zip(*sorted_pairs)

# Find successive x intervals with the same y value
same_y_intervals = []
start_x = x_sorted[0]
current_y = y_sorted[0]

for i in range(1, len(y_sorted)):
    if y_sorted[i] != current_y:
        if start_x != x_sorted[i-1]:
            same_y_intervals.append((start_x, x_sorted[i-1], current_y))
        start_x = x_sorted[i]
        current_y = y_sorted[i]



# Check the last section
if start_x != x_sorted[-1]:
    same_y_intervals.append((start_x, x_sorted[-1], current_y))

# Output the results
print("Intervals where y-values are constant while x is increasing:")
for interval in same_y_intervals:
    print(f"From x = {interval[0]} to x = {interval[1]} (y = {interval[2]})")

model.fit(X2, y2)

# Draw a scatterplot
plt.figure(figsize=(12, 6))

# Obtain the end length of the graph
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# Normal Chart Display
plt.subplot(2, 1, 1)
plt.scatter(X, y, c='orange', s=65, label="Original Data", alpha=0.3)
plt.plot(X, model.predict(X), color='blue', linewidth=1, label="Regression Line")
plt.xlabel(f"{tag1}")
plt.ylabel(f"{tag2}")

# Draw confidence intervals
plt.fill_between(X.flatten(), (y_pred - confidence_interval), (y_pred + confidence_interval), color='gray', alpha=0.5)
plt.title("Original Data")
plt.legend()
# MSE Mean Squared Error Display 1
mse1 = mean_squared_error(y, y_pred)
plt.text(xlim[1]*0.9, ylim[1]*0.9, f'MSE = {mse1}', fontsize=9) 


# Error chart display
plt.subplot(2, 1, 2)
plt.scatter(X2, y2, c='orange', s=65, label="Error Data", alpha=0.3)
plt.plot(X, model.predict(X), color='blue', linewidth=1, label="Regression Line")
plt.scatter(X2[outliers], y2[outliers], color='red', label="Outliers", marker='x', zorder=5)
plt.xlabel(f"{tag1}")
plt.ylabel(f"{tag2}")

# Draw confidence intervals
plt.fill_between(X.flatten(), (y_pred - confidence_interval), (y_pred + confidence_interval), color='gray', alpha=0.5)
x_position = 63
y_position = 616
plt.annotate('95% Confidence Interval', xy=(x_position, y_position), xytext=(x_position+20, y_position + 20),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12)
plt.title("Error Data")
plt.legend()

# # Show intervals with the same y value
for interval in same_y_intervals:
    plt.hlines(interval[2], interval[0], interval[1], colors='r', linestyles='dashed')
mean_value = np.mean(y_pred)
#print(f"표준편차{np.std(y_pred)} 3표준편차{mean_value - (3*np.std(y_pred))} 상한값{mean_value + (3*np.std(y_pred))}")

# Display MSE mean square error 2 (mean square difference between y2 and y_pred)
mse2 = mean_squared_error(y2, y_pred)
plt.text(xlim[1]*0.9, ylim[1]*0.9, f'MSE = {mse2:.4f}', fontsize=9) # MSE 값을 표시
plt.tight_layout()
end_time = time.time()
print(f"총 훈련시간(Training time): {end_time - start_time} seconds")
plt.show()
