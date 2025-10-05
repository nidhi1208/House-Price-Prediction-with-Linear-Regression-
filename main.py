import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

data = pd.read_csv("D:\house_price_prediction\home_dataset - home_dataset.csv")

house_size = data['HouseSize'].values
house_price = data['HousePrice'].values
plt.scatter(house_size, house_price, marker ='o', color ='blue')
plt.title('House Prices vs. House Size')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(house_size, house_price, test_size=0.2, random_state=42)

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

plt.scatter(x_test, y_test, marker='o', color ='blue', label='Actual Prices')
plt.plot(x_test, predictions, color = 'red', linewidth=2, label='Predicted Prices')
plt.title('Dumbo Property Price Prediction with Linear Regression ')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()
