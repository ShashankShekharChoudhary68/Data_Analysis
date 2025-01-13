# # Importing the Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Data Collection & Preprocessing
# Loading the data from csv file to pandas Dataframe
car_dataset = pd.read_csv(r"D:\Desktop\Data Science\Python\Car Price Prediction\.venv\car data.csv")
# Inspecting the first 5 rows of the Dataframe
print(car_dataset.head())
# Checking the number of rows & columns
print(car_dataset.shape)
# Getting some information about the dataset
print(car_dataset.info())
# Checking number of missing values
print(car_dataset.isnull().sum())
# Checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# Encoding the categorical data
# Encoding "Fuel_Type" Column
encoder = LabelEncoder()
car_dataset['Fuel_Type'] = encoder.fit_transform(car_dataset['Fuel_Type'])
print(car_dataset['Fuel_Type'])
car_dataset['Seller_Type'] = encoder.fit_transform(car_dataset['Seller_Type'])
print(car_dataset['Seller_Type'])
car_dataset['Transmission'] = encoder.fit_transform(car_dataset['Transmission'])
print(car_dataset['Transmission'])

# Splitting the data into Training Data & Test Data
x = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
y = car_dataset['Selling_Price']

# Splitting Training Data & Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Model Training
# Linear Regression
# Loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(x_train, y_train)

# Model Evaluation
# Prediction on Training Data
training_data_pred = lin_reg_model.predict(x_train)
# R squared error
error_score_training = metrics.r2_score(y_train, training_data_pred)
print("R Squared Error(for training Data) Linear:", error_score_training)

# Visualize the actual prices and predicted prices
plt.scatter(y_train, training_data_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# Prediction on Testing Data
test_data_pred = lin_reg_model.predict(x_test)
# R squared error
error_score_test = metrics.r2_score(y_test, test_data_pred)
print("R Squared Error(for test data) Linear:", error_score_test)

# Visualize the actual prices and predicted prices
plt.scatter(y_test, test_data_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# Lasso Regression
lass_reg_model = Lasso()
lass_reg_model.fit(x_train, y_train)

# Model Evaluation
# Prediction on Training Data
training_data_pred2 = lass_reg_model.predict(x_train)
# R squared error
error_score_training2 = metrics.r2_score(y_train, training_data_pred2)
print("R Squared Error(for training Data) Lasso:", error_score_training2)

# Visualize the actual prices and predicted prices
plt.scatter(y_train, training_data_pred2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# Prediction on Testing Data
test_data_pred2 = lass_reg_model.predict(x_test)
# R squared error
error_score_test2 = metrics.r2_score(y_test, test_data_pred2)
print("R Squared Error(for test data) Lasso:", error_score_test)

# Visualize the actual prices and predicted prices
plt.scatter(y_test, test_data_pred2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
