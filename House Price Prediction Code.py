# # Importing the Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import  XGBRegressor
from sklearn import metrics

# # Setting the Display
pd.set_option('display.max_columns', None)
# # Importing the Boston House Price Dataset
house_price_dataset = pd.read_csv(r"D:\Desktop\Data Science\Python\House Price Prediction\.venv\boston_housing_dataset.csv")
# print(house_price_dataset)
# # Loading the dataset to a Pandas Dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset)
# print(house_price_dataframe)
# # Print first 5 rows of the dataframe
# print(house_price_dataframe.head())
# # Checking the number of rows & columns in the dataframe
# print(house_price_dataframe.shape)

# # Data Preprocessing
# # Check missing values
# print(house_price_dataframe.isnull().sum())
# # Statistical Measures of the dataset
# print(house_price_dataframe.describe())
# print(house_price_dataframe.columns)

# # Understanding the correlation between various features of dataset
# correlation = house_price_dataframe.corr()
# print(correlation)
# # Constructing a heatmap to understand the correlation
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# plt.show()

# # Splitting the data & target
x = house_price_dataframe.drop(['MEDV'],axis=1)
y = house_price_dataframe['MEDV']
# print(x)
# print(y)

# # Splitting the data into Training Data & Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# print(x.shape, x_train.shape, x_test.shape)
# print(y.shape, y_train.shape, y_test.shape)

# # Model Training
# # XGBoost Regressor
# # Loading the Model
model = XGBRegressor()
# # Traing the model with x_train
model.fit(x_train, y_train)

# # Model Evaluation
# # Prediction on Training Data
# # Accuracy for Prediction on Training Data
training_data_pred = model.predict(x_train)
# print(training_data_pred)
# # R Suqared Error
# score_1 = metrics.r2_score(y_train, training_data_pred)
# # Mean Absolute Error
# score_2 = metrics.mean_absolute_error(y_train, training_data_pred)
# print("R Squared Error :", score_1)
# print("Mean Absolute Error :", score_2)

# # Visualize the actual prices & predicted prices for training data
# plt.scatter(y_train, training_data_pred)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Price vs Predicted Price')
# plt.show()

# # Accuracy for Prediction on Training Data
test_data_pred = model.predict(x_test)
# print(test_data_pred)
# # R Suqared Error
# score_1 = metrics.r2_score(y_test, test_data_pred)
# # Mean Absolute Error
# score_2 = metrics.mean_absolute_error(y_test, test_data_pred)
# print("R Squared Error :", score_1)
# print("Mean Absolute Error :", score_2)

# # Visualize the actual prices & predicted prices for test data
# plt.scatter(y_test, test_data_pred)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Price vs Predicted Price')
# plt.show()

# # Making Predictive System
# input_data = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# input_data_as_numpy_array = np.asarray(input_data)
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
# input_data_df = pd.DataFrame(input_data_reshaped, columns=columns)
# prediction = model.predict(input_data_df)
# print(f"The Price of the house is ${prediction}")


