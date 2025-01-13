# # Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# # Data Collection & Data Preprocessing
# # Loading the dataset to pandas Dataframe
sonar_data = pd.read_csv(r"D:\Desktop\Data Science\Python\Sonar Rock vs Mine Prediction\.venv\SONAR Dataset.csv", header=None)
# print(sonar_data.head())
# # Number of rows & columns
# print(sonar_data.shape)
# # Statistical Measures
# print(sonar_data.describe())
# print(sonar_data[60].value_counts())
# print(sonar_data.groupby(60).mean())

# # Plot between Rock vs Mine Parameters
# r = sonar_data[sonar_data[60] == 'R'].iloc[:, :-1].mean()
# m = sonar_data[sonar_data[60] == 'M'].iloc[:, :-1].mean()
# plt.figure(figsize=(12,8))
# plt.plot(r, marker='o', linestyle='-', label='Rock Parameters Mean', markersize=6, color='green', linewidth=2)
# plt.plot(m, marker='o', linestyle='-', label='Mine Parameters Mean', markersize=6, color='red', linewidth=2)
# plt.title('Rock vs Mine Parameters', fontsize=14)
# plt.xlabel('Means of Parameters')
# plt.ylabel('Value of Means')
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #Separating Data & Labels
# x = sonar_data.drop(columns=60, axis=0)
# y = sonar_data[60]

# # Training & Test Data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
# print(x.shape, x_train.shape, x_test.shape)

# # Model Training (Logistic RegressionModel)
# model = LogisticRegression()
# # Training the Logistic Regression Model with Training Data
# model.fit(x_train, y_train)

# # Model Evaluation
# # Accuracy on training data
# x_train_prediction = model.predict(x_train)
# training_data_accuracy = accuracy_score(x_train_prediction, y_train)
# print("Accuracy on training data:", training_data_accuracy)
# # Accuracy on test data
# x_test_prediction = model.predict(x_test)
# test_data_accuracy = accuracy_score(x_test_prediction, y_test)
# print("Accuracy on test data:", test_data_accuracy)

# # Making a Predictive System
# input_data = [0.0712,0.0901,0.1276,0.1497,0.1284,0.1165,0.1285,0.1684,0.1830,0.2127,0.2891,0.3985,0.4576,0.5821,0.5027,0.1930,0.2579,0.3177,0.2745,0.6186,0.8958,0.7442,0.5188,0.2811,0.1773,0.6607,0.7576,0.5122,0.4701,0.5479,0.4347,0.1276,0.0846,0.0927,0.0313,0.0998,0.1781,0.1586,0.3001,0.2208,0.1455,0.2895,0.3203,0.1414,0.0629,0.0734,0.0805,0.0608,0.0565,0.0286,0.0154,0.0154,0.0156,0.0054,0.0030,0.0048,0.0087,0.0101,0.0095,0.0068]
# # Changing the input data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)
# # Reshape the np array as we are predicting for one instance
# input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
# prediction = model.predict(input_data_reshape)
# print(prediction)
# if (prediction[0]=='R'):
#     print("The object is a Rock.")
# else:
#     print("The object is a Mine.")
