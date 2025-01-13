# # Importing the Dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# # Loading the Diabetes Dataset to Pandas Dataframe
diabetes_dataset = pd.read_csv(r"D:\Desktop\Data Science\Python\Diabetes Prediction\.venv\Diabetes Dataset.csv")

# # Printing first 5 rows of the dataset
# print(diabetes_dataset.head())
# print(diabetes_dataset.columns)

# # Number of rows & columns in this dataset
# print(diabetes_dataset.shape)

# # Getting the statistical measures of the data
# print(diabetes_dataset.describe())
# print(diabetes_dataset['Outcome'].value_counts())

# # Parameters Mean comparison between Diabetic and Non-Diabetic
# print(diabetes_dataset.groupby('Outcome').mean())

# # Separating the data & labels
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']
# print(x)
# print(y)

# # Data Standardization (When the parameters in the dataset have different range, standardization is done to create a better ML model)
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
# print(standardized_data)
x = standardized_data
# print(x)
# print(y)

# # Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)
# print(x.shape, x_train.shape, x_test.shape)

# # Training the Model
classifier = svm.SVC(kernel='linear')
# # Training the Support Vector MAchine Classifier
classifier.fit(x_train, y_train)

# # Model Evaluation
# # Accuracy Score
# # Accuracy Score on the Training Data
x_train_pred = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_pred, y_train)
# print("Accuracy Score for Training Data:", training_data_accuracy)
# # Accuracy Score on the Test Data
x_test_pred = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_pred, y_test)
# print("Accuracy Score for Test Data:", test_data_accuracy)

# # Making a Predictive System
input_data = (5,166,72,19,175,25.8,0.587,51)
# # Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# # Reshape this array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# # Standardize the input data
std_data = scaler.transform(input_data_reshaped)
# print(std_data)
# # Prediction
# prediction = classifier.predict(std_data)
# print(prediction)
# if prediction[0] == 0:
#     print("The Person is not Diabetic")
# else:
#     print("The person is Diabetic")












