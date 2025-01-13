# # Importing the Dependenices
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set display options to show all columns
pd.set_option('display.max_columns', None)

# Data Collection & Processing
# Loading the dataset to Pandas Dataframe
loan_dataset = pd.read_csv(r"D:\Desktop\Data Science\Python\Loan Status Prediction\.venv\Loan Status Dataset.csv")
print(type(loan_dataset))
# Printing the first 5 rows of the dataframe
print(loan_dataset.head())
# Number of rows & columns
print(loan_dataset.shape)
print(loan_dataset.columns)

# Statistical Measures
print(loan_dataset.describe())
# Number of missing values in each column
print(loan_dataset.isnull().sum())
# Dropping the missing values
loan_dataset = loan_dataset.dropna()
print(loan_dataset.isnull().sum())
# Label Encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
# Dependent column values
print(loan_dataset['Dependents'].value_counts())
# Replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
# Dependent Values
print(loan_dataset['Dependents'].value_counts())

# Data Visualization

# Education & Loan Status
sns.countplot(x='Education',hue='Loan_Status', data=loan_dataset)
plt.show()

# Marital Status & Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.show()

# Gender & Loan Status
sns.countplot(x='Gender', hue='Loan_Status', data=loan_dataset)
plt.show()

# Dependents & Loan Status
# Group data by 'Dependents' and 'Loan_Status' and calculate the counts
plot_data = (loan_dataset.groupby(['Dependents', 'Loan_Status']).size().reset_index(name='Count'))
# Create the lineplot
sns.lineplot(data=plot_data, x='Dependents', y='Count', hue='Loan_Status', marker='o')
# Add labels and title
plt.xlabel('Dependents')
plt.ylabel('Count')
plt.title('Loan Status by No. of Dependents')
# Display the plot
plt.show()

# Self-Employed & Loan Status
sns.countplot(x='Self_Employed', hue='Loan_Status', data=loan_dataset)
plt.show()

# Applicant Income & Loan Status
sns.histplot(x='ApplicantIncome', hue='Loan_Status', data=loan_dataset, kde=False, bins=30, multiple='stack')
plt.xlabel('Applicant Income')
plt.ylabel('Count')
plt.title('Distribution of Applicant Income by Loan Status')
plt.show()

# Loan Amount & Loan Status
sns.histplot(x='LoanAmount', hue='Loan_Status', data=loan_dataset, kde=False, bins=20,multiple='stack')
plt.title("Distribution of Loan Amount by Loan Status")
plt.xlabel("Loan Amount")
plt.ylabel("Count")
plt.show()

# Credit History & Loan Status
sns.countplot(x='Credit_History', hue='Loan_Status', data=loan_dataset)
plt.show()

# Property Area & Loan Status
sns.countplot(x='Property_Area', hue='Loan_Status', data=loan_dataset)
plt.show()

# Convert Categorical Columns to Numerical Values
loan_dataset.replace({'Married':{'No':0,'Yes':1}, 'Gender':{'Male':1,'Female':0}, 'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}, 'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# Separating the Labels & Data
x = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = loan_dataset['Loan_Status']
print(x)
print(y)

# Splitting the data into Training Data & Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# Training the Model: Support Vector Machine Model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Model Evaluation
# Accuracy Score on Training Data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy Score for Training Data: ", training_data_accuracy)
# Accuracy Score on Test Data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy Score for Test Data: ", test_data_accuracy)

# Making a Predictive System
input_data = ['LP001005','Male','Yes',0,'Graduate','Yes',3000,0,66,360,1,'Urban']
del input_data[0]
columns = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
input_data_df = pd.DataFrame([input_data], columns=columns)
input_data_df.replace({'Gender':{'Male':1,'Female':0}, 'Married':{'No':0,'Yes':1}, 'Education':{'Graduate':1,'Not Graduate':0},
                    'Self_Employed':{'No':0,'Yes':1}, 'Property_Area':{'Rural':0,'Semiurban':1,'Urban':1}},inplace=True)
prediction = classifier.predict(input_data_df)
print(prediction)
if prediction[0] == 0:
    print("Loan Status: Not Approved")
else:
    print("Loan Status: Approved")






