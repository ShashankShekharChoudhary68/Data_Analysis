import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM

data = pd.read_csv(r"D:\Desktop\Data Science\Python\Fraud Detection Credit Card 2023\.venv\creditcard.csv")
df = pd.DataFrame(data)
# print(df)
# print(df.head())

# # Check Null Values
# print(df.isnull().sum())

# # Check Duplicates
# print(df.duplicated().sum())

# # Data Description
# print(df.info())
# print(df.describe())
# print(df.columns)
# print(df.shape)
# print(df.dtypes)
# print(df['Class'].value_counts())

# # Group the data by Class
# class_count = df['Class'].value_counts()
# print(class_count)

# # Distribution of Valid & Fraudulent Transactions
# class_count = df['Class'].value_counts()
# plt.figure(figsize=(15,9))
# plt.bar(class_count.index, class_count.values, color=['blue','orange'])
# plt.title('Valid vs Fraud Transactions', fontsize=14)
# plt.yscale('log')  # We use log scale for better visualization since small values are not visible due to large scale
# plt.xlabel('Class', fontsize=12)
# plt.ylabel('No. of Transactions', fontsize=12)
# plt.xticks(ticks=class_count.index, labels=['Valid', 'Fraud'], fontsize=10)
# plt.tight_layout()
# plt.show()

# # Get the Fraud & Valid Dataset
fraud = df[df['Class']==1]
valid = df[df['Class']==0]
# print(fraud.value_counts(), valid.value_counts())
# print(fraud.shape, valid.shape)

# # We need to analyze more amount of information like the amount of money used in different transaction classes
# print(fraud.Amount.describe())
# print(valid.Amount.describe())

# # Histogram of Amount of Fraud Transcations
# plt.figure(figsize=(12,8))
# plt.hist(fraud['Amount'], bins=50, color='green')
# plt.title('Fraud Transactions Amount Plot', fontsize=14)
# plt.yscale('log')
# plt.xlabel('Amount Range', fontsize=10)
# plt.ylabel('Transaction Frequency', fontsize=10)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(12,8))
# plt.hist(valid['Amount'], bins=50, color='yellow')
# plt.title('Valid Transactions Amount Plot', fontsize=14)
# plt.yscale('log')
# plt.xlabel('Amount Range', fontsize=10)
# plt.ylabel('Transaction Frequency', fontsize=10)
# plt.tight_layout()
# plt.show()

# # By Subplots
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# f.suptitle('Amount per transaction by class')
# bins = 50
# ax1.hist(fraud.Amount, bins = bins)
# ax1.set_title('Fraud')
# ax2.hist(valid.Amount, bins = bins)
# ax2.set_title('Normal')
# plt.xlabel('Amount ($)')
# plt.ylabel('Number of Transactions')
# plt.xlim((0, 20000))
# plt.yscale('log')
# plt.show();

# # We will check Do fraudulent transactions occur more often during certain time frame? Let us find out with a visual representation.
# # Time of Transaction vs Amount by Class
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# f.suptitle('Time of Transaction vs Amount by Class')
# ax1.scatter(fraud.Time, fraud.Amount)
# ax1.set_title('Fraud')
# ax1.set_xlabel('Time (in Seconds)', fontsize=10)
# ax1.set_ylabel('Amount', fontsize=10)
# ax2.scatter(valid.Time, valid.Amount)
# ax2.set_title('Valid')
# ax2.set_xlabel('Time (in Seconds)', fontsize=10)
# ax2.set_ylabel('Amount', fontsize=10)
# plt.show()

# # Correlation Matrix
# # Get correlations of each feature
# corrmat = data.corr()
# # Select a subset of the top correlated features
# top_corr_features = corrmat.index[:15]  # Adjust the number based on your dataset
# # Plot heatmap for the selected features
# plt.figure(figsize=(25, 25))
# sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn", fmt=".2f", annot_kws={"size": 10})
# plt.tight_layout()
# plt.show()

# # Build a Sample Dataset containing similar distribution of valid transaction & fraudulent transaction
legit_sample = valid.sample(n=492, random_state=42)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
# print(new_dataset)
# # Check null values in new_dataset
# print(new_dataset.isnull().sum())
# # Checking the nature of new_dataset
# print(new_dataset.groupby('Class').mean())

# # Plot a line chart of means of each column of new_dataset
x = new_dataset[new_dataset['Class']==0].mean()
y = new_dataset[new_dataset['Class']==1].mean()
print(x,y)
plt.figure(figsize=(12,8))
plt.plot(x, marker='o', linestyle='-', label='Valid Trans', color='blue', linewidth=2, markersize=6)
plt.plot(y, marker='o', linestyle='-', label='Fraud Trans', color='red', linewidth=2, markersize=6)
plt.title('Means of Valid & Fraud Transcations in new_dataset', fontsize=14)
plt.yscale('symlog', linthresh=1)
plt.minorticks_on()
plt.xlabel('Means of Columns')
plt.ylabel('Value of Mean')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# # Splitting the data into input(features) and output(target labels)
input = df.drop(columns='Class', axis=1)
output = df['Class']
# # print(input, "\n", output)

# # Splitting the data into training & testing data
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=2, stratify=output)    # Stratify is used when the data is imbalanced
# print(input.shape, x_train.shape, x_test.shape)

# # Model Training
# # Define models
# # Train and evaluate Logistic Regression model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(x_train, y_train)

logistic_train_pred = logistic_model.predict(x_train)
logistic_test_pred = logistic_model.predict(x_test)

logistic_train_accuracy = accuracy_score(y_train, logistic_train_pred)
logistic_test_accuracy = accuracy_score(y_test, logistic_test_pred)

print(f"Logistic Regression - Train Accuracy: {logistic_train_accuracy:.4f}")
print(f"Logistic Regression - Test Accuracy: {logistic_test_accuracy:.4f}")

# # Define outlier detection models
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(x_train), contamination=0.01,
                                        random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01)
}

# # Evaluate each outlier detection method
for clf_name, clf in classifiers.items():
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x_test)  # LOF is unsupervised
        scores_prediction = clf.negative_outlier_factor_
    else:
        clf.fit(x_train)
        scores_prediction = clf.decision_function(x_test)
        y_pred = clf.predict(x_test)

    # # Convert the outlier predictions to binary (inliers=0, outliers=1)
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    # # Calculate errors and accuracy
    n_errors = (y_pred != y_test).sum()
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{clf_name} Outlier Detection:")
    print(f"Number of errors: {n_errors}")
    print(f"Accuracy Score: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # # Optionally, plot the decision function scores
    # if clf_name != "Local Outlier Factor":  # LOF doesn't use decision_function
    #     plt.figure(figsize=(10, 6))
    #     plt.title(f"Outlier Detection with {clf_name}")
    #     plt.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], c=y_pred, cmap='coolwarm', marker='o')
    #     plt.xlabel('Feature 1')
    #     plt.ylabel('Feature 2')
    #     plt.show()

# # ML Model Prediction on sample data
new_data = pd.read_csv(r"D:\Desktop\Data Science\Python\Fraud Detection Credit Card 2023\.venv\sample data credit card.csv")

new_predictions = logistic_model.predict(new_data)
print(new_predictions)

unique_predictions = np.unique(new_predictions)
print("Unique predictions:", unique_predictions)


import matplotlib.pyplot as plt

plt.hist(new_predictions, bins=2, edgecolor='black')
plt.xticks([0, 1], ['Valid (0)', 'Fraud (1)'])
plt.xlabel("Prediction")
plt.ylabel("Frequency")
plt.title("Prediction Distribution")
plt.show()

# Initialize counters
count_0 = 0
count_1 = 0

for pred in new_predictions:
    if pred == 0:
        count_0 += 1
    elif pred == 1:
        count_1 += 1

print(f"Valid (0): {count_0}, Fraud (1): {count_1}")


