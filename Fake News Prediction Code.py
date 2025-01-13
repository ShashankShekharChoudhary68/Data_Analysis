# # Importing the Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
# Printing the stopwords in english
print(stopwords.words('english'))

# # Data Preprocessing
# # Loading the dataset to a pandas dataframe
news_dataset = pd.read_csv(r"D:\Desktop\Data Science\Python\Fake News Prediction\.venv\Fake News Dataset.csv")
# print(news_dataset.shape)
# # Print the first 5 rows for the dataframe
# print(news_dataset.head())
# # Counting the number of missing values in the dataset
# print(news_dataset.isnull().sum())
# # Replacing the null values with empty string
news_dataset = news_dataset.fillna('')
# # Merging the author name & news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
# print(news_dataset['content'])

# # Separating the data & label
# x = news_dataset.drop(columns='label',axis=1)
# y = news_dataset['label']
# print(x)
# print(y)

# # Steming: is the process of reducing a word to its Root Word
# # example: actor, actress, acting --> act
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z ]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
# print(news_dataset['content'])

# Separating the data & labels
x = news_dataset['content'].values
y = news_dataset['label'].values
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)

# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
# print(x)

# Splitting the data to training & test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)

# Training the Model: Logistic Regression
model = LogisticRegression()
model.fit(x_train,y_train)

# # Model Evaluation
# # Accuracy score on training data
# x_train_prediction = model.predict(x_train)
# training_data_accuracy = accuracy_score(y_train, x_train_prediction)
# print("Accuracy Score on Training Data: ", training_data_accuracy)
# # Accuracy Score on test data
# x_test_prediction = model.predict(x_test)
# test_data_accuracy = accuracy_score(y_test, x_test_prediction)
# print("Accuracy Score on Test Data: ", test_data_accuracy)

# # Making a Predictive System
# x_news = x_test[1]
# prediction = model.predict(x_news)
# print(prediction)
# if prediction[0]==0:
#     print("The News is Real.")
# else:
#     print("The News is Fake.")
#
# print(y_test[1])











