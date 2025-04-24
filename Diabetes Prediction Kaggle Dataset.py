# # Importing the Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# # Loading the Diabetes Dataset from Kaggle.com
diabetes_dataset = pd.read_csv(r"D:\Desktop\Data Science\Python\Diabetes Prediction\.venv\Diabetes Dataset Kaggle.csv")
# print(diabetes_dataset)

# # Printing the first 5 rows
# print("First 5 Rows:", diabetes_dataset.head())
# print("Columns:", diabetes_dataset.columns)
# print("Shape of the Dataset:", diabetes_dataset.shape)
# print("Data types in the dataset:", diabetes_dataset.dtypes)

# # Checking Null values & Duplicates
# print("Sum of Null Values:", diabetes_dataset.isnull().sum())
# print("Sum of Duplicate Values:", diabetes_dataset.duplicated().sum())
# # Dropping the duplicate rows
new_dataset = diabetes_dataset.drop_duplicates()
# print("Shape of New Dataset", new_dataset.shape)
# print(new_dataset['gender'].unique())
# print('Shape before removing Other values in Gender: ', new_dataset.shape)
new_dataset = new_dataset[new_dataset['gender'] != 'Other']
# print('Shape after removing Other values in Gender: ', new_dataset.shape)
# print(new_dataset['gender'].unique())
# print(new_dataset['smoking_history'].unique())
# print(new_dataset.shape)

# # Statistical Measures of Dataset
# print(new_dataset.describe())
numeric_dataset = new_dataset.select_dtypes(include=['int64','float64'])
# print(numeric_dataset.groupby('diabetes').mean())
# print(new_dataset['diabetes'].value_counts())

# # Visualizations of the Dataset
#
# # Bar Plot for Categorical Variables
# categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']
# for feature in categorical_features:
#     plt.figure(figsize=(8, 6))
#     sns.countplot(data=new_dataset, x=feature, palette='viridis')
#     plt.title(f"Distribution of {feature.capitalize()}")
#     plt.xlabel(feature.capitalize())
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# # Histogram for Continuous Variables
# continuous_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
# for feature in continuous_features:
#     plt.figure(figsize=(8, 6))
#     sns.histplot(data=new_dataset, x=feature, kde=True, color='blue', bins=30)
#     plt.title(f"Distribution of {feature.capitalize()}")
#     plt.xlabel(feature.capitalize())
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.show()

# # Box Plot for continuous variables by Diabetes Status
# for feature in continuous_features:
#     plt.figure(figsize=(8,6))
#     sns.boxplot(data=new_dataset, x='diabetes', y=feature, palette='coolwarm')
#     plt.title(f'{feature.capitalize()} by Diabetes Status')
#     plt.xlabel('Diabetes')
#     plt.ylabel(feature.capitalize())
#     plt.tight_layout()
#     plt.show()

# # Pie Chart for Proportion of Gender & Smoking History Categories
# cat_features = ['gender', 'smoking_history']
# for feature in cat_features:
#     counts = new_dataset[feature].value_counts()
#     plt.figure(figsize=(8,6))
#     plt.pie(counts, labels=counts.index, autopct='%1.2f%%', startangle=90)
#     plt.title(f"Proportions of {feature.capitalize()}")
#     plt.tight_layout()
#     plt.show()
#
# # Count of Diabetic vs Non-Diabetic Pie Chart
# count1 = new_dataset['diabetes'].value_counts()
# plt.figure(figsize=(8,8))
# plt.pie(count1, colors=['green','red'], autopct='%1.1f%%', startangle=90)
# plt.title('Count of Diabetic vs Non-Diabetic', fontsize=14)
# plt.legend(['Non-Diabetic','Diabetic'])
# plt.tight_layout()
# plt.show()
#
# # Count of Diabetic vs Non-Diabetic Genderwise Bar Plot
# count2 = new_dataset.groupby('gender')['diabetes'].value_counts().reset_index(name='count')
# plt.figure(figsize=(12,8))
# sns.barplot(data=count2, x='gender', y='count', hue='diabetes', palette=['blue','orange'])
# plt.title('Count of Diabetic vs Non-Diabetic Genderwise', fontsize=14)
# plt.xlabel('Gender', fontsize=10)
# plt.ylabel('Count', fontsize=10)
# plt.yscale('log')
# plt.legend(title='0=Diabetic,1=Non-Diabetic')
# plt.tight_layout()
# plt.show()
#
# # Average Age of Diabetic vs Non-Diabetic Bar Plot
# avg_age = new_dataset.groupby('diabetes')['age'].mean()
# plt.figure(figsize=(12,8))
# plt.bar(avg_age.index, avg_age.values, color=['green', 'red'])
# plt.title('Average ages of Diabetic vs Non-Diabetic', fontsize=14)
# plt.xlabel('Diabetes', fontsize=10)
# plt.ylabel('Average Age', fontsize=10)
# plt.xticks([0, 1], labels=['Non-Diabetic', 'Diabetic'])
# plt.tight_layout()
# plt.show()

# # Count Plot of People with Diabetes in Age Group 20-50
# age_filtered_data = new_dataset[(new_dataset['age']>=20) & (new_dataset['age']<=50)]
# plt.figure(figsize=(8,6))
# sns.countplot(data=age_filtered_data, x='diabetes', palette=['green', 'red'])
# plt.title('Count of People with Diabetes in Age Group 20-50', fontsize=14)
# plt.xlabel('Diabetes Status', fontsize=10)
# plt.ylabel('Count', fontsize=10)
# plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
# plt.tight_layout()
# plt.show()

# # Box-Plot Diagrams for all numerical columns
# for col in numeric_dataset:
#     plt.figure(figsize=(12, 8))  # Set figure size for each plot
#     sns.boxplot(y=numeric_dataset[col])
#     plt.title(f'Box Plot of {col}')
#     plt.show()
# # Insights: There are outliers in BMI, HbA1c_level & blood_glucose_level columns.

# Calculation of Quartiles & Removal of Outliers
# Calculation of Q1 & Q3 for BMI
q1_bmi = new_dataset['bmi'].quantile(0.25)
q3_bmi = new_dataset['bmi'].quantile(0.75)
iqr_bmi = q3_bmi-q1_bmi
lower_bmi = q1_bmi-1.5*iqr_bmi
upper_bmi = q3_bmi+1.5*iqr_bmi
# Calculation of Q1 & Q3 for HbA1c_level
q1_HbA1c = new_dataset['HbA1c_level'].quantile(0.25)
q3_HbA1c = new_dataset['HbA1c_level'].quantile(0.75)
iqr_HbA1c = q3_HbA1c-q1_HbA1c
lower_HbA1c = q1_HbA1c-1.5*iqr_HbA1c
upper_HbA1c = q3_HbA1c+1.5*iqr_HbA1c
# Calculation of Q1 & Q3 for blood_glucose_level
q1_glucose = new_dataset['blood_glucose_level'].quantile(0.25)
q3_glucose = new_dataset['blood_glucose_level'].quantile(0.75)
iqr_glucose = q3_glucose-q1_glucose
lower_glucose = q1_glucose-1.5*iqr_glucose
upper_glucose = q3_glucose+1.5*iqr_glucose
# Removing Outliers
new_dataset = new_dataset[(new_dataset['bmi']>=lower_bmi) & (new_dataset['bmi']<=upper_bmi)]
new_dataset = new_dataset[(new_dataset['HbA1c_level']>=lower_HbA1c) & (new_dataset['HbA1c_level']<=upper_HbA1c)]
new_dataset = new_dataset[(new_dataset['blood_glucose_level']>=lower_glucose) & (new_dataset['blood_glucose_level']<=upper_glucose)]

# # Splitting the Data & Labels
x = new_dataset.drop(columns='diabetes', axis=1)
y = new_dataset['diabetes']
# print(x)
# print(y)

# # Converting Categorical Labels into Numeric Format
encoder = LabelEncoder()
# x = new_dataset.copy()
x['gender'] = encoder.fit_transform(x['gender'])
# print(x['gender'])
x['smoking_history'] = encoder.fit_transform(x['smoking_history'])
# print(x['smoking_history'])

# # Data Standardization
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x)

# # Splitting the Data into Training Data & Testing Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)
# print(x.shape, x_train.shape, x_test.shape)
# print(y.shape, y_train.shape, y_test.shape)

# # Creating the Models for SVM, Logistic, Decision Tree, Random Forest Algorithns
# svm_model = svm.SVC(kernel='linear')
# logistic_model = LogisticRegression(max_iter=500)
# decision_model = DecisionTreeClassifier()
random_model = RandomForestClassifier(class_weight='balanced', max_depth=5, max_features='sqrt', min_samples_leaf=2, min_samples_split=20, n_estimators=200, random_state=42)

# # Training the Models of SVM, Logistic, Decision Tree, Random Forest Algorithms
# svm_model.fit(x_train, y_train)
# logistic_model.fit(x_train, y_train)
# decision_model.fit(x_train, y_train)
# random_model.fit(x_train, y_train)

# # Applying SMOTE to Balance the Dataset
# # Apply SMOTE to balance the training dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
# # Check the new class distribution
# print("Before SMOTE:", y_train.value_counts())
# print("After SMOTE:", pd.Series(y_train_resampled).value_counts())
# # Train the model
random_model.fit(x_train_resampled, y_train_resampled)

# # Optuna-based Hyperparameter Tuning (Using SMOTE Data)
# # # Importing the Dependencies
# import optuna
# # Define the Optuna objective function
# def objective(trial):
#     # Suggest hyperparameters
#     n_estimators = trial.suggest_int("n_estimators", 50, 150, step=25)  # Limit tree count for generalization
#     max_depth = trial.suggest_int("max_depth", 5, 25, step=5)  # Prevent very deep trees
#     min_samples_split = trial.suggest_int("min_samples_split", 5, 20, step=2)  # Avoid small splits
#     min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 15, step=2)  # Encourage bigger leaf nodes
#     max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])  # Control feature selection per split
#     class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])  # Compare options
#
#     # Create the RandomForest model with suggested hyperparameters
#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         max_features=max_features,
#         class_weight=class_weight,
#         random_state=42
#     )
#
#     # Perform cross-validation
#     scores = cross_val_score(model, x_train_resampled, y_train_resampled, cv=5, scoring='recall')
#
#     # Return the mean accuracy
#     return scores.mean()
#
# # Run Optuna optimization
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)
#
# # Get the best hyperparameters
# best_params = study.best_params
# print("Best Hyperparameters:", best_params)

# # Predicting the Values on Training Data
# svm_x_train_pred = svm_model.predict(x_train)
# logistic_x_train_pred = logistic_model.predict(x_train)
# decision_x_train_pred = decision_model.predict(x_train)
random_x_train_pred = random_model.predict(x_train_resampled)

# # Model Evaluation
# # Accuracy Score on Training Data
# svm_train_accuracy_score = accuracy_score(y_train, svm_x_train_pred)
# print(f"Accuracy Score on Training Data for SVM(SVC) Model:{svm_train_accuracy_score}")
# logistic_train_accuracy_score = accuracy_score(y_train, logistic_x_train_pred)
# print(f"Accuracy Score on Training Data for Logistic Regression Model:{logistic_train_accuracy_score}")
# decision_train_accuracy_score = accuracy_score(y_train, decision_x_train_pred)
# print(f"Accuracy Score on Training Data for Decision Tree Classifier Model:{decision_train_accuracy_score}")
# random_train_accuracy_score = accuracy_score(y_train_resampled, random_x_train_pred)
# print(f"Accuracy Score on Training Data for Random Forest Classifier Model:{random_train_accuracy_score}")

# # Predicting the Values on Test Data
# svm_x_test_pred = svm_model.predict(x_test)
# logistic_x_test_pred = logistic_model.predict(x_test)
# decision_x_test_pred = decision_model.predict(x_test)
random_x_test_pred = random_model.predict(x_test)

# # Accuracy Score on Test Data
# svm_test_accuracy_score = accuracy_score(y_test, svm_x_test_pred)
# print(f"Accuracy Score on Test Data for SVM(SVC) Model:{svm_test_accuracy_score}")
# logistic_test_accuracy_score = accuracy_score(y_test, logistic_x_test_pred)
# print(f"Accuracy Score on Test Data for Logistic Regression Model:{logistic_test_accuracy_score}")
# decision_test_accuracy_score = accuracy_score(y_test, decision_x_test_pred)
# print(f"Accuracy Score on Test Data for Decision Tree Classifier Model:{decision_test_accuracy_score}")
# random_test_accuracy_score = accuracy_score(y_test, random_x_test_pred)
# print(f"Accuracy Score on Test Data for Random Forest Classifier Model:{random_test_accuracy_score}")

# # Precision Score for Random Forest Classifier Model
# # For Training Data
# random_training_prec_score = precision_score(y_train_resampled, random_x_train_pred)
# print("Precision Score for Random Forest Classifier Model on Training Data:",random_training_prec_score)
# # For Test Data
# random_test_prec_score = precision_score(y_test, random_x_test_pred)
# print("Precision Score for Random Forest Classifier Model on Test Data:", random_test_prec_score)

# # Recall Score for Random Forest Classifier
# # For Training Data
# random_train_recall_score = recall_score(y_train_resampled, random_x_train_pred)
# print("Recall Score for Random Forest Classifier on Training Data:",random_train_recall_score)
# # For Test Data
# random_test_recall_score = recall_score(y_test, random_x_test_pred)
# print("Recall Score for Random Forest Classifier on Test Data:",random_test_recall_score)

# # F1 Score for Random Forest Classifier
# # For Training Data
# random_train_f1_score = f1_score(y_train_resampled, random_x_train_pred)
# print("F1 Score for Random Forest Classifier on Training Data:",random_train_f1_score)
# # For Test Data
# random_test_f1_score = f1_score(y_test, random_x_test_pred)
# print("F1 Score for Random Forest Classifier on Test Data:",random_test_f1_score)

# #Cross Validation Score for Random Forest Classifier Model
# cv_scores = cross_val_score(random_model, x, y, cv=5, scoring='accuracy')
# print(f"Cross-Validation Score for each fold (Random Forest Classifier Model):{cv_scores}")
# print(f"Mean Cross Validation Scores(Random Forest Classifier Model):{cv_scores.mean()}")

# # Evaluate model performance by Classification Report
# print(classification_report(y_test, random_x_test_pred))

# # ROC Curve & AUC
# y_pred_prob = random_model.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# roc_auc = auc(fpr, tpr)
# # Plot Curve
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.grid(True)
#
# # PR Curve (Precision-Recall Curve)
# precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
# average_precision = average_precision_score(y_test, y_pred_prob)
# pr_auc = auc(recall, precision)
# # # Curve
# plt.subplot(1, 2, 2)
# plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP={average_precision:.2f}, AUC={pr_auc})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # To reduce the overfitting of random forest model, we will do hyper-parameter tuning by param_grid which will give the best hyper-parameters.
# # Define the hyperparameter grid for Random Forest
# param_grid = {
#     'n_estimators': [50, 100, 200],  # Reduce trees to prevent excessive fitting
#     'max_depth': [5, 10, 15],  # Restrict depth to control overfitting
#     'min_samples_split': [5, 10, 20],  # Prevent splits on very small samples
#     'min_samples_leaf': [2, 5, 10],  # Ensure larger leaf sizes
#     'max_features': ['sqrt', 'log2'],  # Reduce number of features used per split
#     'class_weight': ['balanced', None]  # Handle class imbalance if needed
# }
# # Initialize GridSearchCV with accuracy as the refit metric
# grid_search = GridSearchCV(
#     estimator=random_model,
#     param_grid=param_grid,
#     scoring={'precision': 'precision', 'recall': 'recall'},
#     refit='recall',  # Select the best model based on accuracy
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )
# # Perform Grid Search
# grid_search.fit(x_train, y_train)
# # Get the best parameters based on accuracy
# best_params = grid_search.best_params_
# print("Best Hyperparameters (Based on Accuracy):", best_params)

# # Making a Predictive System
# # Getting the input data
# input_data = ['Female',61.0,0,0,'not current',39.36,9.0,140]
# # Initializing the Label Encoders for Categorical Variables
# gender_encoder = LabelEncoder()
# smoking_history_encoder = LabelEncoder()
# # Fit the Encoders with the categories used during training
# gender_encoder.fit(['Female','Male','Other'])
# smoking_history_encoder.fit(['never','No Info','current','former','ever','not current'])
# # Convert the Categorical Labels into Numerical Format
# input_data[0] = gender_encoder.transform([input_data[0]])[0]
# input_data[4] = smoking_history_encoder.transform([input_data[4]])[0]
# # Changing the input data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)
# # Reshape this data as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# print(input_data_reshaped.shape)
# # Create a DataFrame to ensure correct feature names
# columns = ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']
# input_data_df = pd.DataFrame(input_data_reshaped, columns=columns)
# print(input_data_df)
# # Standardize the input data
# std_data = scaler.transform(input_data_df)
# # Convert the scaled data back to a DataFrame, maintaining the column names
# std_data_df = pd.DataFrame(std_data, columns=columns)
# print(std_data_df)
# # Prediction
# prediction = random_model.predict(std_data_df)
# print(prediction)
# if prediction[0] == 0:
#     print("The Patient is Non-Diabetic.")
# else:
#     print("The Patient is Diabetic.")











