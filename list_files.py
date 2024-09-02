#Importing necessary libraries
import pandas as pd
import numpy as np
import os


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




directory_path = '/Users/bintssss/Desktop/idsproject'  

#This lists all files in the input directory to comnnfirm the correct file name
for dirname, _, filenames in os.walk(directory_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# This step loads the dataset
df = pd.read_csv('Phishing_Legitimate_full.csv')
print(df.columns)
#To check the column names in order to find the correct label column

#this replaces the 'label' with 'CLASS_LABEL'
x = df.drop('CLASS_LABEL', axis=1) # for the features
y = df['CLASS_LABEL'] #labels (0: Legitimate, 1: Phishing)

print(y.value_counts()) #to check the class distribution

#This step is to explore the dataset
print("First few rows of the dataset:")
print(df.head())

print("/nDataset Information:")
print(df.info())

print("/Summary statistics of numerical features:")
print(df.describe())

print("/nChecking for missing values:")
print(df.isnull().sum())

# To check class balance
print("/nClass distribution (Legitimate vs Phishing):")
print(df['CLASS_LABEL'].value_counts())

#Preprocessing the data 
# firstly, seperate the features and labels
x = df.drop('CLASS_LABEL', axis =1) #Features
y = df['CLASS_LABEL'] #This line targers Variable (0: Legitimate, 1: Phishing)

#This is to standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Building and training the SVM Model by statrting wth a basic SVM model (RBF kernel)
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

#To Make predictions and evaluate the model
y_pred = svm_model.predict(X_test)

#To calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the basic SVM model: {accuracy*100:.2f}%")

#to print a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#for the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Hyperparameter Tuning with GridSearchCV
#This defines the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

#Perfom Grid Search with 5-fold cross-validation
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

#To gather the best parameters from the grid search
print(f"\nBest hyperparameters from Grid Search: {grid.best_params_}")

#Now I will evaluate the tuned model
y_pred_tuned = grid.predict(X_test)

# The calculate the accuracy for the tuned model
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"\nAccuracy of the tuned SVM model: {tuned_accuracy*100:.2f}%")

#Gathering a classification report for the tuned model
print("\nTuned Model Classification Report:")
print(classification_report(y_test, y_pred_tuned))

#Now to get the confusion matrix for the tuned model
print("\nConfusion Matrix (Tuned Model):")
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Tuned Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the tuned model and scaler
import joblib

joblib.dump(grid, 'tuned_svm_model.pkl')  
joblib.dump(scaler, 'scaler.pkl')

