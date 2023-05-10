# import clf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import joblib as jlib

# Importing dataset for the Linear Regression implementation
df = pd.read_csv("../LinearRegressionModel/DataSet/LinearRegressionSheet1.csv")

# Getting the insight of the data with the statistical information about data
print(df.describe())

# Data Wrangling
# The correlation coefficient is a statistical measure that indicates the degree of linear relationship between two variables
print(df.corr())

# Check is there any missing values across each column
print(df.isnull().any())

# Count of missing values of each column
print(df.isna().sum())

df_null_check = df.isnull()
df_null_check['X'] = df_null_check['X'].astype(int)
df_null_check['Y'] = df_null_check['Y'].astype(int)

print(df_null_check.head())
msno.bar(df_null_check)

# No null value found
# Now implementing Linear Regression on dataset

X = df.drop('Y', axis=1).values
Y = df[['Y']].values

# Checking the relationship between X and Y columns

plt.figure(figsize=(10, 10))
plt.scatter(X, Y)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('Graph', fontsize=20)
plt.show()

# Spliting the data into train test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Initializing the Model
# model = LinearRegression()

# Fitting the Model
# model.fit(X_train, Y_train)

# Prediction
model = jlib.load("myPrediction.joblib")
prediction = model.predict(X_test)

# jlib.dump(model, "myPrediction.joblib")


# Measuring the Accuracy
score = model.score(Y_test, prediction)
# score = confusion_matrix(Y_test, prediction)
avg_score = np.mean(score)
print("Score : " + str(avg_score * 100) + "%")
