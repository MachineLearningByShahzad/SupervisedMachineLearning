import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from LinearRegression_Model import LinearRegressionModel
from sklearn.model_selection import train_test_split
import joblib as jlib

"""Implementation of the Linear Regression Model that I have created in LinearRegression_Model.py

There are 8 step to implement the machine learning model:
1)    Import the data
2)    Data Wrangling
3)    Data Visualization
4)    Splitting the data into train test datasets
5)    Initializing the ML Model
6)    Predict the Model
7)    Evaluate the Model Prediction
8)    Deploy the Model

Let's get started
"""

# Step 1 - Importing and analyzing the data
df = pd.read_csv('../LinearRegressionModel/DataSet/LinearRegressionSheet1.csv')
print(df.head())
print('###################################')
print(df.describe())
print('###################################')
print(df.info())

# Step 2 - Data Wrangling The correlation coefficient is a statistical measure that indicates the degree of linear
# relationship between two variables
print(df.corr())
print('###################################')
# Check is there any missing values across each column
print(df.isnull().any())
print('###################################')
# Count of missing values of each column
print(df.isna().sum())
print('###################################')

# Step 3 - Visualizing the data to get better visibility on data for the data cleaning process
df_null_check = df.isnull()
df_null_check['X'] = df_null_check['X'].astype(int)
df_null_check['Y'] = df_null_check['Y'].astype(int)
print(df_null_check.head())
msno.bar(df_null_check)
X = df.drop('Y', axis=1).values
Y = df[['Y']].values
plt.figure(figsize=(10, 10))
plt.scatter(X, Y)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('Graph', fontsize=20)
plt.show()

# Step 4 - Spliting the data into train test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Step 5 - Initializing the Linear Regression Model that I have created using linear algebra and stats principles
model = LinearRegressionModel()

# Step 6 - Fitting & Predicting using my Linear Regression Model
model.fit(Y_train, Y_train)
prediction = model.predict(X_test)

# Step 7 - Evaluating the Model
# Measuring the Accuracy
score = model.score(Y_test, prediction)
score_percentage = score * 100
score_avg_percentage = "{:.2f}".format(score_percentage)
print("Score : " + str(score_avg_percentage) + "%")

# Step 8 - Deployment of the model using jlib.dump to save trained model and then you will only call model.predict
# method next time to get the predictions from the pretrained model using jlib.load

# jlib.load jlib.dump(model, "LinearRegPrediction.joblib")
# model = jlib.load("LinearRegPrediction.joblib")

