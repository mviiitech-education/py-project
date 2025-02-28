import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('Dataset.csv')

# Display the first 20 rows of the dataset
print(dataset.head(20))

# Display the shape of the dataset
print(dataset.shape)

# Display statistical details of the dataset
print(dataset.describe())

# Define features (X) and target (y)
X = dataset.iloc[:, :-1].values  # All columns except the last one (Humidity and Heatcelcius)
y = dataset.iloc[:, -1].values   # Last column (MoisturePercentage)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest Regression
rf_regressor = RandomForestRegressor(n_estimators=20, random_state=0)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate Random Forest Regression
print('Random Forest Regression Metrics:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print('R2 Score:', metrics.r2_score(y_test, y_pred_rf))

# Create a DataFrame to compare actual and predicted values
df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
print(df_rf)

# Linear Regression
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)
y_pred_lr = lr_regressor.predict(X_test)

# Evaluate Linear Regression
print('\nLinear Regression Metrics:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_lr))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_lr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))
print('R2 Score:', metrics.r2_score(y_test, y_pred_lr))

# Create a DataFrame to compare actual and predicted values
df_lr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lr})
print(df_lr)

# Plotting
plt.figure(figsize=(12, 6))

# Humidity vs MoisturePercentage
plt.subplot(1, 2, 1)
plt.scatter(dataset['Humidity'], dataset['MoisturePercentage'], color='blue')
plt.title('Humidity vs MoisturePercentage')
plt.xlabel('Humidity')
plt.ylabel('MoisturePercentage')

# Heatcelcius vs MoisturePercentage
plt.subplot(1, 2, 2)
plt.scatter(dataset['Heatcelcius'], dataset['MoisturePercentage'], color='red')
plt.title('Heat-Air Temperature vs MoisturePercentage')
plt.xlabel('Heat in Celcius')
plt.ylabel('MoisturePercentage')

plt.show()

# Print Linear Regression coefficients
print('\nLinear Regression Coefficients:')
print('Intercept:', lr_regressor.intercept_)
print('Coefficients:', lr_regressor.coef_)