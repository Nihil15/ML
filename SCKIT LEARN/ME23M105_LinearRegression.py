import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
data = pd.read_csv('LinearRegression -Training_data.csv')

# Split the data into x& y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalizing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
class MLR:
    def __init__(self):
        pass

    def fit(self, X, y, alpha=0.01, num_iterations=1000, tolerance=1e-4):
        X = np.insert(X, 0, 1, axis=1)
        m, n = X.shape
        self.weights = np.zeros(n)  # Initialize weights
        self.cost_history = []  # Store cost history
        self.r_squared_history = []  # Store R-squared history
        self.tolerance = tolerance

        for _ in range(num_iterations):
            y_pred = X @ self.weights
            error = y_pred - y
            cost = np.sum(error ** 2) / (2 * m)
            self.cost_history.append(cost)

            r_squared = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))
            self.r_squared_history.append(r_squared)

            if len(self.cost_history) > 1 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break

            gradient = X.T @ error / m
            self.weights -= alpha * gradient

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # bias
        return X @ self.weights

# Initialize the MLR model
mlr_model = MLR()
mlr_model.fit(X_train_scaled, y_train)

# predictions
y_pred_mlr = mlr_model.predict(X_test_scaled)

# Train scikit-learn model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# predictions using the LinearRegression model
y_pred_lr = lr_model.predict(X_test_scaled)


print("MLR Results:")
print("Final error:", mlr_model.cost_history[-1])
print("R-squared:", mlr_model.r_squared_history[-1])
print("MAE:", mean_absolute_error(y_test, y_pred_mlr))
print("MSE:", mean_squared_error(y_test, y_pred_mlr))

print("SK-learn results:")
print("R-squared:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))

# Plot cost vs iterations for MLR
plt.plot(range(len(mlr_model.cost_history)), mlr_model.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations (MLR)')
plt.show()

# Plot R-squared vs iterations for MLR
plt.plot(range(len(mlr_model.r_squared_history)), mlr_model.r_squared_history)
plt.xlabel('Iterations')
plt.ylabel('R-squared')
plt.title('R-squared vs Iterations (MLR)')
plt.show()
