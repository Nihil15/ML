import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # to prevent log(0) case
    cost = (1/m) * ((-y).T.dot(np.log(h + epsilon)) - (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost

# Gradient of the cost function
def compute_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (1/m) * X.T.dot(h - y)
    return grad

# Gradient Descent function with early stopping
def gradient_descent(X, y, theta, learning_rate, iterations, tol):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        grad = compute_gradient(X, y, theta)
        theta = theta - learning_rate * grad
        cost = compute_cost(X, y, theta)
        cost_history[i] = cost.item()
        if i > 0 and (cost_history[i-1] - cost_history[i]) < tol:
            print(f"Converged at iteration {i}")
            break
    return theta, cost_history[:i]

# Load and preprocess data
data = pd.read_csv(r"C:\Users\user\Desktop\Concrete_Data.xlsx - Sheet1.csv")
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Manual scaling
X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Train the model created from scratch
theta_scratch, _ = gradient_descent(X_train, y_train.values.reshape(-1, 1), np.zeros((X_train.shape[1], 1)), learning_rate=0.01, iterations=10000, tol=1e-5)
y_pred_scratch = (sigmoid(X_test.dot(theta_scratch)) > 0.5).astype(int)
# Evaluate model created from scratch
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
precision_scratch = precision_score(y_test, y_pred_scratch)
recall_scratch = recall_score(y_test, y_pred_scratch)
conf_matrix_scratch = confusion_matrix(y_test, y_pred_scratch)


# Compute ROC curve for both models
y_pred_prob_scratch = sigmoid(X_test.dot(theta_scratch))
fpr_scratch, tpr_scratch, _ = roc_curve(y_test, y_pred_prob_scratch)


# Plot ROC curve for both models
plt.plot(fpr_scratch, tpr_scratch, linestyle='--', color='blue', label='Scratch ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='-', color='red', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# Plot random classifier line

print("Model created from scratch:")
print("Accuracy:", accuracy_scratch)
print("Precision:", precision_scratch)
print("Recall:", recall_scratch)
print("Confusion Matrix:\n", conf_matrix_scratch)
