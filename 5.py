import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def lift_dataset(X):
    """
    Lifts the dataset by generating interaction features.

    Args:
    - X: Features

    Returns:
    - X_lifted: Features with interaction features
    """
    num_features = X.shape[1]
    interaction_features = []
    for i in range(num_features):
        for j in range(i, num_features):
            interaction_features.append(X[:, i] * X[:, j])
    X_lifted = np.hstack((X, np.column_stack(interaction_features)))
    return X_lifted

# Load dataset
X = np.load("X_N_1000_d_40_sig_0_01.npy")
y = np.load("y_N_1000_d_40_sig_0_01.npy")

# Lift the dataset
X_lifted = lift_dataset(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_lifted, y, test_size=0.30, random_state=42)

# Set up 5-fold cross-validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Define alpha values for Lasso regularization
alphas = np.logspace(-10, 10, 21)

# Lists to store mean and standard deviation of RMSE for each alpha value
mean_rmse = []
std_rmse = []

# Perform 5-fold cross-validation for different alpha values
for alpha in alphas:
    # Create Lasso regression model with current alpha value
    model = Lasso(alpha=alpha, max_iter=10000)
    # Compute cross-validated RMSE scores
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    # Append mean and standard deviation of RMSE to respective lists
    mean_rmse.append(-np.mean(scores))
    std_rmse.append(np.std(scores))

# Find the optimal alpha that minimizes the mean RMSE across folds
optimal_alpha = alphas[np.argmin(mean_rmse)]

# Train Lasso model on entire training set using the optimal alpha
model_optimal = Lasso(alpha=optimal_alpha, max_iter=10000)
model_optimal.fit(X_train, y_train)

# Compute RMSE for train and test sets with the optimal alpha
train_rmse = np.sqrt(mean_squared_error(y_train, model_optimal.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, model_optimal.predict(X_test)))

# Print the train and test RMSE with the optimal alpha
print(f"Train RMSE with optimal alpha {optimal_alpha}: {train_rmse}")
print(f"Test RMSE with optimal alpha {optimal_alpha}: {test_rmse}")

# Concatenate model coefficients and intercept
params = np.hstack((model_optimal.intercept_, model_optimal.coef_))
# Select parameters with absolute value larger than 1e-3
significant_params = params[np.abs(params) > 1e-3]

# Print parameters with absolute value larger than 1e-3
print("Parameters with absolute value larger than 1e-3:", significant_params)

# Plot cross-validation mean RMSE as a function of alpha with error bars
plt.errorbar(alphas, mean_rmse, yerr=std_rmse, fmt='-o', capsize=5)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean RMSE')
plt.title('Cross-validation Mean RMSE vs Alpha')
plt.grid(True)
plt.show