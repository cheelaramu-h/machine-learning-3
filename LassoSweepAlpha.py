import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from data_generator import postfix, liftDataset

# Number of samples
N = 1000

# Noise variance
sigma = 0.01

# Feature dimension
d = 40

psfx = postfix(N, d, sigma)
unliftX = np.load("X" + psfx + ".npy")
y = np.load("y" + psfx + ".npy")

print("Dataset has n=%d samples, each with d=%d features," % unliftX.shape, "as well as %d labels." % y.shape[0])
X = np.array(liftDataset(unliftX))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0], X_test.shape[0]))

alpha_val = np.linspace(2 ** -10, 2 ** 10, 1000)
mean_rmse_vals = []
stdev_rmse_vals = []

for alpha in alpha_val:
    model = Lasso(alpha=alpha)
    cv = KFold(
        n_splits=5,
        random_state=42,
        shuffle=True
    )
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    mean_rmse_vals.append(np.mean(scores))
    stdev_rmse_vals.append(np.std(scores))

max_rmse_index = np.argmax(mean_rmse_vals)  # Finding the index of the maximum negative RMSE value
optimum_rmse = mean_rmse_vals[max_rmse_index]
optimal_alpha = alpha_val[max_rmse_index]
print("Highest Negative RMSE:", optimum_rmse)
print("Optimal Alpha:", optimal_alpha)

model = Lasso(alpha=optimal_alpha)

cv = KFold(
    n_splits=5,
    random_state=42,
    shuffle=True
)

scores = cross_val_score(
    model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

print("Cross-validation RMSE for α=%f : %f ± %f" % (optimal_alpha, -np.mean(scores), np.std(scores)))

print("Fitting linear model over entire training set...", end="")
model.fit(X_train, y_train)
print(" done")

# Compute RMSE
rmse_train = np.sqrt(np.mean((y_train - model.predict(X_train)) ** 2))
rmse_test = np.sqrt(np.mean((y_test - model.predict(X_test)) ** 2))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train, rmse_test))

# Plotting Predictive Performance VS Alpha
plt.figure(figsize=(10, 6))
plt.errorbar(alpha_val, mean_rmse_vals, yerr=stdev_rmse_vals, fmt="-o")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Cross-Validation RMSE")
plt.title("Cross-Validation RMSE vs Alpha")
plt.show()

selectedCoeffs = model.coef_[np.abs(model.coef_) > 1e-3]

print("Coefficients: ")
for val in selectedCoeffs:
    print(val)

print()
print("Intercept: ", model.intercept_)
