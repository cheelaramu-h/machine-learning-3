import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data_generator import postfix, liftDataset

def main():
    # Dataset parameters
    N = 1000
    sigma = 0.01
    d = 5

    psfx = postfix(N, d, sigma)

    # Load dataset and lift features
    X = np.load("X" + psfx + ".npy")
    X = liftDataset(X)
    y = np.load("y" + psfx + ".npy")

    print("Dataset has n=%d samples, each with d=%d features," % X.shape,
          "as well as %d labels." % y.shape[0])

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    print("Randomly split dataset to %d training and %d test samples" % (
        X_train.shape[0], X_test.shape[0]))

    # Fraction values
    fractions = np.arange(0.1, 1.1, 0.1)

    train_rmse_values = []
    test_rmse_values = []

    # Train linear regression models with different fractions of the training set
    for fr in fractions:
        # Number of samples to use for training
        n_train_samples = int(fr * X_train.shape[0])
        
        # Select subset of training data
        X_train_subset = X_train[:n_train_samples]
        y_train_subset = y_train[:n_train_samples]

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_subset, y_train_subset)

        # Predictions on training set
        y_train_pred = model.predict(X_train_subset)
        train_rmse = np.sqrt(mean_squared_error(y_train_subset, y_train_pred))

        # Predictions on test set
        y_test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Store RMSE values
        train_rmse_values.append(train_rmse)
        test_rmse_values.append(test_rmse)

        print("Fraction: %.2f, Train RMSE: %.4f, Test RMSE: %.4f" % (fr, train_rmse, test_rmse))

    # Plot RMSE values
    plt.plot(fractions, train_rmse_values, label='Train RMSE')
    plt.plot(fractions, test_rmse_values, label='Test RMSE')
    plt.xlabel('Fraction of Training Samples')
    plt.ylabel('RMSE')
    plt.title('Train and Test RMSE vs. Fraction of Training Samples')
    plt.legend()
    plt.show()

    # Finally trained model coefficients and intercept
    print("Finally trained model coefficients:", model.coef_)
    print("Finally trained model intercept:", model.intercept_)

if __name__ == "__main__":
    main()
