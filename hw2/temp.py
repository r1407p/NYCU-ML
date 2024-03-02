# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.001, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None
        self.cost = []

    # This function normalizes the data, returns the mean and std of each column.
    def normalization(self, X):
        # Normalize the data, also return the mean and std of each column.
        # Return the normalized data, mean and std of each column.
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        return X, mean, std
    
    def cross_entropy(self, y_pred, y):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        num_samples, num_features = X.shape
        # X, mean, std = self.normalization(X)
        X_prime = np.c_[X, np.ones(num_samples)]
        self.weights = np.zeros(num_features)
        self.intercept = 0

        # for _ in range(self.iteration):
        #     y_pred = self.sigmoid(X @ self.weights + self.intercept)
        #     gradient = -(X_prime.T) @ (y - y_pred) / num_samples
        #     self.weights -= self.learning_rate * gradient[:-1]
        #     self.intercept -= self.learning_rate * gradient[-1]
        #     loss = self.cross_entropy(y_pred, y)
        #     self.loss.append(loss)
        for _ in range(self.iteration):
            y_pred = self.sigmoid(X @ self.weights + self.intercept)
            # Compute gradients
            dw = (1 / num_samples) * np.sum(X.T @ (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            # Update weights and intercept
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
            cost = self.cross_entropy(y_pred, y)
            self.cost.append(cost)

            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        linear_model = X @ self.weights + self.intercept
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        data0, data1 = X[y == 0], X[y == 1]
        self.m0 = np.mean(data0, axis=0)
        self.m1 = np.mean(data1, axis=0)
        self.sw = np.matmul((data0 - self.m0).T, (data0 - self.m0)) + np.matmul((data1 - self.m1).T, (data1 - self.m1))
        # ////
        self.sb = np.outer((self.m0 - self.m1), (self.m0 - self.m1))
        m = np.linalg.inv(self.sw).dot(self.sb)
        eigvals, eigvecs = np.linalg.eig(m)
        self.w = eigvecs[:, np.argmax(eigvals)]

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        projections = np.dot(X, self.w)
        y_pred = [0 if projection < 0 else 1 for projection in projections]
        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        pass
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    # for i in range(1,100):
    #     print(i)
    #     LR = LogisticRegression(learning_rate=0.00001*i, iteration=20000)
    #     LR.fit(X_train, y_train)
    #     y_pred = LR.predict(X_test)
    #     accuracy = accuracy_score(y_test , y_pred)
    #     print(f"Part 1: Logistic Regression")
    #     print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    #     print(f"Accuracy: {accuracy}")

    # # You must pass this assertion in order to get full score for this part.
    # assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

