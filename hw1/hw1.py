# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.train_loss = []
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        
        X_with_ones = np.insert(X, 0, 1, axis=1)
        # print(X_with_ones)
        X_transpose = np.transpose(X_with_ones)
        X_transpose_X = np.dot(X_transpose, X_with_ones)
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        theta = np.dot(np.dot(X_transpose_X_inv, X_transpose), y)
        
        self.closed_form_weights = theta[1:]
        self.closed_form_intercept = theta[0]
        return
        pass

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs, batch_size=32):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        sample_num, var_num = X.shape
        
        self.gradient_descent_weights = np.zeros(var_num)
        self.gradient_descent_intercept = 0
        self.train_loss = []
        
        for epoch in range(epochs):
            #every epoch we shuffle the data to avoid the order of data
            indices = np.arange(sample_num)
            np.random.shuffle(indices)
            # print(indices)
            X = X[indices]
            y = y[indices]
            # print(X)
            # print(y)
            for batch in range(sample_num//batch_size):
                # here is the start and end index of the batch
                start = batch * batch_size
                end = start + batch_size
                # print(start, end)
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                prediction = self.gradient_descent_predict(X_batch)
                gradient_weight = (-2/batch_size) * np.dot(np.transpose(X_batch), (y_batch - prediction))
                gradient_intercept = (-2/batch_size) * np.sum(y_batch - prediction)
                self.gradient_descent_weights -= lr * gradient_weight
                self.gradient_descent_intercept -= lr * gradient_intercept
            
            loss = self.gradient_descent_evaluate(X, y)
            self.train_loss.append(loss)
            # input()
        return 
        pass
        

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        diff = prediction - ground_truth
        diff = np.square(diff)
        result = np.sum(diff) / len(diff)
        return result
        # Return the value.
        pass

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        result = np.dot(X, self.closed_form_weights)
        result = result + self.closed_form_intercept
        return result
        # Return the prediction.
        pass

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        result = np.dot(X, self.gradient_descent_weights)
        result = result + self.gradient_descent_intercept
        return result
        # Return the prediction.
        pass
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, marker='o', linestyle='-')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.show()
        return 
        pass

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    # print(train_x)
    # print(train_y)
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.000180, epochs=1500)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    # closed_form_loss = LR.closed_form_evaluate(train_x, train_y)
    # gradient_descent_loss = LR.gradient_descent_evaluate(train_x, train_y)
    # print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    # print(closed_form_loss, gradient_descent_loss)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
    LR.plot_learning_curve()