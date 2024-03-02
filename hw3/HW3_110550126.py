# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    label, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini_impurity = 1 - np.sum(probabilities ** 2)
    return gini_impurity
    pass

# This function computes the entropy of a label array.
def entropy(y):
    label, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy_value = -np.sum(probabilities * np.log(probabilities))
    return entropy_value
    pass
def cal(func,y):
    if(func=='gini'):
        return gini(y)
    if(func=='entropy'):
        return entropy(y)
    
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class Node():
    def __init__(self, predicted_class,feature_index,threshold, depth, is_leaf):
        self.predicted_class = predicted_class
        self.feature_index = feature_index
        self.threshold = threshold
        self.depth = depth
        self.is_leaf = is_leaf
        self.left = None
        self.right = None
        
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.num_classes = None
        self.num_features = None
        self.root = None
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def find_best_split(self, X, y):
        ideal_feature_index = None
        ideal_threshold = None
        ideal_impurity = None
        num_samples = len(y)
        if num_samples <= 1:
            return None, None
        best_impurity = cal(self.criterion,y)
        for feature_index in range(self.num_features):
            feature_values = X[:, feature_index]
            # print(feature_values)
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                y_left = y[X[:, feature_index] <= threshold]
                y_right = y[X[:, feature_index] > threshold]
                if len(y_left) > 0 and len(y_right) > 0:
                    current_impurity = (len(y_left) / num_samples) * cal(self.criterion,y_left) + (len(y_right) / num_samples) * cal(self.criterion,y_right)
                    if current_impurity < best_impurity:
                        best_impurity = current_impurity
                        ideal_feature_index = feature_index
                        ideal_threshold = threshold
        return ideal_feature_index, ideal_threshold
    
    def grow_tree(self, X, y, depth):
        if depth == self.max_depth:# leaf node
            return Node(predicted_class=np.argmax(np.bincount(y)),feature_index=None,threshold=None, depth=depth, is_leaf=True)
        feature_index, threshold = self.find_best_split(X, y)
        if feature_index is None or threshold is None:
            return Node(predicted_class=np.argmax(np.bincount(y)),feature_index=None,threshold=None, depth=depth, is_leaf=True)
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        current_node = Node(None,feature_index,threshold, depth, False)
        current_node.left = self.grow_tree(X[left_indices], y[left_indices], depth + 1)
        current_node.right = self.grow_tree(X[right_indices], y[right_indices], depth + 1)
        return current_node
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.root = self.grow_tree(X, y, 0)
        return 
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    
    def predict_single(self,x, node):
        while node.is_leaf == False:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
        
        
    def predict(self, X):
        res = []
        for i in range(len(X)):
            temp = self.predict_single(X[i],self.root)
            res.append(temp)
        numpy_res = np.array(res)
        return numpy_res
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        importances = np.zeros(columns.shape[0])
        queue = [self.root]
        while len(queue) > 0:
            
            if queue[0].is_leaf == False:
                importances[queue[0].feature_index] +=1
                queue.append(queue[0].left)
                queue.append(queue[0].right)
            queue.pop(0)
                
        # print(importances)
        plt.barh(columns, importances)  # Use barh instead of bar for horizontal bar chart
        
        plt.title("Feature Importance")
        # plt.xticks(rotation=90)
        plt.show()
        pass

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.weak_classifiers = []
        self.alphas = []
        
    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        num_samples = X.shape[0]
        weights = np.full(num_samples, (1 / num_samples))
        for _ in range(self.n_estimators):
            indices = np.random.choice(num_samples, num_samples, p=weights)
            X_sample = X[indices]
            y_sample = y[indices]
            weak_classifier = DecisionTree(criterion=self.criterion, max_depth=1)
            weak_classifier.fit(X_sample, y_sample)
            predictions = weak_classifier.predict(X)
            correct = np.where(predictions == y, -1, 1)            
            error = np.sum(weights[y != predictions])
            alpha = 0.5 * np.log((1 - error) / error)
            
            # print(alpha)
            new_weights = weights * np.exp(alpha * correct)
            weights = new_weights / np.sum(new_weights)
            self.weak_classifiers.append(weak_classifier)
            self.alphas.append(alpha)
        pass

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for alpha, weak_classifier in zip(self.alphas, self.weak_classifiers):
            temp = weak_classifier.predict(X)
            temp = np.where(temp == 0, -1, 1)
            predictions += alpha * temp
        prediction = np.sign(predictions)
        prediction = np.where(prediction == -1, 0, 1)
        return prediction
        pass

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    # print(X_train, y_train)
    # num_observation = y_train.shape[0]
    # print(num_observation)
    # print(y_train.reshape(num_observation),)
# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    #tree.plot_feature_importance_img(train_df.drop(["target"], axis=1).columns)
# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=12)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
