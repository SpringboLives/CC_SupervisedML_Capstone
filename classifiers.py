from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn import metrics
import timeit



# Decision Tree Classifier
def decision_tree(features_train, labels_train, features_test, labels_test):
    best_accuracy = float('-inf')
    best_depth = 0
    for i in range(1, 20):
        classifier = DecisionTreeClassifier(max_depth=i, random_state=25)
        classifier.fit(features_train, labels_train)
        y_pred = classifier.predict(features_test)
        if metrics.accuracy_score(labels_test, y_pred) > best_accuracy:
            best_accuracy = metrics.accuracy_score(labels_test, y_pred)
            best_depth = i
    return "**DECISION TREE**\nBest accuracy = {best_accuracy}\nBest Depth = {best_depth}\n".format(best_accuracy=best_accuracy, best_depth=best_depth)


# Random Forest Classifier
def random_forest(features_train, labels_train, features_test, labels_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=3)
    classifier.fit(features_train, labels_train.ravel())
    y_pred = classifier.predict(features_test)
    return "**RANDOM FOREST**\nAccuracy = {}\n".format(metrics.accuracy_score(labels_test, y_pred))


def k_nearest_neighbor(features_train, labels_train, features_test, labels_test):
    best_accuracy = float('-inf')
    best_neighbor_amt = 0
    for i in range(1, 100):
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(features_train, labels_train.ravel())
        y_pred = classifier.predict(features_test)
        if metrics.accuracy_score(labels_test, y_pred) > best_accuracy:
            best_accuracy = metrics.accuracy_score(labels_test, y_pred)
            best_neighbor_amt = i
    return "**K NEAREST NEIGHBOR**\nBest Accuracy = {best_accuracy}\nBest Neighbor Amt. = {best_neighbor_amt}\n".format(best_accuracy=best_accuracy,
                                                                                              best_neighbor_amt=best_neighbor_amt)
def k_nearest_neighbor_graph(features_train, labels_train, features_test, labels_test):
    values = []
    for i in range(1, 100):
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(features_train, labels_train.ravel())
        y_pred = classifier.predict(features_test)
        values.append(metrics.accuracy_score(y_pred=y_pred, y_true=labels_test))
    plt.plot(range(1, 100), values)
    plt.show()

def decision_tree_graph(features_train, labels_train, features_test, labels_test):
    values = []
    for i in range(1, 20):
        classifier = DecisionTreeClassifier(max_depth=i, random_state=25)
        classifier.fit(features_train, labels_train)
        y_pred = classifier.predict(features_test)
        values.append(metrics.accuracy_score(y_pred=y_pred, y_true=labels_test))
    plt.plot(range(1, 20), values)
    plt.show()

def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    score = model.score(x, y)
    y_pred = model.predict(x)
    plt.scatter(x, y, alpha=0.002)
    plt.plot(x, y_pred, color='red')
    plt.xlabel('Education Level')
    plt.ylabel('Income Reported')
    plt.title('Income Reported vs. Education Level')
    plt.show()
    return score

def multi_linear_regression(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    return score, model.coef_
