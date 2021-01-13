import os
import time
import scipy
import numpy as np
import sklearn.svm as svm
import sklearn.tree as tree
from sklearn import datasets
import matplotlib.pyplot as plt
from Submission.Code import kaggle
from sklearn import model_selection
import sklearn.linear_model as linear
import sklearn.neural_network as neural
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors.regression import KNeighborsRegressor


def decision_trees(train_x, train_y, test_x):
    time_list_temp = []
    result = None
    for max_depth in range(3, 16, 3):
        time_starts = time.time()
        result = cross_val_score(estimator = tree.DecisionTreeRegressor(max_depth = max_depth), X = train_x, y = train_y, scoring = "neg_mean_absolute_error", cv = 5, n_jobs = -1)
        result_average = result.mean() * -1
        time_used = time.time() - time_starts
        time_list_temp.append(time_used)
        print(result_average)
    time_list = [int(t * 1000) for t in time_list_temp]
    model = tree.DecisionTreeRegressor(max_depth = 9)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    #decision_tree_plot(time_list)
    return prediction


def decision_tree_plot(time):
    plt.plot(range(3, 16, 3), time)
    plt.xlabel("Max_depth")
    plt.ylabel("Time(ms)")
    plt.savefig("../Figures/Question1.png")


def nearest_neighbors(train_x, train_y, test_x):
    results = []
    nearest_neighbors = [5]
    for num in nearest_neighbors:
        result = cross_val_score(estimator = KNeighborsRegressor(n_neighbors = num, p = 100), X = train_x, y = train_y, scoring = "neg_mean_absolute_error", cv = 5, n_jobs = -1)
        result_average = result.mean() * -1
        results.append(result_average)
        print(results)
    # model = KNeighborsRegressor(n_neighbors = 5, p = 1)
    # model.fit(train_x, train_y)
    # prediction = model.predict(test_x)
    return prediction


def linear_model(train_x, train_y, test_x):
    alpha = [1e-6, 1e-4, 1e-2, 1, 10]
    ridge_error = []
    lasso_error = []
    #for cons in alpha:
    #    result_r = cross_val_score(linear.Ridge(alpha = cons), X = train_x, y = train_y, n_jobs = -1, scoring = "neg_mean_absolute_error", cv = 5)
    #    result_l = cross_val_score(linear.Lasso(alpha = cons), X = train_x, y = train_y, n_jobs = -1, scoring = "neg_mean_absolute_error", cv = 5)
    #Lasso with alpha = 10
    model = linear.Lasso(alpha = 10)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    print(model.coef_)
    print(np.argsort(model.coef_))
    print(np.argsort(model.coef_)[:4])
    return prediction


def SVM(train_x, train_y, test_x):
    results = []
    train_x = StandardScaler().fit_transform(train_x)
    test_x = StandardScaler().fit_transform(test_x)
    kernel = ["poly", "rbf"]
    degrees = [1, 2]
    # for ker in kernel:
    #     for deg in degrees:
    #         result = cross_val_score(estimator = svm.SVC(kernel = ker, degree = deg), X = train_x, y = train_y, scoring = "neg_mean_absolute_error", cv = 5, n_jobs = -1)
    #         results.append(np.mean(result))
    #         print(ker, deg, result)
    parameters = {"kernel": ["linear", "poly", "rbf"], "gamma": [1]}
    #model = model_selection.GridSearchCV(svm.SVC(), param_grid = parameters, n_jobs = -1, verbose = 10)
    model = svm.SVC(kernel = "poly", degree = 1)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return prediction


def neural_networks(train_x, train_y, test_x):
    results = []
    train_x = StandardScaler().fit_transform(train_x)
    test_x = StandardScaler().fit_transform(test_x)
    layers = range(10, 41, 10)
    for layer in layers:
        result = cross_val_score(estimator = neural.MLPRegressor(hidden_layer_sizes = layer), X = train_x, y = train_y, scoring = "neg_mean_absolute_error", cv = 5, n_jobs = -1)
        result_average = result.mean() * -1
        results.append(result_average)
    model = model_selection.GridSearchCV(neural.MLPRegressor(), scoring = "neg_mean_absolute_error" , param_grid = {"hidden_layer_sizes" : range(10, 41, 10)}, n_jobs = -1, verbose = 10, return_train_score=True)
    model.fit(train_x, train_y)
    print(model.cv_results_)
    print(model.score(train_x, train_y))

    # model = neural.MLPRegressor(hidden_layer_sizes = 20)
    # model.fit(train_x, train_y)
    # prediction = model.predict(test_x)
    return prediction


def second_level_running(train_x, train_y, test_x):
    #prediction = decision_trees(train_x, train_y, test_x)
    #prediction = nearest_neighbors(train_x, train_y, test_x)
    #prediction = linear_model(train_x, train_y, test_x)
    prediction = SVM(train_x, train_y, test_x)
    #prediction = neural_networks(train_x, train_y, test_x)
    return prediction


def read_data_fb():
    print("Reading facebook dataset ...")
    train_x = np.loadtxt("../../Data/data.csv", delimiter=",")
    train_y = np.loadtxt("../../Data/labels.csv", delimiter=",")
    kaggle_data = np.loadtxt("../../Data/kaggle_data.csv", delimiter=",")
    return (train_x, train_y, kaggle_data)


def compute_error(y_hat, y):
    return np.abs(y_hat - y).mean()


if __name__ == "__main__":
    train_x, train_y, test_x = read_data_fb()
    prediction = second_level_running(train_x, train_y, test_x)
    print("Train=", train_x.shape)
    print("Test=", test_x.shape)
    predicted_y = np.ones(test_x.shape[0]) * -1                 #[-1. -1. -1. ..., -1. -1. -1.]
    file_name = "../Predictions/best.csv"
    # Writing output in Kaggle format
    print("Writing output to ", file_name)
    kaggle.kaggleize(prediction, file_name)