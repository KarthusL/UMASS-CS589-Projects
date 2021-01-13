import os
import sys
import time
import numpy
import scipy
import pandas
import numpy as np
from sklearn import svm
from Library import kaggle
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

'''for fun, but not related to the homework'''
def GBC(training_set, test_set, kaggle):
    parameters = {"n_estimators": [3000], "min_samples_split": [9, 12, 15], "learning_rate": [0.1]}
    model = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid = parameters, n_jobs = 8, verbose = 10)
    #model = ensemble.GradientBoostingClassifier(n_estimators = 3000, min_samples_split= 9, learning_rate = 0.1)
    X_1, y_1 = datasets.load_svmlight_file(training_set)
    X_2, y_2 = datasets.load_svmlight_file(test_set)
    X = scipy.sparse.vstack((X_1, X_2))
    y = numpy.array(list(y_1) + list(y_2))
    model.fit(X_1, y_1)
    print(model.score(X_2, y_2))

    #print("score for GBC", model.score(X_2, y_2))
    #return model.predict(kaggle)

'''for fun, but not related to the homework'''
def ABC(training_set, test_set, kaggle):
    parameters = {"n_estimators": [1500, 2000, 2500, 3000],
                  "learning_rate": [0.1]}
    model = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid=parameters, n_jobs=4, verbose=10)
    #model = ensemble.AdaBoostClassifier(n_estimators = 300, learning_rate = 0.1)                                                          `
    X_1, y_1 = datasets.load_svmlight_file(training_set)
    X_2, y_2 = datasets.load_svmlight_file(test_set)
    model.fit(X_1, y_1)
    print(model.score(X_2, y_2))

'''for fun, but not related to the homework'''
def Voting(training_set, test_set, kaggle):
    RF = RandomForestClassifier(n_estimators = 3000, min_samples_leaf = 1, min_samples_split = 9)
    GBC = ensemble.GradientBoostingClassifier(n_estimators = 3000, min_samples_split= 9)
    ABD = ensemble.AdaBoostClassifier(n_estimators = 3000)
    model = ensemble.VotingClassifier(estimators = [("RF", RF), ("GBC", GBC), ("ABD", ABD)], voting = "soft", n_jobs = 8)
    print("Here")
    X_1, y_1 = datasets.load_svmlight_file(training_set)
    X_2, y_2 = datasets.load_svmlight_file(test_set)
    X = scipy.sparse.vstack((X_1, X_2))
    y = numpy.array(list(y_1) + list(y_2))
    X_scale = preprocessing.scale(X, with_mean=False)
    kaggle_scale = preprocessing.scale(kaggle, with_mean = False)
    print("There")
    model.fit(X_scale, y)
    print("Where")
    return model.predict(kaggle_scale)

'''Selected for subchoice'''
def SVM(training_set, test_set, kaggle):
    # parameters = {"kernel": ["linear"], "gamma": [1]}
    # model = model_selection.GridSearchCV(svm.SVC(), param_grid = parameters, n_jobs = 8, verbose = 10)
    # model = model_selection.GridSearchCV(svm.LinearSVC(), param_grid = parameters, n_jobs = 8, verbose = 10)
    model = svm.SVC()
    X, y = datasets.load_svmlight_file(training_set)
    SVM_fit_starts = time.time()
    print("-----SVM fit starts-----")
    model.fit(X, y)
    print("-----SVM fit uses-----", time.time() - SVM_fit_starts)
    # print(model.best_params_)
    X, y = datasets.load_svmlight_file(test_set)
    SVM_predict_starts = time.time()
    print("-----SVM predict starts-----")
    print(model.predict(kaggle))
    print("-----SVM predict uses-----", time.time() - SVM_predict_starts)
    return model.predict(kaggle)

'''Selected for subchoice'''
def logistic_regression(training_set, test_set, kaggle):
    parameters = {}
    # model = model_selection.GridSearchCV(logistic_regression(), param_grid = parameters, n_jobs=8, verbose = 10)
    x_training, y_training = datasets.load_svmlight_file(training_set)
    x_test, y_test = datasets.load_svmlight_file(test_set)
    '''Default parameter for Logistic Regression'''
    model = linear_model.LogisticRegression()
    Lr_fit_starts = time.time()
    print("-----Lr fit starts-----")
    model.fit(x_training, y_training)
    print("-----Lr fit uses-----", time.time() - Lr_fit_starts)
    # print(model.best_params_)
    Lr_predict_starts = time.time()
    print("-----Lr predict starts-----")
    print(model.predict(kaggle))
    print("-----Lr predict uses-----", time.time() - Lr_predict_starts)
    return model.predict(kaggle)


'''Primary selected for this homework'''
def random_forest(training_set, test_set, kaggle):
    parameters = {"n_estimators": range(300, 601, 100), "min_samples_leaf": [1], "min_samples_split": [5]}     #use this for 4-a
    model = model_selection.GridSearchCV(RandomForestClassifier(), param_grid = parameters, n_jobs = 8, verbose = 30, return_train_score=True)
    '''Default parameter for random forest classifer'''
    #model = RandomForestClassifier(n_jobs = 600, min_samples_leaf = 1, min_samples_split = 5)   #use this for 3-1, best parameter for 4-a
    X_1, y_1 = datasets.load_svmlight_file(training_set)
    X_2, y_2 = datasets.load_svmlight_file(test_set)
    X = scipy.sparse.vstack((X_1, X_2))                                 #Add two data sets for better result
    y = numpy.array(list(y_1) + list(y_2))                              #Add two data sets for better result
    rf_fit_starts = time.time()
    print("-----rf fit starts-----")
    #model.fit(X_1, y_1)
    print("-----rf fit uses-----", time.time() - rf_fit_starts)
    #print(model.cv_results_["params"])
    #make_csv(model)        #csv result for 4-a
    #plot()                 #use this for graph 4-a
    #print(model.best_params_)  #the best parameter for GridSearchCV
    rf_predict_starts = time.time()
    print("-----rf predict starts-----")
    model.score(X_2, y_2)
    print("-----rf predict uses-----", time.time() - rf_predict_starts)
    return model.predict(kaggle)

#make a csv file for random_forest tree.
def make_csv(model):
    params = model.cv_results_["params"]
    mean_train_score = model.cv_results_["mean_train_score"]
    mean_test_socre = model.cv_results_["mean_test_score"]
    os.chdir("../Predictions")
    with open("random_forest.csv", "w") as fout:
        fout.write("n_estimators,min_samples_leaf,min_samples_split,mean_train_score,mean_test_score,training_error\n")
        for params, mean_train_score, mean_test_socre in zip(params, mean_train_score, mean_test_socre):
            fout.write("{},{},{},{},{}\n".format(params["n_estimators"], params["min_samples_leaf"], params["min_samples_split"], mean_train_score, mean_test_socre, 1 - mean_train_score))

#Plot for the graph. This only will be used for random_forest tree, rf is the primary method for this project.
def plot():
    os.chdir("../Predictions")
    file = pandas.read_csv("random_forest.csv")
    os.chdir("../Figures")
    ne_list = file[["mean_train_score", "mean_test_score", "training_error", "n_estimators"]]
    plt.plot(ne_list["n_estimators"], ne_list["mean_train_score"], label = "train_score")
    plt.plot(ne_list["n_estimators"], ne_list["mean_test_score"], label="test_score")
    plt.plot(ne_list["n_estimators"], ne_list["training_error"], label = "training_error")
    plt.legend(loc = "best")
    plt.title("Result")
    plt.savefig("../Figures/n_estimators.png")


if __name__ == '__main__':
    hw2_train = "../../Data/HW2.train.txt"
    hw2_test = "../../Data/HW2.test.txt"
    hw2_kaggle = "../../Data/HW2.kaggle.txt"
    new_kaggle = "../../Data/HW2_kaggle_new.txt"
    csv_file = "../../Library/result.csv"
    # with open(hw2_kaggle) as fin, open(new_kaggle, "w") as fout:            #change the format of kaggle.txt to the same as regular files
    #     for line in fin:
    #         fout.write("0 " + line)
    kaggle_x, y = datasets.load_svmlight_file(new_kaggle)
    #prediction = SVM(hw2_train, hw2_test, kaggle_x)
    #prediction = random_forest(hw2_train, hw2_test, kaggle_x)
    #prediction = logistic_regression(hw2_train, hw2_test, kaggle_x)
    #prediction = GBC(hw2_train, hw2_test, kaggle_x)             #Writing for higher score on Kaggle
    #prediction = ABC(hw2_train, hw2_test, kaggle_x)             #Writing for higher score on Kaggle
    prediction = Voting(hw2_train, hw2_test, kaggle_x)          #Writing for higher score on Kaggle
    '''For scoring'''
    kaggle.kaggleize(prediction, csv_file)