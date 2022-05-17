from datetime import datetime
from click import style
import pandas as pd
from tabulate import tabulate
from process_dataset import process_data
from utility import print_scores, print_result
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier

def naive_bayes_classifier():
    print ("TRAINING NAIVE BAYES ALGORITHM")
    start_time = datetime.now()
    mnb = MultinomialNB().fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()), 'Score on train (Naive Bayes)', 'Score on test (Naive Bayes)', str(mnb.score(x_train,y_train)), str(mnb.score(x_test,y_test)))

def logistic_regression_classifier():
    print ("TRAINING LOGISTIC REGRESSION ALGORITHM")
    start_time = datetime.now()
    lr = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Logistic Regression)', 'Score on test (Logistic Regression)', str(lr.score(x_train,y_train)), str(lr.score(x_test,y_test)))

def k_nearest_neighbors_classifier():
    print ("TRAINING K-NEAREST NEIGHBOR ALGORITHM")
    start_time = datetime.now()
    knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1).fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (K-Nearest Neighbor)', 'Score on test (K-Nearest Neighbor)', str(knn.score(x_train,y_train)), str(knn.score(x_test,y_test)))

def support_vector_machine_classifier():
    print ("TRAINING SUPPORT VECTOR MACHINE ALGORITHM")
    start_time = datetime.now()
    svm = LinearSVC(C=0.0001).fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Support Vector Machine)', 'Score on test (Support Vector Machine)', str(svm.score(x_train,y_train)), str(svm.score(x_test,y_test)))

def decision_tree_classifier():
    print ("TRAINING DECISION TREE CLASSIFIER")
    start_time = datetime.now()
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Decision Tree)', 'Score on test (Decision Tree)', str(dt.score(x_train,y_train)), str(dt.score(x_test,y_test)))

def bagging_decision_tree_classifier():
    print ("TRAINING BAGGING DECISION TREE CLASSIFIER")
    # max_samples = maximum size 0.5=50% of each sample taken from the full dataset
    # max_features =  maximum of features 1=100% taken here all 10K 
    # n_estimators: number of decision trees 
    start_time = datetime.now()
    bdt = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10).fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Bagging Decision Tree)', 'Score on test (Bagging Decision Tree)', str(bdt.score(x_train,y_train)), str(bdt.score(x_test,y_test)))

def boosting_decision_tree_classifier():
    print ("TRAINING BOOSTING DECISION TREE CLASSIFIER")
    start_time = datetime.now()
    adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4), n_estimators=10,learning_rate=0.6).fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Boosting Decision Tree)', 'Score on test (Boosting Decision Tree)', str(adb.score(x_train,y_train)), str(adb.score(x_test,y_test)))

def random_forest_classifier():
    print ("TRAINING RANDOM FOREST CLASSIFIER")
    start_time = datetime.now()
    rf = RandomForestClassifier(n_estimators=30, max_depth=9).fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Random Forest)', 'Score on test (Random Forest)', str(rf.score(x_train,y_train)), str(rf.score(x_test,y_test)))

def voting_classifier():
    print ("TRAINING VOTING CLASSIFIER")
    mnb = MultinomialNB()
    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=30, max_depth=9)
    svm = LinearSVC(C=0.0001)
    start_time = datetime.now()
    evc = VotingClassifier(estimators=[('mnb', mnb), ('lr', lr), ('rf', rf), ('svm', svm)], voting='hard').fit(x_train, y_train)
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()),'Score on train (Voting Classifier)', 'Score on test (Voting Classifier)', str(evc.score(x_train,y_train)), str(evc.score(x_test,y_test)))

def main():
    naive_bayes_classifier()
    logistic_regression_classifier()
    k_nearest_neighbors_classifier()
    support_vector_machine_classifier() # due to large number of features we use LinearSVC
    decision_tree_classifier()
    # bagging_decision_tree_classifier()
    # boosting_decision_tree_classifier()
    random_forest_classifier()
    voting_classifier()
    print_result()

if __name__ == "__main__":
    print ("------------------")
    x_train, y_train, x_test, y_test = process_data()
    main()
