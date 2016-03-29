__author__ = 'justinkreft'

import pandas as pd
import numpy as np
import random
import math
import pickle
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.ensemble

#Global Definitions for variables
ID_COL = ['id']
TARGET_COL = ["over_50k"]
CATEGORICAL_COL = ['country', 'sex', 'race', 'relationship', 'occupation', 'marital_status', 'education_level', 'workclass']
NUMERIC_COL = ["age", "education_num", "capital_gain", "capital_loss", "hours_week"]
ALL_DATA_COL = NUMERIC_COL + CATEGORICAL_COL

def main():

#### Load Training data
    train = pd.read_csv('Ex1_train.csv')
    test = pd.read_csv('Ex1_test.csv')

#### Calculate Information Gain for each variable for initial reference and first round feature elimination
    dropped_features = []
    features = list(set(ALL_DATA_COL) - set(dropped_features))
    #Prep data to be passed to info gain function
    entropy_data = train[list(features)]
    entropy_data.insert(0, str(TARGET_COL[0]), train[TARGET_COL])
    entropy_data = entropy_data.values

    count = 0
    for feature in entropy_data.T:
        #Calculate parent entropy of target
        if count == 0:
            target_entropy = entropy(feature, 0)
            print("Target entropy:", TARGET_COL[0], target_entropy, "\n")
            count += 1
        #Calculate child entropy and information gain
        else:
            print("natural entropy", features[count-1], entropy(feature, 0))
            print("information gain:", features[count-1], infogain(entropy_data, count, 0), "\n")
            count += 1


#### Model Selection

    # Round 1 Baselines
    #feature set round one: All data variables left in
    dropped_features = []
    features = list(set(ALL_DATA_COL) - set(dropped_features))
    train_x_features = train[list(features)].values
    train_y = train[TARGET_COL].values.ravel()

    #Baseline Random Forest
    clfrf1 = sklearn.ensemble.RandomForestClassifier()
    rfscore1 = sklearn.cross_validation.cross_val_score(clfrf1, train_x_features, train_y, cv=10)


    #Baseline Balance Log Regression
    clflog1 = linear_model.LogisticRegression(class_weight='balanced')
    logscores1 = cross_validation.cross_val_score(clflog1, train_x_features, train_y, cv=10)

    #Baseline K-Nearest Neighbors with default K=5
    clfknn1 = KNeighborsClassifier(n_neighbors=5)
    knnscores1 = cross_validation.cross_val_score(clfknn1, train_x_features, train_y, cv=10)

    #Baseline Decision tree minimizing overfitting with stops on splitting @ 50 and max branch depth set to 8
    clfdt1 = DecisionTreeClassifier(min_samples_split=50, max_depth=10)
    dtscores1 = cross_validation.cross_val_score(clfdt1, train_x_features, train_y, cv=10)



    # Round 2 Features reductions -> Models
    # feature set round two: feature set drops:
    #           relationship -> collinearity with Sex (husband being the most numerous tax filing class)
    #           education_level -> duplicate of education_num and categorical loses value of ordered sequence
    #           country -> too many categories, just noise for other variables (info gain < .025)
    #           race ->  Low information gain + questionable application (info gain < .025)
    #           workclass -> Overlap with occupation which has higher information gain (info gain < .025)
    dropped_features = ["relationship", "education_level", "country", "workclass", "race"]
    features = list(set(ALL_DATA_COL) - set(dropped_features))
    train_x_features = train[list(features)].values
    train_y = train[TARGET_COL].values.ravel()

    #Random Forest
    clfrf2 = sklearn.ensemble.RandomForestClassifier()
    rfscore2 = sklearn.cross_validation.cross_val_score(clfrf2, train_x_features, train_y, cv=10)

    #Balance Log Regression
    clflog2 = linear_model.LogisticRegression(class_weight='balanced')
    logscores2 = cross_validation.cross_val_score(clflog2, train_x_features, train_y, cv=10)

    #K-Nearest Neighbors with default K=5
    clfknn2 = KNeighborsClassifier(n_neighbors=5)
    knnscores2 = cross_validation.cross_val_score(clfknn2, train_x_features, train_y, cv=10)

    #Decision tree minimizing overfitting with stops on splitting @ 50 and max branch depth set to 8
    clfdt2 = DecisionTreeClassifier(min_samples_split=50, max_depth=8)
    dtscores2 = cross_validation.cross_val_score(clfdt2, train_x_features, train_y, cv=10)


    # Evaluate significance of improvements:

    print("\nRnd 1 RF Scores:", rfscore1, str(sum(rfscore1)/10))
    print("Rnd 2 RF Scores:", rfscore2, str(sum(rfscore2)/10))
    print(twotailfisher(rfscore1, rfscore2))


    print("\nRnd 1 Logistic Regression Scores:", logscores1, str(sum(logscores1)/10))
    print("Rnd 2 Logistic Regression Scores:", logscores2, str(sum(logscores2)/10))
    print(twotailfisher(logscores1, logscores2))

    print("\nRnd 1 KNN Scores:", knnscores1, str(sum(knnscores1)/10))
    print("Rnd 2 KNN Scores:", knnscores2, str(sum(knnscores2)/10))
    print(twotailfisher(knnscores1, knnscores2))

    print("\nRnd 1 DT Scores:", dtscores1, str(sum(dtscores1)/10))
    print("Rnd 2 DT Scores:", dtscores2, str(sum(dtscores2)/10))
    print(twotailfisher(dtscores1, dtscores2))

    # Round 3: Narrow Models and Model tuning: Evaluate KNN parameters
    # Random Forest saw a sig reduction in performance
    # Log Regression saw sig improvement, but accuracy well below other models
    # Decision Tree classifier saw sig improvement, but incrementally. Do not want to overfit by dropping depth
    # Keep Reduced features from Round 2

    #Tune K from 5 to 50 by 5 increments generates tunedparameter object
    print("\nRnd 2 KNN Scores @5:", knnscores2, str(sum(knnscores2)/10))
    #Set test parameter to base KNN test @ 5 using tunedparameter object
    k = Tunedparameter(5, sum(knnscores2)/10, 0, knnscores2)
    ktest = k

    for i in range(2,6):
        testklabel = ktest.parameter
        clfknn2atk = KNeighborsClassifier(n_neighbors=(i*5))
        knnscores2atk = cross_validation.cross_val_score(clfknn2atk, train_x_features, train_y, cv=10)
        sig = twotailfisher(ktest.set, knnscores2atk)
        testaccuracy = sum(knnscores2atk)/10
        if testaccuracy > ktest.accuracy and testaccuracy > k.accuracy and sig < .05:
            ktest = Tunedparameter(i*5, testaccuracy, sig, knnscores2atk)

        print("\nRnd 3 KNN Scores @%s:" % str(i*5), knnscores2atk, str(sum(knnscores2atk)/10))
        print("sig @%s" % testklabel, "vs @%s" % str(i*5), sig)

    k = ktest
    print("\nFinal tuned parameter:", k.parameter, k.accuracy, k.sig)

####Train Final Models
    #Train on training data from Round 2 using tuned parameter from Round 3
    #Pickle final models and send out for final tests against test set

    clfknn2 = KNeighborsClassifier(n_neighbors=k.parameter)
    knnTrained = clfknn2.fit(train_x_features, train_y)
    with open("knnmodel.pkl", "wb") as fout:
        pickle.dump(knnTrained, fout, protocol=-1)

####Test Model vs test set

    test_x_features = test[list(features)].values
    test_y = test[TARGET_COL].values.ravel()

#### KNN model test, score and save
    predict = knnTrained.predict(test_x_features)
    print(predict)
    score = knnTrained.score(test_x_features, test_y)
    print(score)

    #send results to csv files -> note: ignore warnings
    predictions_out = test[list(features)]
    predictions_out['over_50kActual'] = test_y
    predictions_out['over_50kPredicted'] = predict
    predictions_out.to_csv("Ex1_KnnResults.csv")


####Functions and Objects

def entropy(data, target_attr):
    # This function takes a pandas dataset and attribute label and returns the natural entropy values of the attribute
    val_freq = {}
    data_entropy = 0.0
    if len(data.shape) == 1:
        for record in data:
            if record in val_freq:
                val_freq[record] += 1.0
            else:
                val_freq[record] = 1.0
    else:
    # Calculate the frequency of each of the values in the target attr
        for record in data:
            if record[target_attr] in val_freq:
                val_freq[record[target_attr]] += 1.0
            else:
                val_freq[record[target_attr]] = 1.0
    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return data_entropy

def infogain(data, child_attr, parent_attr):
    # This function takes a pandas dataset, a parent attribute and a child attribute and returns the information gain
    # (reduction in entropy) with respect to the parent
    val_freq = {}
    subset_entropy = 0.0
    parent_entropy = entropy(data,parent_attr)

    # Calculate the frequency of each of the values in the child attribute
    for record in data.T[child_attr]:
        if record in val_freq:
            val_freq[record] += 1.0
        else:
            val_freq[record] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = data[data[:,child_attr] == val]
        #debug: print("value:(", val, ") value prob:(", val_prob, ") child class entropy:(", entropy(data_subset, parent_attr),")")
        subset_entropy += val_prob * entropy(data_subset, parent_attr)
        #debug: print("subset_entropy:", subset_entropy)

    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return parent_entropy - subset_entropy

def twotailfisher(set1, set2):
    # This function takes two accuracy sets (i.e. a cross validation of k-fold), calculates average performance
    # and returns a two-tailed p value for sig

    test_statistic = abs(sum(set1)/len(set1) - sum(set2)/len(set2))
    n = 100000
    count = 0
    while n > 0:
        test_set1 = set1
        test_set2 = set2
        for i in range(0,len(test_set1)):
            chance = random.uniform(0, 1)
            if chance > .5:
                temp = test_set1[i]
                test_set1[i] = test_set2[i]
                test_set2[i] = temp

        randomized_set1_statistic = sum(test_set1)/len(test_set1)
        randomized_set2_statistic = sum(test_set2)/len(test_set2)
        randomized_test_statistics = abs(randomized_set1_statistic - randomized_set2_statistic)

        if randomized_test_statistics >= test_statistic:
            count += 1
        n -= 1

    sig = count / 100000
    return sig

class Tunedparameter:
    def __init__(self, parameter, accuracy, sig, set):
        self.parameter = parameter
        self.accuracy = accuracy
        self.sig = sig
        self.set = set

main()