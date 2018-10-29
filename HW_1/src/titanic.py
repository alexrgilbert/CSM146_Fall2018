"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
import numpy as np
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        total_targets = y.shape[0]
        probs  = Counter(y).most_common(2)
        probs[0] = (probs[0][0],float(probs[0][1])/total_targets)
        probs[1] = (probs[1][0],float(probs[1][1])/total_targets)
        self.probabilities_ = probs
        #majority = np.divide(self.probabilities_[:][1],total_targets)
        #self.probabilities_ = [[1.0,0.0],[0.0,0.0]]
        #for example_lable in y:
        #    if(example_lable == self.probabilities_[0][0]):
        #        self.probabilities_[0][1] = self.probabilities_[0][1]+1.0
        #    if(example_lable == self.probabilities_[1][0]):
        #        self.probabilities_[0][1] = self.probabilities_[0][1]+1.0
        #self.probabilities_[:][0] = self.probabilities_[:][1]/total_targets

        #print ("MaxTarget_Label = {} MaxTarget_Val = {}".format(self.probabilities_[0][0],self.probabilities_[0][1]))
        #print ("MinTarget_Label = {} MinTarget_Val = {}".format(self.probabilities_[1][0],self.probabilities_[1][1]))
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        labels = ((self.probabilities_[0][0]),(self.probabilities_[1][0]))
        probs = (self.probabilities_[0][1],self.probabilities_[1][1])
        #print(labels)
        #print(probs)

        n,d = X.shape
        #y = [self.prediction_] * n
        y = np.zeros(n,)
        for prediction in range(n):
            y[prediction] = np.random.choice(labels,p = probs)
        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels

def error(clf, X, y, ntrials=100, test_size=0.2, train_size=1.0):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    avg_trainError = 0; avg_testError = 0;
    for trial in range(ntrials):
        X_train_whole, X_test, y_train_whole, y_test = train_test_split(X, y, test_size=0.2, random_state=trial)
        if (train_size == 1.0):
            X_train_part = X_train_whole
            y_train_part = y_train_whole
        else:
            X_train_part, X_train_rest, y_train_part, y_train_rest = train_test_split(X_train_whole, y_train_whole, train_size=train_size, random_state=trial)
        clf.fit(X_train_part, y_train_part)
        y_pred_train = clf.predict(X_train_part)
        avg_trainError = avg_trainError + ((1 - metrics.accuracy_score(y_train_part, y_pred_train, normalize=True))/ntrials)
        y_pred_test = clf.predict(X_test)
        avg_testError = avg_testError + ((1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True))/ntrials)
        train_error = avg_trainError
        test_error = avg_testError

    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    #print('Plotting...')
    #for i in range(d) :
    #    plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    # print('Classifying using Majority Vote...')
    # clfMV = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    # clfMV.fit(X, y)                  # fit training data using the classifier
    # y_predMV = clfMV.predict(X)        # take the classifier and run it on the training data
    # train_errorMV = 1 - metrics.accuracy_score(y, y_predMV, normalize=True)
    # print('\t-- training error: %.3f' % train_errorMV)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    # print('Classifying using Random...')
    # clfRand = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    # clfRand.fit(X, y)                  # fit training data using the classifier
    # y_predRand = clfRand.predict(X)        # take the classifier and run it on the training data
    # train_errorRand = 1 - metrics.accuracy_score(y, y_predRand, normalize=True)
    # print('\t-- training error: %.3f' % train_errorRand)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    # print('Classifying using Decision Tree...')
    # clfDT = DecisionTreeClassifier(criterion='entropy')
    # clfDT.fit(X, y)
    # y_predDT = clfDT.predict(X)        # take the classifier and run it on the training data
    # train_errorDT = 1 - metrics.accuracy_score(y, y_predDT, normalize=True)
    # print('\t-- training error: %.3f' % train_errorDT)


    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    # print('Classifying using 3 k-Nearest Neighbors...')
    # clfKNN3 = KNeighborsClassifier(n_neighbors=3)
    # clfKNN3.fit(X, y)
    # y_predKNN3 = clfKNN3.predict(X)        # take the classifier and run it on the training data
    # train_errorKNN3 = 1 - metrics.accuracy_score(y, y_predKNN3, normalize=True)
    # print('\t-- training error: %.3f' % train_errorKNN3)
    #
    # print('Classifying using 5 k-Nearest Neighbors...')
    # clfKNN5 = KNeighborsClassifier(n_neighbors=5)
    # clfKNN5.fit(X, y)
    # y_predKNN5 = clfKNN5.predict(X)        # take the classifier and run it on the training data
    # train_errorKNN5 = 1 - metrics.accuracy_score(y, y_predKNN5, normalize=True)
    # print('\t-- training error: %.3f' % train_errorKNN5)
    #
    # print('Classifying using 7 k-Nearest Neighbors...')
    # clfKNN7 = KNeighborsClassifier(n_neighbors=7)
    # clfKNN7.fit(X, y)
    # y_predKNN7 = clfKNN7.predict(X)        # take the classifier and run it on the training data
    # train_errorKNN7 = 1 - metrics.accuracy_score(y, y_predKNN7, normalize=True)
    # print('\t-- training error: %.3f' % train_errorKNN7)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    # print('Investigating various classifiers...')
    #
    # clfMV_TTSplit = MajorityVoteClassifier()
    # (avg_trainError_MV,avg_testError_MV) = error(clfMV_TTSplit, X, y, ntrials=100, test_size=0.2)
    # print('MajorityVote: -- training error: %.3f\t-- testing error: %.3f' % (avg_trainError_MV, avg_testError_MV))
    #
    # clfRand_TTSplit = RandomClassifier()
    # (avg_trainError_Rand,avg_testError_Rand) = error(clfRand_TTSplit, X, y, ntrials=100, test_size=0.2)
    # print('Random: -- training error: %.3f\t-- testing error: %.3f' % (avg_trainError_Rand, avg_testError_Rand))
    #
    # clfDT_TTSplit = DecisionTreeClassifier(criterion='entropy')
    # (avg_trainError_DT,avg_testError_DT) = error(clfDT_TTSplit, X, y, ntrials=100, test_size=0.2)
    # print('DecisionTree: -- training error: %.3f\t-- testing error: %.3f' % (avg_trainError_DT, avg_testError_DT))
    #
    # clfKNN5_TTSplit = KNeighborsClassifier(n_neighbors=5)
    # (avg_trainError_KNN5,avg_testError_KNN5) = error(clfKNN5_TTSplit, X, y, ntrials=100, test_size=0.2)
    # print('K-Nearest: -- training error: %.3f\t-- testing error: %.3f' % (avg_trainError_KNN5, avg_testError_KNN5))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    nfold = 100
    max_k = 50
    k_tests = []
    avg_k_validationErrors = []
    for k in range(1,50,2):
        clfKNNk = KNeighborsClassifier(n_neighbors=k)
        k_tests.append(k)
        avg_k_validationErrors.append(1-(np.sum(cross_val_score(clfKNNk,X,y,scoring='accuracy',cv=nfold))/nfold))
    # fig = plt.figure(figsize=(20,15))
    # plt.plot(k_tests,avg_k_validationErrors,color='blue', marker='o', linestyle='dashed', label="KNN Validation Error vs K")
    # plt.grid(linewidth=.5)
    # plt.xticks(k_tests,k_tests)
    # plt.xlabel("# of Nearest Neighbors")
    # plt.ylabel("Validation Error") #plt.legend(loc='upper left')
    # plt.legend()
    # plt.show()
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')

    nfold = 100
    max_depth = 20
    depth_tests = []
    avg_depth_validationErrors = []
    for d in range(1,(max_depth+1)):
        clfDTd = DecisionTreeClassifier(criterion='entropy',max_depth=d)
        depth_tests.append(d)
        avg_depth_validationErrors.append(1-(np.sum(cross_val_score(clfDTd,X,y,scoring='accuracy',cv=nfold))/nfold))
    # fig = plt.figure(figsize=(20,15))
    # plt.plot(depth_tests,avg_depth_validationErrors,color='blue', marker='o', linestyle='dashed', label="DT Validation Error vs Depth Limit")
    # plt.grid(linewidth=.5)
    # plt.xticks(depth_tests,depth_tests)
    # plt.xlabel("Depth Limit")
    # plt.ylabel("Validation Error")
    # plt.legend()#plt.legend(loc='upper left')
    # plt.show()
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')


    best_depth = depth_tests[np.argmin(avg_depth_validationErrors)]#[best_depth_idx]#
    best_k = k_tests[np.argmin(avg_k_validationErrors)]#[best_k_idx]#

    train_fractions = []
    avg_DT_TTSplit_trainErrors = []
    avg_DT_TTSplit_testErrors = []
    avg_KNNbest_TTSplit_trainErrors = []
    avg_KNNbest_TTSplit_testErrors = []
    for train_fraction_int in range(1,10,1):

        train_fraction = float(train_fraction_int)/10
        train_fractions.append(train_fraction)

        clfDT_TTSplit = DecisionTreeClassifier(criterion='entropy',max_depth=best_depth)
        (avg_DT_TTSplit_trainError,avg_DT_TTSplit_testError) = error(clfDT_TTSplit, X, y, ntrials=100, test_size=0.1,train_size=train_fraction)
        avg_DT_TTSplit_trainErrors.append(avg_DT_TTSplit_trainError)
        avg_DT_TTSplit_testErrors.append(avg_DT_TTSplit_testError)


        clfKNNbest_TTSplit = KNeighborsClassifier(n_neighbors=best_k)
        (avg_KNNbest_TTSplit_trainError,avg_KNNbest_TTSplit_testError) = error(clfKNNbest_TTSplit, X, y, ntrials=100, test_size=0.1,train_size=train_fraction)
        avg_KNNbest_TTSplit_trainErrors.append(avg_KNNbest_TTSplit_trainError)
        avg_KNNbest_TTSplit_testErrors.append(avg_KNNbest_TTSplit_testError)
    fig = plt.figure(figsize=(20,15))
    DT_Train_Label = ("Depth Tree (max depth = %d) Training Error" %best_depth)
    DT_Test_Label = ("Depth Tree (max depth = %d) Test Error" %best_depth)
    KNN_Train_Label = ("KNN (k = %d) Training Error" %best_k)
    KNN_Test_Label = ("Knn (k = %d) Test Error" %best_k)
    plt.plot(train_fractions,avg_DT_TTSplit_trainErrors,color='blue', marker='o', linestyle='dashed', label=DT_Train_Label)
    plt.plot(train_fractions,avg_DT_TTSplit_testErrors,color='green', marker='.', linestyle='dashed', label=DT_Test_Label)
    plt.plot(train_fractions,avg_KNNbest_TTSplit_trainErrors,color='red', marker='x', linestyle='dashed', label=KNN_Train_Label)
    plt.plot(train_fractions,avg_KNNbest_TTSplit_testErrors,color='cyan', marker='+', linestyle='dashed', label=KNN_Test_Label)
    plt.grid(linewidth=.5)
    plt.xticks(train_fractions,train_fractions)
    plt.xlabel("Fraction of Training Data Used")
    plt.ylabel("Training/Test Error")
    plt.legend()#plt.legend(loc='upper left')
    plt.show()
    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
