'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn import tree
from sklearn.metrics import accuracy_score

no_of_folds = 10

def evaluatePerformance(numTrials = 100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    split_basic_dt = np.empty([numTrials,numTrials/10,numTrials/10])
    split_level_1_dt = np.empty([numTrials,numTrials/10,numTrials/10])
    split_level_3_dt = np.empty([numTrials,numTrials/10,numTrials/10])

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    for i in xrange(numTrials):
        # shuffle the data
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
		# split the data
		#Xtrain = X[1:101,:]  # train on first 100 instances
		#Xtest = X[101:,:]
		#ytrain = y[1:101,:]  # test on remaining instances
		#ytest = y[101:,:]

        for j in xrange(no_of_folds):
            portion = int(np.ceil([n * 0.1]))
            iBegin = j * portion
            Xtest = X[iBegin: iBegin + portion, :]
            ytest = y[iBegin: iBegin + portion, :]
            Xtrain = np.delete(X, np.s_[iBegin: iBegin + portion], 0)
            ytrain = np.delete(y, np.s_[iBegin: iBegin + portion], 0)
            
            for k in xrange(no_of_folds):
                l = k+1
                split = l * 10
                split_ratio = n * (split/100.0)
                Xtrain_ratio = Xtrain[:split_ratio, :]
                ytrain_ratio = ytrain[:split_ratio, :]
            
			    # train the decision tree
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(Xtrain_ratio, ytrain_ratio)
	   
		    	# train the level 1 decision tree
                clf_level_1 = tree.DecisionTreeClassifier(max_depth = 1)
                clf_level_1 = clf_level_1.fit(Xtrain_ratio, ytrain_ratio)
	
		    	# train the level 3 decision tree
                clf_level_3 = tree.DecisionTreeClassifier(max_depth = 3)
                clf_level_3 = clf_level_3.fit(Xtrain_ratio, ytrain_ratio)
	    
		    	# output predictions on the remaining data
                y_pred = clf.predict(Xtest)
                y_pred_level_1 = clf_level_1.predict(Xtest)
                y_pred_level_3 = clf_level_3.predict(Xtest)

		    	# compute the training accuracy of the model
	    		#meanDecisionTreeAccuracy = accuracy_score(ytest, y_pred)
                split_basic_dt[i, j, k] = accuracy_score(ytest, y_pred)
                split_level_1_dt[i, j, k] = accuracy_score(ytest, y_pred_level_1)
                split_level_3_dt[i, j, k] = accuracy_score(ytest, y_pred_level_3)
    
    # TODO: update these statistics based on the results of your experiment
    meanDecisionTreeAccuracy = np.mean(split_basic_dt[:,:,9])
    stddevDecisionTreeAccuracy = np.std(split_basic_dt[:,:,9])
    meanDecisionStumpAccuracy = np.mean(split_level_1_dt[:,:,9])
    stddevDecisionStumpAccuracy = np.std(split_level_1_dt[:,:,9])
    meanDT3Accuracy = np.mean(split_level_3_dt[:,:,9])
    stddevDT3Accuracy = np.std(split_level_3_dt[:,:,9])

    mean_basic_dt = np.empty([numTrials/10])
    mean_level_1_dt = np.empty([numTrials/10])
    mean_level_3_dt = np.empty([numTrials/10])
    std_basic_dt = np.empty([numTrials/10])
    std_level_1_dt = np.empty([numTrials/10])
    std_level_3_dt = np.empty([numTrials/10])
    test_portion = np.empty([numTrials/10])
    
    for m in xrange(no_of_folds):
        test_portion[m] = (m+1) * 10
        #try:
        mean_basic_dt[m] = np.mean(split_basic_dt[:,:,m])
        #except:
        #    pdb.set_trace()
        #    print "array " + mean_basic_dt + "index " + m
        std_basic_dt[m] = np.std(split_basic_dt[:,:,m])
        mean_level_1_dt[m] = np.mean(split_level_1_dt[:,:,m])
        std_level_1_dt[m] = np.std(split_level_1_dt[:,:,m])
        mean_level_3_dt[m] = np.mean(split_level_3_dt[:,:,m])
        std_level_3_dt[m] = np.std(split_level_3_dt[:,:,m])
        
    plt.title("Learning Curve")
    plt.xlabel("Percentage of Training Data")
    plt.ylabel("mean/std")
    
    plt.errorbar(test_portion[:], mean_basic_dt[:], std_basic_dt[:], ecolor = 'b', label = 'Basic DT')
    plt.errorbar(test_portion[:], mean_level_1_dt[:], std_level_1_dt[:], ecolor = 'g', label = 'Level 1 DT')
    plt.errorbar(test_portion[:], mean_level_3_dt[:], std_level_3_dt[:], ecolor = 'r', label = 'Level 3 DT')
    
    plt.legend()
    #plt.show()
    plt.savefig("LearningCurve.pdf")
    
    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
