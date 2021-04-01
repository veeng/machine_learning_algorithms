#LDA implementation for COMP4900A1
import numpy as np
import pandas as pd
import time

# fit 
# -> used for model training
# X - array containing observational data
# labels - array containing the y (class) values for each sample
def fit(X, labels):
    start_time = time.time()

    y_values = list(np.unique(labels))
    numLabels = len(y_values)

    model = []

    for y in y_values:
        estimate = []

        estimate.append(y)

        # need indices for only rows in X with this specific y value
        rows = np.where(np.isin(labels, y))

        # number of these indices
        numRows = float(len(X[rows]))

        # proportion of the sample with this label (pi)
        labelRatio = float(numRows/len(X)) 
        estimate.append(labelRatio)

        # now find the means of each row in the sample (with this y value)
        mean = (np.sum(X[rows],axis=0) / float(len(X[rows]))).reshape(-1, 1)
        estimate.append(mean)

        variance = (1.0/(numRows - numLabels))

        # top of variance expression
        summation = 0
        for row in X[rows]:
            rowVec = row.reshape(-1, 1) - mean
            summation += rowVec.dot(rowVec.T)

        variance *= summation

        estimate.append(variance)
        model.append(tuple(estimate))

    # now find the final variance
    finalVariance = 0.0
    for estimate in model:
        finalVariance += estimate[3]

    return (model, finalVariance)

# predict
# -> outputs predicted y (class) values for the given data
# X - array containing observational data
# (model, variance) - data built up from fit
def predict(X, model, variance):

    # variance = np.full(X.shape[0],scalar_variance)
    
    bayes = []

    # find the probabilites for each estimate
    for e in model:
        sigInv = np.linalg.inv(variance)
        pi = e[1]
        mean = e[2]

        f = np.linalg.multi_dot([X, sigInv, mean])
        s = 0.5 * np.linalg.multi_dot([mean.T, sigInv, mean])
        
        bayes.append(f - s + np.log(pi)) 

    bayes = np.concatenate(bayes, axis=1)

    maximums = np.argmax(bayes, axis=1)
    
    def predict_class(index):
# the class is in the 0th index of the tuple
        return model[index][0]

	# create a function that does this to a vector
    predict_class_vec = np.vectorize(predict_class)
    predictions = predict_class_vec(maximums)

    return predictions


    # fix this line
    # return np.array(model[maximums][0])


def accuracy_checking(X,label,label_frame):
    start_time = time.time()
    model = fit(X,label)
    train_time = time.time() - start_time
    pred = predict(X, model[0], model[1])
    test_time = time.time() - train_time
    
    pred_df = pd.DataFrame(pred)
    pred_df[0] = pred_df[0].apply(lambda x: 1 if x > 0.4999 or x == 0.4999 else 0)

    final_frame = pd.concat([label_frame,pred_df],axis=1)
    col = ['actual','LDA']
    final_frame.columns = col

    print(final_frame)

    acc = final_frame.loc[final_frame['LDA']==final_frame['actual']].shape[0]/final_frame.shape[0] * 100

    return (acc,train_time,test_time)
    