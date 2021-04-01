

import pandas as pd
import numpy as np 
import sklearn
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import datetime
import os 

import nltk
nltk.download('stopwords')
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#taken from the tutorial, take on the unwanted tags and punctuation
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews] 
    return reviews


#functions that we use to clean data:
stops = stopwords.words('english')
def remove_stop_words(raw_test):
    removed_stop_words = []
    for review in raw_test:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in stops])
        )
    return removed_stop_words

def get_stemmed_text(raw_test):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in raw_test]

def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

#combine them all:
def cleaned_text(raw_test):
    raw_test = remove_stop_words(raw_test)
    # print("remove stop word",raw_test)
    raw_test = get_stemmed_text(raw_test)
    # print("stem",raw_test)
    raw_test = get_lemmatized_text(raw_test)
    # print("lem",raw_test)
    return raw_test

#read the csv file and return the reprocessed reviews
#return (x_train,y_train,x_test,y_test)
#return (bigcoprus,x_train,y_train,x_test,place_holder_id,thresold) #where to split the combo file
def read_file_df(csvtrain,csvtest,useBoth,train_percentage): #useBoth is boolean, true if you dont want to generate train.csv
    train_data = pd.read_csv(csvtrain)
    #shuffle the data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    # train_data = train_data[0:rowlimiTrain]
    df_X_train = train_data['review']
    
    df_label_train = train_data['sentiment']
    df_label_train = df_label_train.apply(lambda x: 1 if x == 'positive' else 0)
    label_train = df_label_train.values

    if (useBoth==False):
        # print(list(df_X_train))
        train_reviews = preprocess_reviews(list(df_X_train))
        train_reviews = cleaned_text(train_reviews)

        total_row = len(df_X_train)
        cut_off_index = int(total_row*train_percentage)

        X_train = train_reviews[0:cut_off_index]
        X_test = train_reviews[cut_off_index:total_row]

        Y_train = label_train[0:cut_off_index]
        Y_test = label_train[cut_off_index:total_row]

        return (X_train,Y_train,X_test,Y_test)

    if (useBoth==True):
        test_data = pd.read_csv(csvtest)
        # test_data = test_data[0:rowlimitTest]
        df_X_test = test_data['review']
        df_id = test_data['id'] #save it to concat after for csv

        train_size = len(df_X_train)
        test_size = len(df_X_test)

        #use to vectorize everything together:
        #concat both reviews together and processed
        #normalize as big corpus
        reviews = list(df_X_train) + list(df_X_test)

        reviews = preprocess_reviews(reviews)
        reviews = cleaned_text(reviews)

        X_train = reviews[0:train_size]
        X_test = reviews[train_size:len(reviews)]

        Y_train = label_train

        return (reviews,X_train,Y_train,X_test,df_id,train_size)


from sklearn.linear_model import LogisticRegression
 #Simple Bag of Word
#choose return csv True if you want to export csv file
def bag_of_word(result,returncsv,classifer,use_corpus,tag): #yes to train with corpus 
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn import model_selection

    cv = CountVectorizer(binary=True)
    if (returncsv==False):
        X_train = result[0]
        X_test = result[2]
        Y_train = result[1]
        Y_test= result[3]

        cv.fit(X_train)
        X_train = cv.transform(list(X_train))
        X_test = cv.transform(list(X_test))

        classifer.fit(X_train,Y_train)
        print("Final Accuracy: %s" 
        % accuracy_score(Y_test, classifer.predict(X_test)))

        #do kfold now:

        kfold = model_selection.KFold(n_splits=10, random_state=100)
        model_kfold = classifer
        results_kfold = model_selection.cross_val_score(model_kfold,X_train,Y_train,cv=kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


#return (bigcoprus,x_train,y_train,x_test,place_holder_id,thresold) #where to split the combo file
    if(returncsv==True):
        corpus = result[0]
        X_train = result[1]
        X_test = result[3]
        Y_train = result[2]

        if (use_corpus==True):
            trainElement = corpus
        else:
            trainElement=X_train

        cv.fit(trainElement)
        X_train = cv.transform(list(X_train))
        X_test = cv.transform(list(X_test))

        classifer.fit(X_train,Y_train)
        kfold = model_selection.KFold(n_splits=10, random_state=100)
        model_kfold = classifer
        results_kfold = model_selection.cross_val_score(model_kfold,X_train,Y_train,cv=kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

        score_string = "_"+str(results_kfold.mean()*100.0)+"_"


        y_pred = classifer.predict(X_test)
        y_pred = list(map(lambda x: 'positive' if x == 1 else 'negative',y_pred))
        y_df = pd.DataFrame(y_pred)
        
        result_df = pd.DataFrame()
        result_df = pd.concat([result[4],y_df],axis=1)
        result_df.columns = ['id','sentiment']


        file_path = os.getcwd() + "/" + tag + score_string + "resultAt_" + datetime.datetime.now().isoformat() +".csv"
        result_df.to_csv(file_path,index=False,header=True)
        

def n_gram(result,returncsv,classifer,range_,use_corpus,tag):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn import model_selection

    cv = CountVectorizer(binary=True,ngram_range=range_)
    if (returncsv==False):
        X_train = result[0]
        X_test = result[2]
        Y_train = result[1]
        Y_test= result[3]

        cv.fit(X_train)
        X_train = cv.transform(list(X_train))
        X_test = cv.transform(list(X_test))

        classifer.fit(X_train,Y_train)
        print("Final Accuracy: %s" 
        % accuracy_score(Y_test, classifer.predict(X_test)))

        #do kfold now:

        kfold = model_selection.KFold(n_splits=10, random_state=100)
        model_kfold = classifer
        results_kfold = model_selection.cross_val_score(model_kfold,X_train,Y_train,cv=kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


#return (bigcoprus,x_train,y_train,x_test,place_holder_id,thresold) #where to split the combo file
    if(returncsv==True):
        corpus = result[0]
        X_train = result[1]
        X_test = result[3]
        Y_train = result[2]

        if (use_corpus==True):
            trainElement = corpus
        else:
            trainElement=X_train

        cv.fit(trainElement)
        X_train = cv.transform(list(X_train))
        X_test = cv.transform(list(X_test))

        classifer.fit(X_train,Y_train)
        kfold = model_selection.KFold(n_splits=10, random_state=100)
        model_kfold = classifer
        results_kfold = model_selection.cross_val_score(model_kfold,X_train,Y_train,cv=kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

        score_string = "_"+str(results_kfold.mean()*100.0)+"_"

        y_pred = classifer.predict(X_test)
        y_pred = list(map(lambda x: 'positive' if x == 1 else 'negative',y_pred))
        y_df = pd.DataFrame(y_pred)
        
        result_df = pd.DataFrame()
        result_df = pd.concat([result[4],y_df],axis=1)
        result_df.columns = ['id','sentiment']


        file_path = os.getcwd() + "/" + tag + score_string + "resultAt_" + datetime.datetime.now().isoformat() +".csv"
        result_df.to_csv(file_path,index=False,header=True)
        


def tf_idf(result,returncsv,classifer,use_corpus,tag):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn import model_selection

    tf_idf = TfidfVectorizer()
    if (returncsv==False):
        X_train = result[0]
        X_test = result[2]
        Y_train = result[1]
        Y_test= result[3]

        tf_idf.fit(X_train)
        X_train = tf_idf.transform(list(X_train))
        X_test = tf_idf.transform(list(X_test))

        classifer.fit(X_train,Y_train)
        print("Final Accuracy: %s" 
        % accuracy_score(Y_test, classifer.predict(X_test)))

        #do kfold now:

        kfold = model_selection.KFold(n_splits=10, random_state=100)
        model_kfold = classifer
        results_kfold = model_selection.cross_val_score(model_kfold,X_train,Y_train,cv=kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


#return (bigcoprus,x_train,y_train,x_test,place_holder_id,thresold) #where to split the combo file
    if(returncsv==True):
        corpus = result[0]
        X_train = result[1]
        X_test = result[3]
        Y_train = result[2]


        if (use_corpus==True):
            trainElement = corpus
        else:
            trainElement=X_train

        tf_idf.fit(trainElement)
        X_train = tf_idf.transform(list(X_train))
        X_test = tf_idf.transform(list(X_test))

        classifer.fit(X_train,Y_train)
        kfold = model_selection.KFold(n_splits=10, random_state=100)
        model_kfold = classifer
        results_kfold = model_selection.cross_val_score(model_kfold,X_train,Y_train,cv=kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

        score_string = "_"+str(results_kfold.mean()*100.0)+"_"
        
        y_pred = classifer.predict(X_test)
        y_pred = list(map(lambda x: 'positive' if x == 1 else 'negative',y_pred))
        y_df = pd.DataFrame(y_pred)
        
        result_df = pd.DataFrame()
        result_df = pd.concat([result[4],y_df],axis=1)
        result_df.columns = ['id','sentiment']


        file_path = os.getcwd() + "/" + tag + score_string + "resultAt_" + datetime.datetime.now().isoformat() +".csv"
        result_df.to_csv(file_path,index=False,header=True)
        





