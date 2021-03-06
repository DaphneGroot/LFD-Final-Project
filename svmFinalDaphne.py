import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.svm import SVC, LinearSVC
import time

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords as sw

from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

import numpy as np
import sys, re, os
import xml.etree.ElementTree as ET
from operator import itemgetter

def main():
    t0 = time.time() #added to see the total duration of the program

    try:
        trainDirectory = str(sys.argv[1])
        testDirectory  = str(sys.argv[2])
    except:
        print("Please use python3 svm.py trainDirectory testDirectory (e.g. training/english testing/english)")
        sys.exit(1)

    try:
        goldPath = str(sys.argv[3])
    except:
        print("Please define the path to the gold-standard file (e.g. english/gold.txt), so accuracy, precision and recall over test-set can be calculated.")
        sys.exit(1)

    trainDocuments, testDocuments = createDocuments(trainDirectory,testDirectory)

    # Preprocess the data (e.g. tokenisation)
    trainDocuments = preprocessData(trainDocuments)
    testDocuments  = preprocessData(testDocuments)


    #create seperate lists for tweets and the genders
    train_ids, train_tweets, train_genders, train_ages = createLists(trainDocuments,"train")
    test_ids, test_tweets = createLists(testDocuments,"test")

    #predict gender
    classifierGender  = classifyGender(train_tweets, train_genders)
    predicted_genders = classifierGender.predict(test_tweets)

    language = testDirectory.split("/")[-1]

    #only predict age for spanish and english
    if "spanish" in language or "english" in language:
        classifierAge  = classifyAge(train_tweets, train_ages)
        predicted_ages = classifierAge.predict(test_tweets)

    else:
        predicted_ages = ["XX-XX" for i in range(len(test_ids))]



    #write predictions to truth file
    outFile = open(testDirectory+"/truthPredicted.txt","w+")
    
    userDictGender = {}
    userDictAge    = {}
    for idx, i in enumerate(test_ids):
        idn     = i
        gender  = predicted_genders[idx]
        age     = predicted_ages[idx]

        if idn not in userDictGender:
            userDictGender[idn] = [gender]
            userDictAge[idn]    = [age]
        else:
            userDictGender[idn].append(gender)
            userDictAge[idn].append(age)

    for key in userDictGender:
        gender = Counter(userDictGender[key]).most_common(1)[0][0]
        age    = Counter(userDictAge[key]).most_common(1)[0][0]

        outFile.write(key+":::"+gender+":::"+age+"\n")

    outFile.close()


    # Calculate metrics
    goldFile      = open(goldPath,"r+").read().split("\n")
    predictedFile = open(testDirectory+"/truthPredicted.txt","r+").read().split("\n")

    goldFile = [line.split(":::")[:3] for line in goldFile if line]
    goldFile = sorted(goldFile, key=itemgetter(0))
    
    predictedFile = [line.split(":::") for line in predictedFile if line]
    predictedFile = sorted(predictedFile, key=itemgetter(0))


    goldGenders      = [i[1] for i in goldFile]
    goldAges         = [i[2] for i in goldFile]
    predictedGenders = [i[1] for i in predictedFile]
    predictedAges    = [i[2] for i in predictedFile]

    goldCombined      = goldGenders + goldAges
    predictedCombined = predictedGenders + predictedAges

    #Gender
    results(goldGenders,predictedGenders,"gender",language)


    if "spanish" in language or "english" in language:
        #Age
        print("age")
        results(goldAges,predictedAges,"age",language)

        #Gender+Age
        results(goldCombined,predictedCombined,"combined",language)

    total_time = time.time() - t0
    print("\ntotal time: ", total_time)
    


def results(gold,predicted,part,language):
    accuracy          = accuracy_score(gold, predicted)
    precision         = precision_score(gold,predicted,average="macro")
    recall            = recall_score(gold,predicted,average="macro")
    f1                = f1_score(gold,predicted,average="macro")
    confusionMatrix   = sklearn.metrics.confusion_matrix(gold, predicted)

    if part == "gender":
        print("\n\n"+'\033[95m'+"Gender"+'\033[0m'+":\nAccuracy = ", round(accuracy,3),"\n")

        labels = ["F","M"]

    elif part == "age":
        print("\n\n"+'\033[95m'+"Age"+'\033[0m'+":\nAccuracy = ", round(accuracy,3),"\n")
        labels = ["18-24","25-34","35-49","50-XX"]

    else:
        print("\n\n"+'\033[95m'+"Combined"+'\033[0m'+":\nAccuracy = ", round(accuracy,3),"\n")
        labels = ["18-24","25-34","35-49","50-XX","F","M"]



    print("\t\t {} \t {} \t {}".format("Precision","Recall","F1-score"))
    for label in labels:
        precisionScore  = sklearn.metrics.precision_score(gold,predicted, average="micro", labels=label)
        recallScore     = sklearn.metrics.recall_score(gold,predicted, average="micro", labels=label)
        f1Score         = sklearn.metrics.f1_score(gold,predicted, average="micro", labels=label)

        print("{} \t\t {} \t\t {} \t\t {}".format(label,round(precisionScore,3),round(recallScore,3),round(f1Score,3)))

    print("\nAvg/total \t {} \t\t {} \t\t {}".format(round(precision,3),round(recall,3),round(f1,3)))

    print("\nConfusion Matrix:")
    createConfusionMatrix(confusionMatrix, part, language)



    # return accuracy, precision, recall, f1, confusionMatrix

    
def identity(x):
    return x


def tweetIdentity(arg):
    tokenizer   = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens      = tokenizer.tokenize(arg)
    return tokens


def customLemmatizer(arg):
    """
    Preprocesser function to test different lemma.
    """
    wnl = WordNetLemmatizer()
    st  = LancasterStemmer()
    return st.stem(wnl.lemmatize(arg))



def classifyGender(train_tweets, train_genders):

    vec_word = TfidfVectorizer(preprocessor = customLemmatizer,
                         tokenizer = tweetIdentity,
                         binary=True,
                         lowercase=False, 
                         analyzer='word', 
                         ngram_range=(1,2))

    vec_char = TfidfVectorizer(preprocessor = customLemmatizer,
                         tokenizer = tweetIdentity,
                         binary=True,
                         lowercase=False, 
                         analyzer='char', 
                         ngram_range=(3,5))


    combined_feats = FeatureUnion([("vec_word", vec_word), ("vec_char", vec_char)])



    classifier = Pipeline([('vec', combined_feats),
                            ('classifier', LinearSVC(multi_class='crammer_singer'))])

    
    classifier.fit(train_tweets, train_genders)  
    return classifier


def classifyAge(train_tweets, train_ages):
   
    vec_word = TfidfVectorizer(preprocessor = customLemmatizer,
                         tokenizer = tweetIdentity,
                         binary=True,
                         lowercase=False, 
                         analyzer='word', 
                         ngram_range=(1,2),
                         max_features=400000)

    vec_char = TfidfVectorizer(preprocessor = customLemmatizer,
                         tokenizer = tweetIdentity,
                         binary=True,
                         lowercase=False, 
                         analyzer='char', 
                         ngram_range=(3,5),
                         max_features=400000)

    combined_feats = FeatureUnion([("vec_word", vec_word), ("vec_char", vec_char)])


    classifier = Pipeline([('vec', combined_feats),
                            ('classifier', LinearSVC())])


    classifier.fit(train_tweets, train_ages)  
    return classifier


def createLists(documents,part):
    if part == "train":
        ids     = []
        tweets  = []
        genders = []
        ages    = []

        for i in documents:
            if i != "":
                idn     = i[0]
                tweet   = i[1]
                gender  = i[2]
                age     = i[3]

                ids.append(idn)
                genders.append(gender)
                tweets.append(tweet)
                ages.append(age)

        return ids, tweets, genders, ages
    else:
        ids     = []
        tweets  = []

        for i in documents:
            if i != "":
                idn     = i[0]
                tweet   = i[1]

                ids.append(idn)
                tweets.append(tweet)

        return ids, tweets
    

def createDocuments(trainDirectory,testDirectory):
    docWithAllTraining = []

    for f in os.listdir(trainDirectory):
        if f != "truth.txt": #ignore the (possible) txt file at the end
            document = open(trainDirectory+'/'+f,"r+").read() 

            goldDocument = open(trainDirectory+'/truth.txt',"r+").read().split("\n")

            # print(document)
            idName  = f[:-4] #filename - .xml
            tree    = ET.parse(trainDirectory+"/"+f)
            root    = tree.getroot()

            #find gold values
            for line in goldDocument:
                if line != "":
                    line     = line.strip().split(":::")
                    idGold   = line[0].strip()
                    gender   = line[1].strip()
                    ageGroup = line[2].strip()

                    if idGold == idName:
                        for child in root:
                            tweet = child.text.strip()
                            # print(tweet)

                            docWithAllTraining.append([idName,tweet,gender,ageGroup])

    docWithAllTesting = []
    for f in os.listdir(testDirectory):
        if f != "truth.txt": #ignore the (possible) txt file at the end
            document = open(testDirectory+'/'+f,"r+").read() 

            # print(document)
            idName = f[:-4] #filename - .xml
            tree   = ET.parse(testDirectory+"/"+f)
            root   = tree.getroot()

            for child in root:
                tweet = child.text.strip()
                # print(tweet)

                docWithAllTesting.append([idName,tweet])


    return docWithAllTraining, docWithAllTesting


def url_to_placeholder(tweet):
    tweet= re.sub(r"http\S+", "url", str(tweet))
    return tweet

def number_to_placeholder(tweet):
    tweet = re.sub(r'[0-9]+', "num", str(tweet))
    return tweet

def tokenize_tweet(tweet):
    tokenizer   = TweetTokenizer(reduce_len=True)
    tokens      = tokenizer.tokenize(tweet)
    return " ".join(tokens)

def clean_tweets(tweet):
    tweet         = tokenize_tweet(tweet)
    cleaned_tweet = url_to_placeholder(number_to_placeholder(tweet))
    return cleaned_tweet
                    
def preprocessData(documents):
    for idx, i in enumerate(documents):
        tweet = i[1]
        tweet = clean_tweets(tweet)

        documents[idx][1] = tweet

    return(documents)

def createConfusionMatrix(confusionMatrix,part,language):
    print()
    if part == "gender":
        labels = ['F','M']
    elif part == "age":
        labels = ['18-24','25-34','35-49','50-XX']
    else:
        labels = ['18-24','25-34','35-49','50-XX','F','M']


    print("{:10s}".format(""), end="")
    [print("{:<8s}".format(item), end="") for item in labels]
    print()

    for idx, elem in enumerate(confusionMatrix):
        print("{:10s}".format(labels[idx]), end="")
        [print("{:<8d}".format(el), end="") for el in elem]
        print()

    
if __name__ == '__main__':
    main()
