import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
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
from sklearn.model_selection import KFold

import numpy as np
import sys, re, os
import xml.etree.ElementTree as ET
from operator import itemgetter

def main():
    t0 = time.time() #added to see the total duration of the program

    ## For developing
    language = str(sys.argv[1])
    files = os.listdir('training/'+language)
    files = [f for f in files if f[-4:] == ".xml"] #files in directory, minus all non .xml-files

    # trainDocuments, testDocuments = createDocuments(files[:trainingSize],files[trainingSize:],language)


    kf = KFold(n_splits=7)
    sumAccGender = 0
    sumAccAge = 0
    sumAccCombined = 0

    sumPrecisionGender = 0
    sumPrecisionAge = 0
    sumPrecisionCombined = 0

    sumRecallGender = 0
    sumRecallAge = 0
    sumRecallCombined = 0

    sumF1Gender = 0
    sumF1Age = 0
    sumF1Combined = 0

    for train, test in kf.split(files):
        trainData = np.array(files)[train]
        testData = np.array(files)[test]

        trainDocuments, testDocuments = createDocuments(trainData,testData,language)

        # Preprocess the data (e.g. tokenisation)
        trainDocuments = preprocessData(trainDocuments)
        testDocuments = preprocessData(testDocuments)


        #create seperate lists for tweets and the genders
        train_ids, train_tweets, train_genders, train_ages = createLists(trainDocuments,"train")
        test_ids, test_tweets = createLists(testDocuments,"test")


        #predict gender
        mostFrequentGender = Counter(train_genders).most_common(1)[0][0]
        # print("Most frequent gender = ", mostFrequentGender)

        #only predict age 
        mostFrequentAge = Counter(train_ages).most_common(1)[0][0]
        # print("Most frequent age = ", mostFrequentAge)

        goldFile = open('training/'+language+'/truth.txt',"r+").read().split("\n")
        
        goldFile = [line.split(":::")[:3] for line in goldFile if line]
        # print(goldFile)
        # for i in goldFile:
        #     print(i[0], i[0]+".xml")
        goldFile = [i for i in goldFile if i[0]+".xml" in testData]
        goldFile = sorted(goldFile, key=itemgetter(0))
        
        # print(goldFile)

        goldGenders = [i[1] for i in goldFile]
        goldAges = [i[2] for i in goldFile]
        predictedGenders = [mostFrequentGender for i in set(test_ids)]
        predictedAges = [mostFrequentAge for i in set(test_ids)]
        goldCombined = goldGenders + goldAges
        predictedCombined = predictedGenders + predictedAges

        accuracyGender = accuracy_score(goldGenders, predictedGenders)
        accuracyAge = accuracy_score(goldAges, predictedAges)
        accuracyCombined = accuracy_score(goldCombined, predictedCombined)

        precisionGender = precision_score(goldGenders, predictedGenders, average="macro")
        precisionAge = precision_score(goldAges, predictedAges, average="macro")
        precisionCombined = precision_score(goldCombined, predictedCombined, average="macro")

        recallGender = recall_score(goldGenders, predictedGenders, average="macro")
        recallAge = recall_score(goldAges, predictedAges, average="macro")
        recallCombined = recall_score(goldCombined, predictedCombined, average="macro")

        f1Gender = f1_score(goldGenders, predictedGenders, average="macro")
        f1Age = f1_score(goldAges, predictedAges, average="macro")
        f1Combined = f1_score(goldCombined, predictedCombined, average="macro")

        # classifier = nltk.NaiveBayesClassifier.train(train_tweets)
        sumAccGender += accuracyGender
        sumAccAge += accuracyAge
        sumAccCombined += accuracyCombined

        sumPrecisionGender += precisionGender
        sumPrecisionAge += precisionAge
        sumPrecisionCombined += precisionCombined

        sumRecallGender += recallGender
        sumRecallAge += recallAge
        sumRecallCombined += recallCombined

        sumF1Gender += f1Gender
        sumF1Age += f1Age
        sumF1Combined += f1Combined

    averageAccGender = sumAccGender/5
    averageAccAge = sumAccAge/5
    averageAccCombined = sumAccCombined/5

    averagePrecisionGender = sumPrecisionGender/5
    averagePrecisionAge = sumPrecisionAge/5
    averagePrecisionCombined = sumPrecisionCombined/5

    averageRecallGender = sumRecallGender/5
    averageRecallAge = sumRecallAge/5
    averageRecallCombined = sumRecallCombined/5

    averagef1Gender = sumF1Gender/5
    averagef1Age = sumF1Age/5
    averagef1Combined = sumF1Combined/5

    print("Avg {:<10s}: acc = {:>.3f} precision = {:>.3f} recall = {:>.3f} f1 = {:>.3f}".format("Gender",round(averageAccGender,3),round(averagePrecisionGender,3),round(averageRecallGender,3),round(averagef1Gender,3)))
    print("Avg {:<10s}: acc = {:>.3f} precision = {:>.3f} recall = {:>.3f} f1 = {:>.3f}".format("Age",round(averageAccAge,3),round(averagePrecisionAge,3),round(averageRecallAge,3),round(averagef1Age,3)))
    print("Avg {:<10s}: acc = {:>.3f} precision = {:>.3f} recall = {:>.3f} f1 = {:>.3f}".format("Combined",round(averageAccCombined,3),round(averagePrecisionCombined,3),round(averageRecallCombined,3),round(averagef1Combined,3)))


    total_time = time.time() - t0
    print("\ntotal time: ", total_time)
    

def createLists(documents,part):
    if part == "train":
        ids = []
        tweets = []
        genders = []
        ages = []

        for i in documents:
            if i != "":
                idn = i[0]
                tweet = i[1]
                gender = i[2]
                age = i[3]

                ids.append(idn)
                genders.append(gender)
                tweets.append(tweet)
                ages.append(age)

        return ids, tweets, genders, ages
    else:
        ids = []
        tweets = []

        for i in documents:
            if i != "":
                idn = i[0]
                tweet = i[1]

                ids.append(idn)
                tweets.append(tweet)

        return ids, tweets
    

def createDocuments(trainDirectory,testDirectory,language):
    docWithAllTraining = []

    for f in trainDirectory:
        if f != "truth.txt": #ignore the (possible) txt file at the end
            document = open('training/'+language+'/'+f,"r+").read() 

            goldDocument = open('training/'+language+'/truth.txt',"r+").read().split("\n")

            # print(document)
            idName = f[:-4] #filename - .xml
            tree = ET.parse('training/'+language+'/'+f)
            root = tree.getroot()

            #find gold values
            for line in goldDocument:
                if line != "":
                    line = line.strip().split(":::")
                    idGold = line[0].strip()
                    gender = line[1].strip()
                    ageGroup = line[2].strip()

                    if idGold == idName:
                        for child in root:
                            tweet = child.text.strip()
                            # print(tweet)

                            docWithAllTraining.append([idName,tweet,gender,ageGroup])

    docWithAllTesting = []
    for f in testDirectory:
        if f != "truth.txt": #ignore the (possible) txt file at the end
            document = open('training/'+language+'/'+f,"r+").read()

            # print(document)
            idName = f[:-4] #filename - .xml
            tree = ET.parse('training/'+language+'/'+f)
            root = tree.getroot()

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
    tokenizer = TweetTokenizer(reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    return " ".join(tokens)

def clean_tweets(tweet):
    tweet = tokenize_tweet(tweet)
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
        if language == "spanish" or language =="english":
            labels = ['18-24','25-34','35-49','50-XX']
        else:
            labels = ['XX-XX']
    else:
        if language == "spanish" or language =="english":
            labels = ['18-24','25-34','35-49','50-XX','F','M']
        else:
            labels = ['XX-XX','F','M']


    print("{:10s}".format(""), end="")
    [print("{:<8s}".format(item), end="") for item in labels]
    print()

    for idx, elem in enumerate(confusionMatrix):
        print("{:10s}".format(labels[idx]), end="")
        [print("{:<8d}".format(el), end="") for el in elem]
        print()

    
if __name__ == '__main__':
    main()
