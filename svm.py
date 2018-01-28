import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        testDirectory = str(sys.argv[2])
    except:
        print("Please use python3 svm.py trainDirectory testDirectory (e.g. training/english testing/english)")

    try:
        goldPath = str(sys.argv[3])
    except:
        print("Please define the path to the gold-standard file (e.g. english/gold.txt), so accuracy, precision and recall over test-set can be calculated.")

    trainDocuments, testDocuments = createDocuments(trainDirectory,testDirectory)

    # Preprocess the data (e.g. tokenisation)
    trainDocuments = preprocessData(trainDocuments)
    testDocuments = preprocessData(testDocuments)


    #create seperate lists for tweets and the genders
    train_ids, train_tweets, train_genders, train_ages = createLists(trainDocuments,"train")
    test_ids, test_tweets = createLists(testDocuments,"test")

    #predict gender
    classifierGender = classifyGender(train_tweets, train_genders)
    predicted_genders = classifierGender.predict(test_tweets)

    #only predict age for spanish and english
    if testDirectory.split("/")[1] == "spanish" or testDirectory.split("/")[1] == "english":
        #predict age
        classifierAge = classifyAge(train_tweets, train_ages)
        predicted_ages = classifierAge.predict(test_tweets)
    else:
        predicted_ages = ["XX-XX" for i in range(len(test_ids))]



    #write predictions to truth file
    outFile = open(testDirectory+"/truth.txt","w+")
    
    userDictGender = {}
    userDictAge = {}
    for idx, i in enumerate(test_ids):
        idn = i
        gender = predicted_genders[idx]
        age = predicted_ages[idx]

        if idn not in userDictGender:
            userDictGender[idn] = [gender]
            userDictAge[idn] = [age]
        else:
            userDictGender[idn].append(gender)
            userDictAge[idn].append(age)

    for key in userDictGender:
        gender = Counter(userDictGender[key]).most_common(1)[0][0]
        age = Counter(userDictAge[key]).most_common(1)[0][0]

        outFile.write(key+":::"+gender+":::"+age+"\n")

    outFile.close()


    # Calculate metrics
    goldFile = open(goldPath,"r+").read().split("\n")
    predictedFile = open(testDirectory+"/truth.txt","r+").read().split("\n")

    goldFile = [line.split(":::")[:3] for line in goldFile if line]
    goldFile = sorted(goldFile, key=itemgetter(0))
    
    predictedFile = [line.split(":::") for line in predictedFile if line]
    predictedFile = sorted(predictedFile, key=itemgetter(0))


    goldGenders = [i[1] for i in goldFile]
    goldAges = [i[2] for i in goldFile]
    predictedGenders = [i[1] for i in predictedFile]
    predictedAges = [i[2] for i in predictedFile]

    goldCombined = goldGenders + goldAges
    predictedCombined = predictedGenders + predictedAges

    #Gender
    accuracyGender = accuracy_score(goldGenders, predictedGenders)
    metricsPerClassGender = classification_report(goldGenders, predictedGenders)
    confusionMatrixGender = sklearn.metrics.confusion_matrix(goldGenders, predictedGenders)
    print("\nGender:\nAccuracy = ", accuracyGender)
    print(metricsPerClassGender)
    print(confusionMatrixGender)

    #Age
    accuracyAge = accuracy_score(goldAges, predictedAges)
    metricsPerClassAge = classification_report(goldAges, predictedAges)
    confusionMatrixAge = sklearn.metrics.confusion_matrix(goldAges, predictedAges)
    print("\nAge:\nAccuracy = ", accuracyAge)
    print(metricsPerClassAge)
    print(confusionMatrixAge)

    #Gender+Age
    accuracyCombined = accuracy_score(goldCombined, predictedCombined)
    metricsPerClassCombined = classification_report(goldCombined, predictedCombined)
    confusionMatrixCombined = sklearn.metrics.confusion_matrix(goldCombined, predictedCombined)
    print("\nCombined:\nAccuracy = ", accuracyCombined)
    print(metricsPerClassCombined)
    print(confusionMatrixCombined)

    total_time = time.time() - t0
    print("total time: ", total_time)
    

    
def identity(x):
    return x


def tweetIdentity(arg):
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(arg)
    return tokens


def customLemmatizer(arg):
    """
    Preprocesser function to test different lemma.
    """
    wnl = WordNetLemmatizer()
    st = LancasterStemmer()
    return st.stem(wnl.lemmatize(arg))



def classifyGender(train_tweets, train_genders):
    #('preprocessor', CustomPreprocessor()),

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


    gender_stereotypes_vec = Pipeline([
     ('stereotypes', LinguisticGenderFeatures()),
     ('vec', DictVectorizer())
     ])

    combined_feats = FeatureUnion([("vec_word", vec_word), ("vec_char", vec_char), ("vec_stereo", gender_stereotypes_vec)])



    classifier = Pipeline([('vec', combined_feats),
                            # ('classifier', SVC(C=1, kernel="linear"))])
                            ('classifier', LinearSVC(multi_class='crammer_singer'))])

    
    classifier.fit(train_tweets, train_genders)  
    return classifier


def classifyAge(train_tweets, train_ages):
    #('preprocessor', CustomPreprocessor()),

   
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
                            # ('classifier', SVC(kernel="linear"))])
                            ('classifier', LinearSVC())])


    classifier.fit(train_tweets, train_ages)  
    return classifier


class LinguisticGenderFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def _get_features(self, doc):
        counts = Counter(doc)
        text_string = " ".join(doc)
        pos_tagged_text = nltk.pos_tag(doc)
        apologetic_words = ["sorry", "scusa", "scusi", "colpa", "excuus", "spijt", "siento", "culpa"]
        tag_questions = ["right?", "isn't it?", "aren't they?", "verdad?", "toch?", "giusto?", "vero?"]
        swearwords = ["shit", "crap", "fuck", "merda", "cazzo", "gvd", "kut", "mierda"]
        return {
                    "swearing": len([word for word in swearwords if word in doc])
                    #"words": len(doc)
                    # "unique_words": len(set(doc)),
                    # "adjectives": len([word[1] for word in pos_tagged_text if word[1] == "JJ"]),
                    # "adverbs": len([word[1] for word in pos_tagged_text if word[1] == "RB"]),
                    # "exclamation": counts["!"],
                    # "apologetic_lang": len([word for word in doc if word in apologetic_words]),
                    # "tag_questions": len([tag for tag in tag_questions if tag in text_string]),
                    # "questions": counts["?"]}
                }
       
    def transform(self, raw_documents):
     return [ self._get_features(doc) for doc in raw_documents]

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
    

def createDocuments(trainDirectory,testDirectory):
    docWithAllTraining = []

    for f in os.listdir(trainDirectory):
        if f != "truth.txt": #ignore the (possible) txt file at the end
            document = open(trainDirectory+'/'+f,"r+").read() 

            goldDocument = open(trainDirectory+'/truth.txt',"r+").read().split("\n")

            # print(document)
            idName = f[:-4] #filename - .xml
            tree = ET.parse(trainDirectory+"/"+f)
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
    for f in os.listdir(testDirectory):
        if f != "truth.txt": #ignore the (possible) txt file at the end
            document = open(testDirectory+'/'+f,"r+").read() 

            # print(document)
            idName = f[:-4] #filename - .xml
            tree = ET.parse(testDirectory+"/"+f)
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





    
if __name__ == '__main__':
    main()
