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


    kf = KFold(n_splits=5)
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
        classifierGender = classifyGender(train_tweets, train_genders)
        predicted_genders = classifierGender.predict(test_tweets)

        #only predict age for spanish and english
        if language == "spanish" or language == "english":
            #predict age
            classifierAge = classifyAge(train_tweets, train_ages)
            predicted_ages = classifierAge.predict(test_tweets)
        else:
            predicted_ages = ["XX-XX" for i in range(len(test_ids))]



        #write predictions to truth file
        # outFile = open('training/'+language+"/truthPredicted.txt","w+")
        
        userDictGender = {}
        userDictAge = {}
        predictedGenders = []
        predictedAges = []
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

            predictedGenders.append(gender)
            predictedAges.append(age)

            # outFile.write(key+":::"+gender+":::"+age+"\n")

        # outFile.close()


        # Calculate metrics
        goldFile = open('training/'+language+'/truth.txt',"r+").read().split("\n")
        # predictedFile = open('training/'+language+"/truthPredicted.txt","r+").read().split("\n")

        goldFile = [line.split(":::")[:3] for line in goldFile if line]
        # goldFile = sorted(goldFile, key=itemgetter(0))[trainingSize:]
        
        # predictedFile = [line.split(":::") for line in predictedFile if line]
        goldFile = [i for i in goldFile if i[0]+".xml" in testData]
        goldFile = sorted(goldFile, key=itemgetter(0))


        goldGenders = [i[1] for i in goldFile]
        goldAges = [i[2] for i in goldFile]
        # predictedGenders = [i[1] for i in predictedFile]
        # predictedAges = [i[2] for i in predictedFile]
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