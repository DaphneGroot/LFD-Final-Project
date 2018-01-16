import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords as sw



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


def main():
    t0 = time.time() #added to see the total duration of the program

    #read documents
    document = open('training/data-all-preprocessed.txt', 'r', encoding="utf-8").read().split("END\n")

    trainDocuments, testDocuments = train_test_split(document, test_size=0.2, random_state=42)

    #create seperate lists for tweets and the genders
    train_tweets, train_genders = createLists(trainDocuments)
    test_tweets, test_genders = createLists(testDocuments)
    
    classifier = classify(train_tweets, train_genders)


    predicted_genders = classifier.predict(test_tweets)
    
    accuracy = accuracy_score(test_genders, predicted_genders)
    print("\nAccuracy: ", accuracy)

    
    metricsPerClass = classification_report(test_genders, predicted_genders)
    print("\nMetrics per class:\n",metricsPerClass)

    total_time = time.time() - t0
    print("total time Gender: ", total_time)
    
    # confusionMatrix = sklearn.metrics.confusion_matrix(test_genders, predicted_genders)
    
    
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


def classify(train_tweets, train_genders):
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


    combined_feats = FeatureUnion([("vec_word", vec_word), ("vec_char", vec_char)])

    classifier = Pipeline([('vec', combined_feats),
                            # ('classifier', SVC(C=1, kernel="linear"))])
                            ('classifier', LinearSVC(multi_class='crammer_singer'))])

    
    classifier.fit(train_tweets, train_genders)  
    return classifier

def createLists(documents):
    tweets = []
    genders = []

    for line in documents:
        if line != "":
            line = line.split("\t")
            tweet = line[1]
            gender = line[2]

            genders.append(gender)
            tweets.append(tweet)

    return tweets, genders
    

    
if __name__ == '__main__':
    main()
