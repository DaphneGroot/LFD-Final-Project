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

from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


def main():
    t0 = time.time() #added to see the total duration of the program

    #read documents
    document = open('training/data-dutch-preprocessed.txt', 'r', encoding="utf-8").read().split("END\n")

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

class LinguisticGenderFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def _get_features(self, doc):
        # if language == "english":
        #     diminutives = ["ie"]
        # elif language == "spanish":
        #     diminutives = ["ito", "ita", "ico", "ica"]
        # elif language == "italian":
        #     diminutives = ["ino", "ina", "etto", "etta", "uolo", "uola", "ucolo", "ucola"]
        # elif language == "dutch":
        #     diminutives = ["tje", "gje", "sje", "pje", "kje"]
        counts = Counter(doc)
        text_string = " ".join(doc)
        pos_tagged_text = nltk.pos_tag(doc)
        apologetic_words = ["sorry", "scusa", "scusi", "colpa", "excuus", "spijt", "siento", "culpa"]
        tag_questions = ["right?", "isn't it?", "aren't they?", "verdad?", "toch?", "giusto?", "vero?"]
        return {"words": len(doc),
                "unique_words": len(set(doc)),
                "adjectives": len([word[1] for word in pos_tagged_text if word[1] == "JJ"]),
                "adverbs": len([word[1] for word in pos_tagged_text if word[1] == "RB"]),
                "exclamation": counts["!"],
                "apologetic_lang": len([word for word in doc if word in apologetic_words]),
                "tag_questions": len([tag for tag in tag_questions if tag in text_string]),
                "questions": counts["?"]}
    def transform(self, raw_documents):
     return [ self._get_features(doc) for doc in raw_documents]

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
