import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
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

import numpy as np



def main():
    t0 = time.time() #added to see the total duration of the program

    #read documents
    document = open('training/data-dutch-preprocessed.txt', 'r', encoding="utf-8").read().split("END\n")
    svm = True

    if svm:

        trainDocuments, testDocuments = train_test_split(document, test_size=0.2, random_state=0)

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
        
        confusionMatrix = sklearn.metrics.confusion_matrix(test_genders, predicted_genders)
    else:
        #only import when running LSTM
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation
        from keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D
        from keras.layers import LSTM
        from keras.preprocessing import sequence
        from keras.preprocessing.text import Tokenizer
        from keras.utils import to_categorical
        from keras.preprocessing.sequence import pad_sequences
        from sklearn.preprocessing import LabelBinarizer


        trainDocuments, testDocuments = train_test_split(document, test_size=0.2, random_state=42)
        #create seperate lists for tweets and the genders
        train_tweets, train_genders = createLists(trainDocuments)
        test_tweets, test_genders = createLists(testDocuments)

        tweets = train_tweets+test_tweets
        genders = train_genders+test_genders

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tweets)
        X = tokenizer.texts_to_sequences(tweets)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X = pad_sequences(X)

        genders = np.asarray(genders)
        # genders = to_categorical(list(map(str,genders)))

        encoder = LabelBinarizer()
        genders = encoder.fit_transform(genders) # Use encoder.classes_ to find mapping of one-hot indices to string labels
        genders = np.where(genders == 1, [0,1], [1,0])

        # print(X[:5])
        # print(genders[:5])
        print('Shape of data tensor:', X.shape)
        print('Shape of label tensor:', genders.shape)


        X_train = X[:len(train_tweets)]
        Y_train = genders[:len(train_genders)]

        X_test = X[len(train_tweets):]
        Y_test = genders[len(train_tweets):]


        X_train = sequence.pad_sequences(X_train, X.shape[1])
        X_test = sequence.pad_sequences(X_test, X.shape[1])

        Y_train = sequence.pad_sequences(Y_train, genders.shape[1])
        Y_test = sequence.pad_sequences(Y_test, genders.shape[1])


        total_words = 5000
        max_length = 120
        embedding_vector_length = 32
        batch_size = 32
        epochs = 5

        # classifier = classify_lstm(X, genders, X_train, Y_train, total_words, max_length, embedding_vector_length, 500, 0.2, "relu")
        classifier = classify_lstm(X, genders, X_train, Y_train, batch_size = batch_size, epochs = epochs)

        predicted_genders = classifier.predict(X_test)

        print(predicted_genders[:10])

        n = 0
        for i in predicted_genders[:10]:
            print(max(i))
            if max(i) >= 0.95:
                n += 1

        print(n)

        predicted_genders = np.argmax(predicted_genders, axis = 1)
        Y_test = np.argmax(Y_test, axis = 1)

        print(predicted_genders[:10])
        # print(Y_test[:10])
        
        accuracy = accuracy_score(Y_test, predicted_genders)
        print("\nAccuracy: ", accuracy)

        
        metricsPerClass = classification_report(Y_test, predicted_genders)
        print("\nMetrics per class:\n",metricsPerClass)

        total_time = time.time() - t0
        print("total time Gender: ", total_time)
        
        # confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, predicted_genders)
    
    
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

# def classify_lstm(X, Y, train_X, train_Y, total_words, max_len, emb_vec_len, nodes, dropout, lstm_activation):
    # train_X = train_X.reshape(len(train_X), 1, len(train_X[0]))
    # #dev_X = dev_X.reshape((len(dev_X), 1, len(dev_X[0])))
    # model = Sequential()
    # #model.add(Embedding(total_words, emb_vec_len, input_length=max_len))
    # model.add(LSTM(nodes, input_shape=(len(train_X[0]), len(train_X[0][0])), dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation=lstm_activation))  
    # model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=dropout, activation=lstm_activation)) 
    # model.add(Dense(int(nodes/2), activation="relu"))
    # model.add(Dense(int(nodes/4), activation="relu"))
    # model.add(Dense(1, activation='sigmoid'))                       
    # model.compile(loss='mse', optimizer="adam", metrics=['mse']) 
    


    # model = Sequential()
    # model.add(Dense(32, input_shape=(X.shape[1],)))
    # model.add(Dense(Y.shape[1], activation='relu'))
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

def classify_lstm(X, Y, train_X, train_Y, batch_size, epochs):
    model = Sequential()
    model.add(Dense(input_dim=X.shape[1], units=Y.shape[1]))
    model.add(Activation("linear"))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.fit(train_X, train_Y,
                        batch_size=batch_size,
                        epochs=epochs)

    return model


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