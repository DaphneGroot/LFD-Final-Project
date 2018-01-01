import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC


def main():
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
    
    # confusionMatrix = sklearn.metrics.confusion_matrix(test_genders, predicted_genders)
    
    
def identity(x):
    return x


def classify(train_tweets, train_genders):
    #('preprocessor', CustomPreprocessor()),

    vec_word = TfidfVectorizer(preprocessor = identity,
                         tokenizer = identity,
                         binary=True,
                         lowercase=False, 
                         analyzer='word', 
                         ngram_range=(1,2))

    vec_char = TfidfVectorizer(preprocessor = identity,
                         tokenizer = identity,
                         binary=True,
                         lowercase=False, 
                         analyzer='char', 
                         ngram_range=(3,5))


    combined_feats = FeatureUnion([("vec_word", vec_word), ("vec_char", vec_char)])

    classifier = Pipeline([('vec', combined_feats),
                            ('classifier', SVC(C=1, kernel="linear"))])

    

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