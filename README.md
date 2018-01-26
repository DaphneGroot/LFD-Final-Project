# LFD-Final-Project

Results so far: <br/> <br/>


SVM with char 3-5 word ngrams 2-3, kernel = linear, other parameters default <br/>
# DUTCH:
Without preprocessing: <br/>
GENDER: 0.75 <br/>
AGE:  <br/>
With preprocessing: <br/>
GENDER: 0.7647 (also tried different settings for C, default (C=1) works best) <br/>
AGE: <br/>


# ENGLISH:
Without preprocessing: <br/>
GENDER: 0.753 <br/>
AGE:  <br/>
With preprocessing: <br/>
GENDER: 0.7597 (also tried different settings for C, default works best) <br/>
AGE: <br/>

# SPANISH
Without preprocessing (also tried different settings for C, default works best): <br/>
GENDER:0.753 <br/>
AGE:  <br/>
With preprocessing: <br/>
GENDER: 0.732 <br/>
AGE: <br/>

# ITALIAN
Without preprocessing: <br/>
GENDER: 0.7675 <br/>
AGE:  <br/>
With preprocessing: <br/>
GENDER: 0.769 <br/>
AGE: <br/>
With preprocessing + C=10: <br/>
GENDER: 0.773 <br/>

# -------------------------

# All Languages
Using LinearSVC with default parameters <br/>
(reported are the acc scores) <br/>

##  Gender
TweetTokenizer = 0.7328 <br/>
POStagging = 0.7301 (removed) <br/>
Lemmatizing = 0.7328 <br/>
Stemming = 0.7376 <br/>
Stemming + Lemmatizing = 0.7376 <br/>
Adding Text length (DictVectorizer) from lab = 0.718 (removed) <br/>
Adding stereotypical linguistic features: 0.734 (all of them) > took very long > maybe still worth it to check them one by one?<br/>
	Only 'questions' = 0.735 (duration: 537 seconds)<br/>
	Only 'tag-questions' = 0.736 (duration: 484 seconds)<br/>
	Only 'apologetic_lang' = 0.736 (duration: 508 seconds)<br/>
	Only 'exclamation' = 0.736 (duration: 933 seconds)<br/>

### Changing tf-idf vectorizer values
What does not work: <br/>
- binary = False <br/>
- lowercase = True (does not improve nor decline) <br/>
- strip_accents (does not improve for both ascii and unicode) <br/>
- removing stopwords <br/>
- min_df <br/>
- norm <br/>
- sublinear_tf <br/> <br/><br/>
- max_features

### Changing classifier parameters
What does not work: <br/>
- penalty <br/>
- multi_class <br/>
---------------------------- <br/><br/>

##  Age
TweetTokenizer = 0.7383 <br/>
POStagging = 0.7344 (removed) <br/>
Lemmatizing = 0.7387 <br/>
Stemming = 0.7385 <br/>
Stemming + Lemmatizing = 0.7385 <br/>
Adding Text length (DictVectorizer) from lab = 0.690 (removed) <br/>

### Changing tf-idf vectorizer values
What does work: <br/>
- max_features = 400.000 --> 0.7387 <br/> <br/>

What does not work: <br/>
- binary = False <br/>
- lowercase = True (does not improve nor decline) <br/>
- strip_accents (does not improve for both ascii and unicode) <br/>
- removing stopwords <br/>
- min_df <br/>
- norm <br/>
- sublinear_tf <br/> <br/>

### Changing classifier parameters
What does not work: <br/>
- penalty <br/>
- multi_class <br/>


# ----------------

## Checking other classifiers (just to be sure) on ALL DATA

### Gender
* all classifiers are default ones + stemmer/lemmatizer and tweettokenizer
- Multinomial Naive Bayes: 0.71
- Bernoulli Naive Bayes: 0.717
- Random Forest Classifier: 0.667
- Adaboost Classifier: 0.655
- Decision Tree: 0.62
- KNN: 0.676

### Age
* all classifiers are default ones + stemmer/lemmatizer and tweettokenizer
- Multinomial Naive Bayes: 0.64
- Bernoulli NB: 0.645
- Random Forest Classifier: 0.624
- Adaboost Classifier: 0.55
- Decision Tree: 0.55
- KNN: 0.65

- MLPclassifier is still running (after more than an hour). Might stop it if it still runs in 1 hour.
- We could potentially have a look at some Naive Bayes classifiers and tune those or just add more/different features to SVM or train neural nets.