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
Lemmatizing = 0.7383 <br/>
Stemming = 0.7383 <br/>
Stemming + Lemmatizing = 0.7383 <br/>

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
