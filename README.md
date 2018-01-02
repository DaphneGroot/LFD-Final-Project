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

# All Languages
Using LinearSVC with default parameters <br/>
(reported are the acc scores) <br/>
##  Gender
TweetTokenizer = 0.7328 <br/>
POStagging = 0.7301 (removed) <br/>
Lemmatizing = 0.7328 <br/>
Stemming = 0.7376 <br/>
Stemming + Lemmatizing = 0.7376 <br/>

