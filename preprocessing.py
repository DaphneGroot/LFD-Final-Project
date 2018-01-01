import re, os, sys
from nltk.tokenize import TweetTokenizer

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
                    

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding="utf-8") as infile, open(sys.argv[2], "w", encoding="utf-8") as outfile:
        lines = infile.read().split("END\n")
        for line in lines:
            if line != "":
                line_items = line.split("\t")
                tweet = clean_tweets(line_items[1])
                new_line = line_items[0] + "\t" + tweet.strip() + "\t" + "\t".join(line_items[2:])
                outfile.write(new_line.strip() + " END\n")


    