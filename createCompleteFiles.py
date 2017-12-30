import pickle
import sys
import numpy as np
import string

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer 
import nltk

#from createConfusionMatrix import main as mainCCM
from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer

import os
import xml.etree.ElementTree as ET



def documents(folderLanguage):
	docWithAll = open("training/data-"+folderLanguage+".txt","w+")

	for f in os.listdir("training/"+folderLanguage):
		if f != "truth.txt": #ignore the html file at the end
			document = open("training/"+folderLanguage+'/'+f,"r+").read() 

			goldDocument = open("training/"+folderLanguage+'/truth.txt',"r+").read().split("\n")

			# print(document)
			idName = f[:-4]
			tree = ET.parse("training/"+folderLanguage+"/"+f)
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

							docWithAll.write(idName+"\t"+tweet+"\t"+gender+"\t"+ageGroup+" END\n")

		# break



def main():
	#read documents
	folderLanguage = "spanish" #change per language, lowercase is necessary 

	documents(folderLanguage)

	
if __name__ == '__main__':
	main()
