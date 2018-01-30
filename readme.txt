How to run the program:
	python3 smvFinal.py [path to training directory] [path to testing directory] [path to gold standard file]
	E.g.: python3 svm.py training/english testing/english testing/english/gold.txt

	(if directory with the language name, is not in a overarching directory (like training), please use ./. E.g.: ./englishTraining ./englishTesting)

	the path to the gold standard file has to be given as well, in order to correctly calculate the metrics for the test set.

	You have to run the program again for every language


The program can be used for each of the four langauges (english, spanish, italian and dutch), and outputs the accuracy, precision, recall, f1-score and a confusion matrix for both gender, age and gender+age.

The file with the predictions written to it, is called truthPredicted.txt, to prevent overwriting of a truth.txt file.