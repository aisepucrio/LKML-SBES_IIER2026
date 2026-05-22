# -*- coding: utf-8 -*-
# Adaptado para usar lkml_200_each.csv
# Some code taken from: 
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py
# The rest implemented by Daniel Schneider 1/17/2016
#
# This program creates a classifier which identifies emails as written by Linus Torvalds or Greg Kroah-Hartman
# --------------------------------------------------------------------------
# from sklearn.cross_validation import train_test_split  # deprecated, not used in this script
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pylab as plt
from time import time
import os
import csv
import sys
import numpy as np

# Aumentar limite de tamanho de campo do CSV para emails grandes
# Usando valor maximo compativel com Python 2
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import random
import re
import operator
from sklearn import preprocessing
from scipy import sparse
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize
import math

# IDs dos autores - multiplos IDs
GREG_IDS = ['33b4687a1bce892dd50a292f1aef7123d34e1e80', '430a69d897a421da23b624bcb40ad627abc7e441', 'e70a43732b757912deeedad36816c065f0e39c9d', '7837fbaa206929b8f395922b48b1fc9e03378645', 'aa271e70d86fad5d6665d697f8baf9716f1ae5f4', '6e4bcf1b60ab5c61e83421c8dc650372e24e8d30', 'b110ccef81be58563f9927ed0ad381fd3b8c27c3', 'fc5c4013575ce84165f39a48752864b2599f0c09', 'acd6510461b419fa8f9a3c98d79c98237621ea13', '70a919d8d4f794454ad3299c9458ede47c4b0cfd', 'f5d7def3c8a06958825ac8cc56a8f219a9f73e2b', '05f50a6602cb7cceea42722d080b0e378054459d', '660ffbe0d7dddc9ed2a1ab75193ea94c00a90ea8', '1ec27bbb0b7176c71e2135efba83d9c62ffcd8a6', 'f82d4f23a31c5580a3275bf1f4536c4c8f4b8de2', '7a64ba77bf8ad161daed7e83f5952233c8142d41', 'dad0af67ed4efb6a6ff758c3087705e396872f44', '72d8fb1d503e792fe448a0c4e0216016ad60a28a', '93c6919221283bb802f996008416db1cd4789cd9', '9f5149445d435c5f1c724425212459bf27b6c7e4', '11982cc8632a0a0687b5cb67db66cf0d0ac58d3e', '2cbd57694a0f21b2a2ce21bedae36c4f2cd1d729', '81699b8de124cca618159d04df6590497890c08c', '9fa2914e8f12c36281abed5a1c3a7224b6fc53c1', 'c8397cb34898c6a4a868f6ef7270e17f356187ab', '385633c311173e5c0f51b999e95adf65e3ee4e22', '4a020a0d2303f242f4b8204c8d376b05a4d87ff4', '14cb28d68fb5f33eaa28ed14c5739adaed3880bd', '92c226f50fae9c43efc70a46f7ad6a5bcc8d0afd', 'b382adecf09126bde82ab379400163903ee58e08', 'e4edc349285a5300ae514b0d4b8b8c4db0999009', '6d15ffbd57d37baa9461a369f811ca0d96afa621', '73f52bb918789ec16852513b78d302e5b3919f53', 'd6ebf53e5f112c4c8aea537bc3ea715f68dfcb1b', '4ba4e05c0b2bc2cccd575216f26a3bf04761a1a9', 'dd64aa010861c2735bdf54f4759d659f7b46decf', '8a9549962d4ab03b239758aed7e35f751ef2151e', '1ae126ae742ad1a11d5864d97d754427c511fefb', '7b61913195110a2a07ed1ef0fea94499ca5af970', 'f8e04f5367e3af276dca62b0e75b95907d6b8e47', 'b712a9ef26397c93236c51e442ecd21b3b29e1a7', '9701dc3adea0736ecd51cbf3079896550e868865', '032a520bf885d1546678ed1ff5c0e6510d1b14a5', '3d4f58cf724baee4fe738b35eb7f7ddbc2681aea', 'fe81b24c0c778fc14772a208cfec923639ced911', 'a7144fc79211e2cdbcc528cc65ef6da0df5ae8ae', 'e634071a9b93d8e96019ca338326b9236667b525', '2511d2677026be73a4f0712376af93ae3e267bb9', '97dcd263f2de3b332d8a0060f1605f8932151d38', '4eedfc192372cf3da734b3707e2054cc9a9be6e9', '857007023ceb9504f51e9b078476c99f49c7bec6', '0e853ad2dd743960f49da7512b8491bb0f2fb32b', 'd296b3c2a8be2c6ee03aaf8a4f59d0bfcf440520', 'ad1011be223a024fbc2621eba8e2ebf317cf29a6', 'fb6690f09aa527dcd64eb3ccc41f560b5417cf5b', '0ebbabf54ea23f67e597fad03505ea3b618661a8']
LINUS_IDS = ['a97dfd08c1662803a39537e7e42c93bf2db27128', 'd04d508bd126c72cc5eb1b81ab25ef19678060be', '3507a1d944456b02389f26658705675c228d3c1a', '3bd208ad71e484c15efd05cf4e5f5c46d77168c9', '8dc38ed98e552fdf8ed1c2cda11d2bba43c467f0', 'd4db8227a2f6acd9396e2f96a574416cbbf86304', '38d186e8d1752771441f67080ca38409d807c50a', 'a86865bfe5b8f588be22e0f968b86ca05809ca1d', 'e7ee4d5fa7b3eb2fbb1005393cee4d3f8f0aa74f', '49e50779221eb10c41e9309c6613755da56299c8', '07ab1331b0fbe2deb4a8e5349795bfeed7480588', 'f5c9a22d72bffbfb7c3c6690db9a8982ab1c57ac']

#Initialize multiple variables for this program
#numtests is the number of iterations the classifier will run
#overallaccuracy is used to compute the accuracy of the classifier for 1 or more runs
#avgTruePositives,avgFalseNegative,avgFalsePositive,avgTrueNegative are variables 
#that build a confusion matrix overall for the number of iterations the classifier ran
#averageAllTerms computes the average score for each of the most discerning words computed by the classifier
#avgAllRatios computes the Linus:Greg ratio for each of the most discerning words computed by the classifier
numTests = 10
overallAccuracy = 0
avgTruePositives = 0
avgFalseNegative = 0
avgFalsePositive = 0
avgTrueNegative = 0
avgPrecision = 0
avgRecall = 0
avgAllTerms = {}
avgAllRatios = {}

#Computes the Linus:Greg ratio for a supplied word
def findRatioForCertainWord(vectorizer, clf, word):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.feature_log_prob_[0], clf.feature_log_prob_[1], feature_names))
    for (myScore1, myScore2, myName) in coefs_with_fns:
        if myName == word:
            logTransformedScore1 = math.exp(myScore1)
            logTransformedScore2 = math.exp(myScore2)
            return logTransformedScore1 / (logTransformedScore1 + logTransformedScore2)

# Initialize posCorpus and negCorpus arrays
# These arrays will hold emails whether written by Greg (posCorpus) or Linus (negCorpus)
posCorpus = []  # Greg
negCorpus = []  # Linus

# Diretorio do script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ler o CSV e separar emails por autor
csv_path = os.path.join(script_dir, "messages_output.csv")
with open(csv_path, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        author_id = row['from']
        contents = row['raw_body']
        
        # Processar conteúdo
        contents = contents.lower()
        
        # Remover sequências de x's
        xsToRemove = "xx"
        for i in range(2, 20):
            contents = contents.replace(xsToRemove, "")
            xsToRemove = xsToRemove + "x"
        
        if str(author_id) in LINUS_IDS:
            # Remover nome do Linus
            contents = contents.replace("linus", "")
            contents = contents.replace("torvalds", "")
            negCorpus.append(contents)
        elif str(author_id) in GREG_IDS:
            # Remover nome do Greg
            contents = contents.replace("greg", "")
            contents = contents.replace("hartman", "")
            contents = contents.replace("kroah", "")
            contents = contents.replace("kh", "")
            posCorpus.append(contents)

print "FINISHED LOADING DATA"
print "Loaded", len(negCorpus), "emails from Linus"
print "Loaded", len(posCorpus), "emails from Greg"

#Read in a list of expletive words compiled by Google https://gist.github.com/jamiew/1112488
#Assign this list to a variable called expletiveList
expletiveList = []
with open(os.path.join(script_dir, "googleListOfExpletives")) as fh:
    contents = fh.read()
    contents = contents.lower()
    for word in contents.split():
        expletiveList.append(word)

#Run the classifier a pre-decided number of times
for x in range(0, numTests):
    # shuffle the positve and negative corpuses
    random.shuffle(posCorpus)
    random.shuffle(negCorpus)
    
    # assign 8/10 of Linus's emails and Greg's emails to the training set
    dataTrain = posCorpus[:len(posCorpus)*8/10] + negCorpus[:len(negCorpus)*8/10] # combined corpus
    
    #this portion of code initializes three lists which will be used as custom terms in the TFIDF classifier
    #dataTrainExpletiveCount is a count of the number of expletives for each of the email documents
    #dataTrainExclamationsCount is a count of the number of exclamations for each of the email documents
    #dataTrainAdverbCount is a count of the number of adverbs for each of the email documents
    dataTrainExpletiveCounts = []
    dataTrainExclamationsCount = []
    dataTrainAdverbCount = []
    
    for doc in dataTrain:
        #Count the number of adverbs in a document and put them in a list called dataTrainAdverbCount
        #adverbs generated from nltk's built in part-of-speech tagging feature
        adverbCount = 0
        try:
            text = word_tokenize(doc)
            for phrase in nltk.pos_tag(text):
                if phrase[1] == "RB" or phrase[1] == "RBR":
                    adverbCount = adverbCount + 1
        except:
            pass
        dataTrainAdverbCount.append(float(adverbCount));
        
        
        #Count expletives and number of ! marks and put them in lists
        expletiveCount = 0
        for word in doc.split():
            if len(word) > 1 and word in expletiveList and word not in ['hit', 'job', 'cum', 'cox', 'pisses', 'willy', 'knob', 'carpet', 'fudge']:
                expletiveCount = expletiveCount + 1
        dataTrainExpletiveCounts.append(float(expletiveCount))
        dataTrainExclamationsCount.append(float(doc.count("!")))
    
    # set up a vector to hold the class label (0 or 1)
    y_train_pos = np.ones(len(posCorpus[:len(posCorpus)*8/10]))
    y_train_neg = np.zeros(len(negCorpus[:len(negCorpus)*8/10]))
    y_train = np.hstack((y_train_pos, y_train_neg))
    
    #CHOOSE VECTORIZER TO RUN WITH
    #create a TF-IDF vectorizers and train it with the training documents
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    
    #comment previous vectorizer and uncomment next line to run code with frequency count vectors
    #vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                #stop_words='english')
                                
    #comment previous vectorizer and uncomment next line to run code with bigrams                        
    #vectorizer = TfidfVectorizer(min_df=1, ngram_range=(2,2))
    
    #comment previous vectorizer and uncomment next line to run code with bigrams                        
    #vectorizer = TfidfVectorizer(min_df=1, ngram_range=(2,2))
    
    X_train = vectorizer.fit_transform(dataTrain)
    
    #normalize the custom features lists from 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    normalizedTrainList = min_max_scaler.fit_transform(np.array(dataTrainExpletiveCounts).reshape(-1, 1))
    normalizedExclamationsTrainList = min_max_scaler.fit_transform(np.array(dataTrainExclamationsCount).reshape(-1, 1))
    normalizedAdverbTrainList = min_max_scaler.fit_transform(np.array(dataTrainAdverbCount).reshape(-1, 1))
    
    #add the normalized lists to the training vectors
    X_train = sparse.hstack((X_train, sparse.csr_matrix(normalizedTrainList)))
    X_train = sparse.hstack((X_train, sparse.csr_matrix(normalizedExclamationsTrainList)))
    X_train = sparse.hstack((X_train, sparse.csr_matrix(normalizedAdverbTrainList)))

   
    # create test group from 2/10 of the documents
    dataTest = posCorpus[len(posCorpus)*8/10:] + negCorpus[len(negCorpus)*8/10:]
    
    #this portion of code initializes three lists which will be used as custom terms in the TFIDF classifier
    #dataTrainExpletiveCount is a count of the number of expletives for each of the email documents
    #dataTrainExclamationsCount is a count of the number of exclamations for each of the email documents
    #dataTrainAdverbCount is a count of the number of adverbs for each of the email documents
    dataTestExpletiveCounts = []
    dataTestExclamationsCounts = []
    dataTestAdverbCount = []
    for doc in dataTest:
        adverbCount = 0
        try:
            text = word_tokenize(doc)
            for phrase in nltk.pos_tag(text):
                if phrase[1] == "RB" or phrase[1] == "RBR":
                    adverbCount = adverbCount + 1
        except:
            pass
        dataTestAdverbCount.append(float(adverbCount));
        
        expletiveCount = 0
        for word in doc.split():
            if len(word) > 1 and word in expletiveList and word not in ['hit', 'job', 'cum', 'cox', 'pisses', 'willy', 'knob', 'carpet', 'fudge']:
                expletiveCount = expletiveCount + 1
        dataTestExpletiveCounts.append(float(expletiveCount))
        dataTestExclamationsCounts.append(float(doc.count("!")))
    
    # set up a vector to hold the class label (0 or 1)
    y_test_pos = np.ones(len(posCorpus[len(posCorpus)*8/10:]))
    y_test_neg = np.zeros(len(negCorpus[len(negCorpus)*8/10:]))
    y_test = np.hstack((y_test_pos, y_test_neg))
    
    # each row is a test document, each column is a tfidf score
    X_test = vectorizer.transform(dataTest)
    
    #normalize the custom features lists from 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    normalizedTestList = min_max_scaler.fit_transform(np.array(dataTestExpletiveCounts).reshape(-1, 1))
    normalizedExclamationsTestList = min_max_scaler.fit_transform(np.array(dataTestExclamationsCounts).reshape(-1, 1))
    normalizedAdverbTestList = min_max_scaler.fit_transform(np.array(dataTestAdverbCount).reshape(-1, 1))
    
    #add the normalized lists to the test vectors    
    X_test = sparse.hstack((X_test, sparse.csr_matrix(normalizedTestList)))
    X_test = sparse.hstack((X_test, sparse.csr_matrix(normalizedExclamationsTestList)))
    X_test = sparse.hstack((X_test, sparse.csr_matrix(normalizedAdverbTestList)))
    
    #create a multinomial naive bayes classifier and fit it with the training data
    #predict the test data group based on the training
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_test_pred = nb.predict(X_test)
    y_test_probs = nb.predict_proba(X_test)
    
    # Compare predicted class labels with actual class labels
    accuracy = nb.score(X_test, y_test)
    
    #Add this accuracy to overall accuracy to be averaged later
    overallAccuracy = overallAccuracy + accuracy
    
    #Add precision and recall to overall sums to be averaged later
    avgPrecision = avgPrecision + precision_recall_fscore_support(y_test, y_test_pred, average='binary')[0]
    avgRecall = avgRecall + precision_recall_fscore_support(y_test, y_test_pred, average='binary')[1]
    
    #construct a confusion matrix and store the measures as sums to be averaged later
    nbcm = confusion_matrix(y_test, y_test_pred)
    avgTruePositives = avgTruePositives + nbcm[0][0]
    avgFalseNegative = avgFalseNegative + nbcm[0][1]
    avgFalsePositive = avgFalsePositive + nbcm[1][0]
    avgTrueNegative = avgTrueNegative + nbcm[1][1]
    
    #calculate the top 100 most influential features and put them in a dictionary with their score
    nFeats = 100
    ch2 = SelectKBest(chi2, k=nFeats)
    X_train_2 = ch2.fit_transform(X_train, y_train)
    X_test_2 = ch2.transform(X_test)
    
    #find all feature_names and add the custom feature names to this list
    feature_names = vectorizer.get_feature_names()
    feature_names.append("expletivesCount")
    feature_names.append("exclamationsCount")
    feature_names.append("adverbCount")
    
    
    #loop through the computed most influential features and determine their score and ratios
    #add scores to a dictionary called avgAllTerms where the name of the feature is the key
    #and the sum of the scores as well as the number of times this term appeared in each of the classifier iterations is the value
    #add ratios to a dictionary called avgAllRatios where the name of the feature is the key
    #and the sum of the ratios as well as the number of times this term appeared in each of the classifier iterations is the value
    #the sum of the scores and ratios will be averaged with the number of times the word appeared as a classifier term later in the program
    #Note: the custom features are ignored for ratios as they are not part of the original vectorizer component, so their ratios cannot be computed
    for i in ch2.get_support(indices=True):
        if feature_names[i] in avgAllTerms.keys():
            avgAllTerms[feature_names[i]] = (avgAllTerms[feature_names[i]][0] + ch2.scores_[i], avgAllTerms[feature_names[i]][1] + 1)
        else:
            avgAllTerms[feature_names[i]] = (ch2.scores_[i], 1)
        if feature_names[i] != 'expletivesCount' and feature_names[i] != 'exclamationsCount' and feature_names[i] != 'adverbCount':
            if feature_names[i] in avgAllRatios.keys():
                avgAllRatios[feature_names[i]] = (avgAllRatios[feature_names[i]][0] + findRatioForCertainWord(vectorizer, nb, feature_names[i]), avgAllRatios[feature_names[i]][1] + 1)
            else:
                avgAllRatios[feature_names[i]] = (findRatioForCertainWord(vectorizer, nb, feature_names[i]), 1)
    print "Iteration " + str(x + 1) + " complete"
    

#Compute average measures by dividing by the number of times they were tested
overallAccuracy = overallAccuracy / numTests
avgTruePositives = float(avgTruePositives) / numTests
avgFalseNegative = float(avgFalseNegative) / numTests
avgFalsePositive = float(avgFalsePositive) / numTests
avgTrueNegative = float(avgTrueNegative) / numTests
avgPrecision = float(avgPrecision) / numTests
avgRecall = avgRecall / numTests

#compute average scores for all terms by dividing by the number of times they appeared in all of the iterations of classification
for word in avgAllTerms:
    avgAllTerms[word] = (avgAllTerms[word][0] / avgAllTerms[word][1], avgAllTerms[word][1])

#compute average ratios for all terms by dividing by the number of times they appeared in all of the iterations of classification
for ratio in avgAllRatios:
    avgAllRatios[ratio] = (avgAllRatios[ratio][0] / avgAllRatios[ratio][1], avgAllRatios[ratio][1])

#sort the score and ratio dictionaries in descending order
sorted_terms = sorted(avgAllTerms.items(), key=operator.itemgetter(1), reverse=True)
sorted_ratios = sorted(avgAllRatios.items(), key=operator.itemgetter(1), reverse=True)

#Print measures
print
print "=" * 60
print "RESULTS"
print "=" * 60
print "NB Accuracy: %.10f" % overallAccuracy
print "NB True Positives: %.10f" % avgTruePositives
print "NB False Negatives: %.10f" % avgFalseNegative
print "NB False Positives: %.10f" % avgFalsePositive
print "NB True Negatives: %.10f" % avgTrueNegative
print "Precision: %.10f" % avgPrecision
print "Recall: %.10f" % avgRecall
print

print "Most important words:"
for word, score in sorted_terms:
    print word + ": " + str(score)
    
print
print "Most important ratios:"
for word, linusRatio in sorted_ratios:
    print word + "            linus:greg         " + str(linusRatio[0]) + ":" + str(1 - linusRatio[0])
    
print
print "Most important ratios sorted by importance of words:"
for word, score in sorted_terms:
    if word in avgAllRatios.keys():
        print word + "            linus:greg         " + str(avgAllRatios[word][0])

# Salvar metricas gerais
output_metrics = os.path.join(script_dir, "results_classifier_metrics.csv")
with open(output_metrics, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Accuracy', overallAccuracy])
    writer.writerow(['True Positives', avgTruePositives])
    writer.writerow(['False Negatives', avgFalseNegative])
    writer.writerow(['False Positives', avgFalsePositive])
    writer.writerow(['True Negatives', avgTrueNegative])
    writer.writerow(['Precision', avgPrecision])
    writer.writerow(['Recall', avgRecall])

print "\nMetrics saved to:", output_metrics

# Salvar palavras importantes
output_words = os.path.join(script_dir, "results_classifier_important_words.csv")
with open(output_words, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Score', 'Times_Appeared', 'Linus_Ratio', 'Greg_Ratio'])
    for word, score in sorted_terms:
        linus_ratio = avgAllRatios.get(word, (None, 0))[0]
        greg_ratio = (1 - linus_ratio) if linus_ratio is not None else None
        writer.writerow([word, score[0], score[1], linus_ratio, greg_ratio])

print "Important words saved to:", output_words
