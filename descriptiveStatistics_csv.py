# -*- coding: utf-8 -*-
# Adaptado para usar lkml_200_each.csv
# Calcula estatisticas descritivas dos emails de Linus e Greg

import os
import csv
import sys
import operator

# Aumentar limite de tamanho de campo do CSV para emails grandes
# Usando valor maximo compativel com Python 2
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
from sklearn import preprocessing
import nltk
from nltk import *
import re

# IDs dos autores - multiplos IDs
GREG_IDS = ['2511d2677026be73a4f0712376af93ae3e267bb9', '6d15ffbd57d37baa9461a369f811ca0d96afa621', 
            '857007023ceb9504f51e9b078476c99f49c7bec6', '93c6919221283bb802f996008416db1cd4789cd9', 
            'b712a9ef26397c93236c51e442ecd21b3b29e1a7', 'e47776b220235deb3b58b762e79e03082cba5b68', 
            'f5d7def3c8a06958825ac8cc56a8f219a9f73e2b']
LINUS_IDS = ['062aad6a1a302c7688ef109864b5c8c0f59ebcbd', '269f83821d924a5a87f6e23d983ca65a1f127531', 
             '38d186e8d1752771441f67080ca38409d807c50a', '8dc38ed98e552fdf8ed1c2cda11d2bba43c467f0', 
             '9d7c2b0cfc3382dc2cbafed83b632890eeaa663e', 'a86865bfe5b8f588be22e0f968b86ca05809ca1d', 
             'a97dfd08c1662803a39537e7e42c93bf2db27128', 'e7ee4d5fa7b3eb2fbb1005393cee4d3f8f0aa74f']

negCorpus = []  # Linus
posCorpus = []  # Greg

# Diretorio do script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ler o CSV e separar emails por autor
csv_path = os.path.join(script_dir, "lkml_messages_greg_linus.csv")
with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        author_id = row['author_id']
        sender = row['sender']  # Coluna sender disponivel para referencia
        contents = row['raw_body']
        
        # Processar conteúdo
        contents = contents.lower()
        
        # Remover sequências de x's
        xsToRemove = "xx"
        for i in range(2, 20):
            contents = contents.replace(xsToRemove, "")
            xsToRemove = xsToRemove + "x"
        
        if author_id in LINUS_IDS:
            # Remover nome do Linus
            contents = contents.replace("linus", "")
            contents = contents.replace("torvalds", "")
            negCorpus.append(contents)
        elif author_id in GREG_IDS:
            # Remover nome do Greg
            contents = contents.replace("greg", "")
            contents = contents.replace("hartman", "")
            contents = contents.replace("kroah", "")
            contents = contents.replace("kh", "")
            posCorpus.append(contents)

print "Loaded", len(negCorpus), "emails from Linus"
print "Loaded", len(posCorpus), "emails from Greg"
print "FINISHED LOADING"

# Funcao para calcular Flesch Reading Ease
def flesch_reading_ease(text, total_sentences, total_words, total_syllables):
    if total_sentences == 0 or total_words == 0:
        return 0
    return 206.835 - 1.015 * (float(total_words) / float(total_sentences)) - 84.6 * (float(total_syllables) / float(total_words))

# Funcao para calcular Flesch-Kincaid Grade Level
def flesch_kincaid_grade(text, total_sentences, total_words, total_syllables):
    if total_sentences == 0 or total_words == 0:
        return 0
    return 0.39 * (float(total_words) / float(total_sentences)) + 11.8 * (float(total_syllables) / float(total_words)) - 15.59

# Funcao simples para contar silabas
def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    # Minimo de 1 silaba por palavra
    if syllable_count == 0:
        syllable_count = 1
    
    # Silaba 'e' muda no final (ex: 'like' = 1 silaba, nao 2)
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return syllable_count

linusWordCount = 0 
linusSentenceCount = 0
linusLexicalDiversity = 0
fleshReadingLinus = 0
fleshGradeLevelLinus = 0
linusTotalSyllables = 0
uncomputableLinus = 0

for doc in negCorpus:
    # Contar palavras
    words = doc.split()
    word_count = len(words)
    linusWordCount = linusWordCount + word_count
    
    # Contar sentencas usando NLTK
    try:
        sentences = nltk.sent_tokenize(doc)
        sentence_count = len(sentences)
        linusSentenceCount = linusSentenceCount + sentence_count
    except:
        sentence_count = 1
        linusSentenceCount = linusSentenceCount + 1
    
    # Lexical diversity
    if word_count > 0:
        linusLexicalDiversity = linusLexicalDiversity + (float(len(set(words))) / float(word_count))
    
    # Contar silabas
    syllable_count = 0
    for word in words:
        # Limpar palavra de pontuacao
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if clean_word:
            syllable_count += count_syllables(clean_word)
    
    linusTotalSyllables = linusTotalSyllables + syllable_count
    
    # Calcular Flesch scores
    try:
        if sentence_count > 0 and word_count > 0:
            flesch_score = flesch_reading_ease(doc, sentence_count, word_count, syllable_count)
            grade_score = flesch_kincaid_grade(doc, sentence_count, word_count, syllable_count)
            fleshReadingLinus = fleshReadingLinus + flesch_score
            fleshGradeLevelLinus = fleshGradeLevelLinus + grade_score
    except:
        uncomputableLinus = uncomputableLinus + 1

gregWordCount = 0 
gregSentenceCount = 0
gregLexicalDiversity = 0
fleshReadingGreg = 0
fleshGradeLevelGreg = 0
gregTotalSyllables = 0
uncomputableGreg = 0

for doc in posCorpus:
    # Contar palavras
    words = doc.split()
    word_count = len(words)
    gregWordCount = gregWordCount + word_count
    
    # Contar sentencas usando NLTK
    try:
        sentences = nltk.sent_tokenize(doc)
        sentence_count = len(sentences)
        gregSentenceCount = gregSentenceCount + sentence_count
    except:
        sentence_count = 1
        gregSentenceCount = gregSentenceCount + 1
    
    # Lexical diversity
    if word_count > 0:
        gregLexicalDiversity = gregLexicalDiversity + (float(len(set(words))) / float(word_count))
    
    # Contar silabas
    syllable_count = 0
    for word in words:
        # Limpar palavra de pontuacao
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if clean_word:
            syllable_count += count_syllables(clean_word)
    
    gregTotalSyllables = gregTotalSyllables + syllable_count
    
    # Calcular Flesch scores
    try:
        if sentence_count > 0 and word_count > 0:
            flesch_score = flesch_reading_ease(doc, sentence_count, word_count, syllable_count)
            grade_score = flesch_kincaid_grade(doc, sentence_count, word_count, syllable_count)
            fleshReadingGreg = fleshReadingGreg + flesch_score
            fleshGradeLevelGreg = fleshGradeLevelGreg + grade_score
    except:
         uncomputableGreg = uncomputableGreg + 1   
         

print
print "=" * 60
print "DESCRIPTIVE STATISTICS"
print "=" * 60
print "Linus Average Word Count: " + str(float(linusWordCount) / len(negCorpus))
print "Greg Average Word Count: " + str(float(gregWordCount) / len(posCorpus))
print
print "Linus Average Sentence Count: " + str(float(linusSentenceCount) / float(len(negCorpus) - uncomputableLinus))
print "Greg Average Sentence Count: " + str(float(gregSentenceCount) / float(len(posCorpus) - uncomputableGreg))
print
print "Linus Average Lexical Diversity: " + str(float(linusLexicalDiversity) / len(negCorpus))
print "Greg Average Lexical Diversity: " + str(float(gregLexicalDiversity) / len(posCorpus))
print
print "Linus Average Flesch Reading Ease Score: " + str(fleshReadingLinus / (len(negCorpus) - uncomputableLinus))
print "Greg Average Flesch Reading Ease Score: " + str(fleshReadingGreg / (len(posCorpus) - uncomputableGreg))
print
print "Linus Average Grade Level: " + str(fleshGradeLevelLinus / (len(negCorpus) - uncomputableLinus))
print "Greg Average Grade Level: " + str(fleshGradeLevelGreg / (len(posCorpus) - uncomputableGreg))

# Salvar resultados em CSV
output_path = os.path.join(script_dir, "results_descriptiveStatistics.csv")
with open(output_path, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Linus', 'Greg'])
    writer.writerow(['Average Word Count', linusWordCount / len(negCorpus), gregWordCount / len(posCorpus)])
    writer.writerow(['Average Sentence Count', 
                     float(linusSentenceCount) / float(len(negCorpus) - uncomputableLinus),
                     float(gregSentenceCount) / float(len(posCorpus) - uncomputableGreg)])
    writer.writerow(['Average Lexical Diversity',
                     float(linusLexicalDiversity) / len(negCorpus),
                     float(gregLexicalDiversity) / len(posCorpus)])
    writer.writerow(['Average Flesch Reading Ease Score',
                     fleshReadingLinus / (len(negCorpus) - uncomputableLinus),
                     fleshReadingGreg / (len(posCorpus) - uncomputableGreg)])
    writer.writerow(['Average Grade Level',
                     fleshGradeLevelLinus / (len(negCorpus) - uncomputableLinus),
                     fleshGradeLevelGreg / (len(posCorpus) - uncomputableGreg)])

print "\nResults saved to:", output_path
