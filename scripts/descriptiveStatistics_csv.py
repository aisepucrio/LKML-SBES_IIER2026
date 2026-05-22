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
GREG_IDS = ['33b4687a1bce892dd50a292f1aef7123d34e1e80', '430a69d897a421da23b624bcb40ad627abc7e441', 'e70a43732b757912deeedad36816c065f0e39c9d', '7837fbaa206929b8f395922b48b1fc9e03378645', 'aa271e70d86fad5d6665d697f8baf9716f1ae5f4', '6e4bcf1b60ab5c61e83421c8dc650372e24e8d30', 'b110ccef81be58563f9927ed0ad381fd3b8c27c3', 'fc5c4013575ce84165f39a48752864b2599f0c09', 'acd6510461b419fa8f9a3c98d79c98237621ea13', '70a919d8d4f794454ad3299c9458ede47c4b0cfd', 'f5d7def3c8a06958825ac8cc56a8f219a9f73e2b', '05f50a6602cb7cceea42722d080b0e378054459d', '660ffbe0d7dddc9ed2a1ab75193ea94c00a90ea8', '1ec27bbb0b7176c71e2135efba83d9c62ffcd8a6', 'f82d4f23a31c5580a3275bf1f4536c4c8f4b8de2', '7a64ba77bf8ad161daed7e83f5952233c8142d41', 'dad0af67ed4efb6a6ff758c3087705e396872f44', '72d8fb1d503e792fe448a0c4e0216016ad60a28a', '93c6919221283bb802f996008416db1cd4789cd9', '9f5149445d435c5f1c724425212459bf27b6c7e4', '11982cc8632a0a0687b5cb67db66cf0d0ac58d3e', '2cbd57694a0f21b2a2ce21bedae36c4f2cd1d729', '81699b8de124cca618159d04df6590497890c08c', '9fa2914e8f12c36281abed5a1c3a7224b6fc53c1', 'c8397cb34898c6a4a868f6ef7270e17f356187ab', '385633c311173e5c0f51b999e95adf65e3ee4e22', '4a020a0d2303f242f4b8204c8d376b05a4d87ff4', '14cb28d68fb5f33eaa28ed14c5739adaed3880bd', '92c226f50fae9c43efc70a46f7ad6a5bcc8d0afd', 'b382adecf09126bde82ab379400163903ee58e08', 'e4edc349285a5300ae514b0d4b8b8c4db0999009', '6d15ffbd57d37baa9461a369f811ca0d96afa621', '73f52bb918789ec16852513b78d302e5b3919f53', 'd6ebf53e5f112c4c8aea537bc3ea715f68dfcb1b', '4ba4e05c0b2bc2cccd575216f26a3bf04761a1a9', 'dd64aa010861c2735bdf54f4759d659f7b46decf', '8a9549962d4ab03b239758aed7e35f751ef2151e', '1ae126ae742ad1a11d5864d97d754427c511fefb', '7b61913195110a2a07ed1ef0fea94499ca5af970', 'f8e04f5367e3af276dca62b0e75b95907d6b8e47', 'b712a9ef26397c93236c51e442ecd21b3b29e1a7', '9701dc3adea0736ecd51cbf3079896550e868865', '032a520bf885d1546678ed1ff5c0e6510d1b14a5', '3d4f58cf724baee4fe738b35eb7f7ddbc2681aea', 'fe81b24c0c778fc14772a208cfec923639ced911', 'a7144fc79211e2cdbcc528cc65ef6da0df5ae8ae', 'e634071a9b93d8e96019ca338326b9236667b525', '2511d2677026be73a4f0712376af93ae3e267bb9', '97dcd263f2de3b332d8a0060f1605f8932151d38', '4eedfc192372cf3da734b3707e2054cc9a9be6e9', '857007023ceb9504f51e9b078476c99f49c7bec6', '0e853ad2dd743960f49da7512b8491bb0f2fb32b', 'd296b3c2a8be2c6ee03aaf8a4f59d0bfcf440520', 'ad1011be223a024fbc2621eba8e2ebf317cf29a6', 'fb6690f09aa527dcd64eb3ccc41f560b5417cf5b', '0ebbabf54ea23f67e597fad03505ea3b618661a8']
LINUS_IDS = ['a97dfd08c1662803a39537e7e42c93bf2db27128', 'd04d508bd126c72cc5eb1b81ab25ef19678060be', '3507a1d944456b02389f26658705675c228d3c1a', '3bd208ad71e484c15efd05cf4e5f5c46d77168c9', '8dc38ed98e552fdf8ed1c2cda11d2bba43c467f0', 'd4db8227a2f6acd9396e2f96a574416cbbf86304', '38d186e8d1752771441f67080ca38409d807c50a', 'a86865bfe5b8f588be22e0f968b86ca05809ca1d', 'e7ee4d5fa7b3eb2fbb1005393cee4d3f8f0aa74f', '49e50779221eb10c41e9309c6613755da56299c8', '07ab1331b0fbe2deb4a8e5349795bfeed7480588', 'f5c9a22d72bffbfb7c3c6690db9a8982ab1c57ac']

negCorpus = []  # Linus
posCorpus = []  # Greg

# Diretorio do script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ler o CSV e separar emails por autor
csv_path = os.path.join(script_dir, "path-to-messages-csv")
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
