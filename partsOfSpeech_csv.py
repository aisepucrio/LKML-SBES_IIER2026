# -*- coding: utf-8 -*-
# Adaptado para usar lkml_200_each.csv
# Analisa partes do discurso (POS tags) para emails de Linus e Greg

import os
import operator
import csv
import sys

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
partsOfSpeechTagsGuideDict = {}

# Carregar o guia de POS tags
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "summaryOfPOSTags")) as fh:
    content = fh.readlines()
    for line in content:
        splitLine = line.split("\t")
        if len(splitLine) >= 3:
            partsOfSpeechTagsGuideDict[splitLine[1]] = splitLine[2].strip()

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

partsofspeechDictLinus = {}
partsofspeechDictGreg = {}

# Processar emails do Linus
print "Processing Linus emails..."
for doc in negCorpus:
    try:
        sentences = nltk.tokenize.sent_tokenize(doc)
        tokens = [nltk.tokenize.word_tokenize(s) for s in sentences]
        pos_linus_tagged_tokens = [nltk.pos_tag(t) for t in tokens]
        
        for sentence in pos_linus_tagged_tokens:
            for posTag in sentence:
                if posTag[1] in partsofspeechDictLinus.keys():
                    partsofspeechDictLinus[posTag[1]] = partsofspeechDictLinus[posTag[1]] + 1
                else:
                    partsofspeechDictLinus[posTag[1]] = 1
    except:
        pass

totalCountOfTermsLinus = 0
print "\n=== LINUS POS TAGS ==="
for tag, numTag in partsofspeechDictLinus.iteritems():
    if tag in partsOfSpeechTagsGuideDict.keys():
        print partsOfSpeechTagsGuideDict[tag], "|", str(numTag)
    else:
        print tag, "|", str(numTag)
    totalCountOfTermsLinus = totalCountOfTermsLinus + numTag
print "Total Number of Terms for Linus:", totalCountOfTermsLinus

# Processar emails do Greg
print "\nProcessing Greg emails..."
for doc in posCorpus:
    try:
        sentences = nltk.tokenize.sent_tokenize(doc)
        tokens = [nltk.tokenize.word_tokenize(s) for s in sentences]
        pos_greg_tagged_tokens = [nltk.pos_tag(t) for t in tokens]
        
        for sentence in pos_greg_tagged_tokens:
            for posTag in sentence:
                if posTag[1] in partsofspeechDictGreg.keys():
                    partsofspeechDictGreg[posTag[1]] = partsofspeechDictGreg[posTag[1]] + 1
                else:
                    partsofspeechDictGreg[posTag[1]] = 1
    except:
        pass

totalCountOfTermsGreg = 0
print "\n=== GREG POS TAGS ==="
for tag, numTag in partsofspeechDictGreg.iteritems():
    if tag in partsOfSpeechTagsGuideDict.keys():
        print partsOfSpeechTagsGuideDict[tag], "|", str(numTag)
    else:
        print tag, "|", str(numTag)
    totalCountOfTermsGreg = totalCountOfTermsGreg + numTag
print "Total Number of Terms for Greg:", totalCountOfTermsGreg

# Salvar resultados em CSV
output_path = os.path.join(script_dir, "results_partsOfSpeech.csv")
with open(output_path, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['POS_Tag', 'Tag_Description', 'Linus_Count', 'Greg_Count', 'Total_Count'])
    
    all_tags = set(partsofspeechDictLinus.keys()) | set(partsofspeechDictGreg.keys())
    for tag in sorted(all_tags):
        linus_count = partsofspeechDictLinus.get(tag, 0)
        greg_count = partsofspeechDictGreg.get(tag, 0)
        description = partsOfSpeechTagsGuideDict.get(tag, tag)
        writer.writerow([tag, description, linus_count, greg_count, linus_count + greg_count])
    
    writer.writerow(['TOTAL', 'All Tags', totalCountOfTermsLinus, totalCountOfTermsGreg, totalCountOfTermsLinus + totalCountOfTermsGreg])

print "\nResults saved to:", output_path
