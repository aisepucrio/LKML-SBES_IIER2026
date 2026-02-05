# -*- coding: utf-8 -*-
# Adaptado para usar lkml_200_each.csv
# Conta palavroes (expletives) nos emails de Linus e Greg

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

# IDs dos autores - multiplos IDs
GREG_IDS = ['2511d2677026be73a4f0712376af93ae3e267bb9', '6d15ffbd57d37baa9461a369f811ca0d96afa621', 
            '857007023ceb9504f51e9b078476c99f49c7bec6', '93c6919221283bb802f996008416db1cd4789cd9', 
            'b712a9ef26397c93236c51e442ecd21b3b29e1a7', 'e47776b220235deb3b58b762e79e03082cba5b68', 
            'f5d7def3c8a06958825ac8cc56a8f219a9f73e2b']
LINUS_IDS = ['062aad6a1a302c7688ef109864b5c8c0f59ebcbd', '269f83821d924a5a87f6e23d983ca65a1f127531', 
             '38d186e8d1752771441f67080ca38409d807c50a', '8dc38ed98e552fdf8ed1c2cda11d2bba43c467f0', 
             '9d7c2b0cfc3382dc2cbafed83b632890eeaa663e', 'a86865bfe5b8f588be22e0f968b86ca05809ca1d', 
             'a97dfd08c1662803a39537e7e42c93bf2db27128', 'e7ee4d5fa7b3eb2fbb1005393cee4d3f8f0aa74f']

myList = []
negCorpus = []  # Linus
posCorpus = []  # Greg
posCorpusCounts = []
negCorpusCounts = []

# Diretorio do script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Carregar lista de palavroes do Google
with open(os.path.join(script_dir, "googleListOfExpletives")) as fh:
    contents = fh.read()
    contents = contents.lower()
    for word in contents.split():
        myList.append(word)

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

# Contar palavroes do Linus
linusExpletiveDict = {}
for doc in negCorpus:
    count = 0
    for word in doc.split():
        if len(word) > 1 and word in myList and word not in ['hit', 'job', 'cum', 'cox', 'pisses', 'willy', 'knob', 'carpet', 'fudge']:
            count = count + 1
            if word in linusExpletiveDict.keys():
                linusExpletiveDict[word] = linusExpletiveDict[word] + 1
            else:
                linusExpletiveDict[word] = 1
    negCorpusCounts.append(float(count))
            
sorted_bad_linus = sorted(linusExpletiveDict.items(), key=operator.itemgetter(1), reverse=True)

# Contar palavroes do Greg
gregExpletiveDict = {}
for doc in posCorpus:
    count = 0
    for word in doc.split():
        #if  len(word)>1 and word in myList and word not in ['hit','job','cum','cox','pisses','willy','knob','carpet','fudge'] :
        if word in myList:
            count = count + 1
            if word in gregExpletiveDict.keys():
                gregExpletiveDict[word] = gregExpletiveDict[word] + 1
            else:
                gregExpletiveDict[word] = 1
    posCorpusCounts.append(float(count))

sorted_bad_greg = sorted(gregExpletiveDict.items(), key=operator.itemgetter(1), reverse=True)

print
print "=" * 60
print "Linus expletives"
print "=" * 60
for item, count in sorted_bad_linus:
    print item + "," + str(count)
    
print
print "=" * 60
print "Greg expletives"
print "=" * 60
for item, count in sorted_bad_greg:
    print item + "," + str(count)

# Salvar resultados em CSV
output_path = os.path.join(script_dir, "results_expletives.csv")
with open(output_path, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Expletive', 'Linus_Count', 'Greg_Count', 'Total_Count'])
    
    all_words = set(linusExpletiveDict.keys()) | set(gregExpletiveDict.keys())
    for word in sorted(all_words):
        linus_count = linusExpletiveDict.get(word, 0)
        greg_count = gregExpletiveDict.get(word, 0)
        writer.writerow([word, linus_count, greg_count, linus_count + greg_count])

print "\nResults saved to:", output_path
    

#min_max_scaler = preprocessing.MinMaxScaler()
#negCorpusBadWordsNormalized = min_max_scaler.fit_transform(negCorpusCounts)
#posCorpusBadWordsNormalized = min_max_scaler.fit_transform(posCorpusCounts)
