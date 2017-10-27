import os
import xml.etree.ElementTree as ET
from lxml import etree
import unicodedata
import sys
import time
from math import log10
from collections import defaultdict
from string import digits
from scipy import spatial
import json
import pip


tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
remove_digits = str.maketrans('', '', digits)
def remove_punctuation(text):
    return text.translate(tbl)

'''
def remove_characters_after_tokenization(tokens):
 pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
 filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
 return filtered_tokens
'''



dictDocLen = {}
collection = {}
docTFIDF = defaultdict(dict)
countDocs = 0
countFiles = 0
sumLengthDocument = 0

#SS = json.load('HAM2-801011-001.txt')
#with open('TFIDF/HAM2-801011-001.txt', 'r') as f:
 #   data = json.load(f)

#HAMID = defaultdict(dict)
#HAMID['DOC1']['WORD1'] = 100
#HAMID['DOC2']['WORD1'] = 200
#HAMID['DOC2']['WORD2'] = 500

#if 'KI' in HAMID['DOC1']:
 #   print("TRUE")



start_time = time.time()
for DirectoryName in os.listdir('Hamshahri'):
    for FileName in os.listdir('Hamshahri/'+DirectoryName+'/'):
        if FileName.endswith('.xml'):
                countFiles = countFiles + 1
              #  print(FileName)
                parser = etree.XMLParser(recover=True)
                tree = ET.parse('Hamshahri/'+DirectoryName+'/'+FileName, parser)
                root = tree.getroot()
                for DOCS in root.iter('DOC'):
                    countDocs = countDocs + 1
                #     print(countDocs)
                #   for TEXT in root.iter('TEXT'):
                    for TEXT in DOCS.iter('TEXT'):
                # Removing Image Tag from XML
                       DOCID = DOCS.find('DOCID').text
                       rawText = "".join(TEXT.xpath("text()")).strip()
                    # آRemoving Unicode Punctuations
                       rawText = remove_punctuation(rawText)
                    #   print(rawText)
                       rawText = rawText.translate(remove_digits)
                    #   print('-----------------------------------------------------')
                    #   print(rawText)
                    #  tokens = word_tokenize(rawText)
                       tokens = rawText.split()
                       dictDocLen[DOCID] = len(tokens)
                       sumLengthDocument+= len(tokens)
                    # Computing Document Frequency
                       for item in set(tokens):
                          if item in collection:
                             collection[item] += 1
                          else:
                             collection[item] = 1
                  #     docTFIDF = {}
                       for item in tokens:
                          if item in docTFIDF[DOCID]:
                              docTFIDF[DOCID][item] += 1
                          else:
                              docTFIDF[DOCID][item] = 1

                       #p = json.dumps(docTFIDF[DOCID],ensure_ascii=False)
                 #      with open('TFIDF/'+DOCID+'.txt', 'w') as file:
                  #        file.write(json.dumps(docTFIDF,ensure_ascii=False))

print('Number of Documents')
print(countDocs)
print('Number of Files')
print(countFiles)
print('Total Length of Documents')
print(sumLengthDocument)

# Calculating IDF
vocabIDF = {}
for word in collection:
    vocabIDF[word] = log10(countDocs/collection[word])

# Calculating TF-IDF
for DOCID in docTFIDF:
    for item in docTFIDF[DOCID]:
        docTFIDF[DOCID][item] = (1+log10(docTFIDF[DOCID][item]))*vocabIDF[item]




maximum = max(dictDocLen, key=dictDocLen.get)
print('Longest Document ID\t\tLength')
print(maximum, '\t\t', dictDocLen[maximum])


for P in sorted(dictDocLen, key=dictDocLen.get):
    if dictDocLen[P]!=0:
        print('Shortest Document ID\tLength')
        print(P, '\t\t', dictDocLen[P])
        break


# Create Vector

query = 'بازار بزرگ تهران'
queryVector = []
similarityDict = {}
for word in collection:
    if word in query.split():
        queryVector.append(1)
    else:
        queryVector.append(0)

i = 0

for DOCID in docTFIDF:
    docVector = []
    if i % 10 == 0:
        print (i)
    i += 1
    for word in collection:
        if word in docTFIDF[DOCID]:
            docVector.append(docTFIDF[DOCID][word])
        else:
            docVector.append(0)
    cosSim = 1 - spatial.distance.cosine(docVector, queryVector)
    similarityDict[DOCID] = cosSim

i = 0
for P in sorted(similarityDict, key=similarityDict.get, reverse=True):
        print(P, '\t\t', similarityDict[P])
        i += 1
        if i == 20:
           break

print("--- %s seconds ---" % (time.time() - start_time))