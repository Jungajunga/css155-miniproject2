
# coding: utf-8

# In[2]:

import csv 
import numpy as np
import array
import nltk

from nltk.tag import pos_tag, map_tag

import copy      #tags = copy.copy(mySonnets) not working


import codecs
import datetime
import os
import re

from HMM2 import supervised_HMM


def loadShakespeareSonnets():
    sonnets = []
    with open('/Users/danielsiebel/Desktop/(CS:CNS:EE 155) Machine Learning & Data Mining/MiniProject2/project2data/shakespeare.txt', 'r') as f:
        sonnet = []
        sonnetToAppend = False
        for line in f:
            if line.strip().split(' ')[-1].isdigit():
                sonnetToAppend = True
                continue
            if line == '\n':
                if sonnetToAppend:
                    sonnets.append(sonnet)
                    sonnet = []
                    sonnetToAppend = False
                continue
            sonnet.append([re.sub(r'[^\w\s\']', '', w) for
                           w in line.strip().split(' ')])
        sonnets.append(sonnet)
    f.close()
    return sonnets

def getUniqueWords(sonnets):
    s = set()
    for sonnet in sonnets:
        for sentence in sonnet:
            s |= set([word.lower() for word in sentence])
    return(list(s))

# converts POS-tag to number between 0 and 11
def TagToNumber(tag):
    if tag == u'ADJ':
        return 0
    elif tag == u'ADP':
        return 1
    elif tag == u'ADV':
        return 2
    elif tag == u'CONJ':
        return 3
    elif tag == u'DET':
        return 4
    elif tag == u'NOUN':
        return 5
    elif tag == u'NUM':
        return 6
    elif tag == u'PRT':
        return 7
    elif tag == u'PRON':
        return 8
    elif tag == u'VERB':
        return 9
    elif tag == u'X':
        return 10
    else:
        return 11

mySonnets  = loadShakespeareSonnets()    #3-dim array of sonnets
dictionary = getUniqueWords(mySonnets)   #list of words appearing in all sonnets

#slw[s][l] = number of words in line l of sonnet s
# slw just a help for looping through sonnets
slw = []
for s in range(154):
    lw       = []
    son      = np.asarray(mySonnets[s])
    no_lines = son.size   
    for l in range(no_lines):
        line     = np.asarray(mySonnets[s][l])
        lw.append(line.size)   
    slw.append(np.asarray(lw))
slw = np.asarray(slw)



#just initialization; later we'll have tags[s][l][w] = POS-tag of word maSonnets[s][l][w]
tags = copy.deepcopy(mySonnets)
#finds tags using nltk.pos_tag and map_tag('en-ptb', 'universal', -)
for s in range(154):       
    for l in range(slw[s].size):
        line = np.asarray(mySonnets[s][l])
        words_and_tags = nltk.pos_tag(line)
        for w in range(slw[s][l]):
            tags[s][l][w] = words_and_tags[w][1]
            #simplified = [(word, simplify_wsj_tag(tag)) for word, tag in tagged_sent]
            tags[s][l][w] = map_tag('en-ptb', 'universal', tags[s][l][w])
   

#tag_count[w][t] = number of how often word w has label t  
tag_count = np.zeros((3232,12))
for s in range(154):
    for l in range(slw[s].size):
        for w in range(slw[s][l]):
            tag_count[dictionary.index(mySonnets[s][l][w].lower())][TagToNumber(tags[s][l][w])] += 1            

            
#tag_count_total[t] = total number of occurences of tag t
tag_count_total = np.zeros(12)
for w in range(3232):
    for t in range(12):
        tag_count_total[t] += tag_count[w][t]

print('Total tag count: ')        
print(tag_count_total)
print('')

# X = list of all lines in all 154 sonnets; X[i] list again of indices representing words
# Y = list of all tags belonging to the lines; i.e. Y[l][w] = POS-tag of X[l][w]
# X,Y could be used as input for supervised_learning
X = []
Y = []
for s in range(154):
    for l in range(slw[s].size):
    
        encoded_line = []
        encoded_tags = []
        
        for w in range(slw[s][l]):
            encoded_line.append(dictionary.index(mySonnets[s][l][w].lower()))
            encoded_tags.append(TagToNumber(tags[s][l][w]))
            
        X.append(encoded_line)
        Y.append(encoded_tags)



# Train the HMM.
print('start supervised training')
HMM = supervised_HMM(X, Y)
print('supervised training done: start generating')
x = HMM.generate_emission(20)
print('done generating')
print(x)


# In[ ]:




# In[ ]:



