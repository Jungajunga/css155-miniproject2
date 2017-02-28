import codecs
import datetime
import os
import re

# open the file.

def loadShakespeareSonnets():
    sonnets = []
    with open('shakespeare.txt', 'r') as f:
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

sonnet = loadShakespeareSonnets()

# making the dictionary
def getUniqueWords(sonnets):
    s = set()
    for sonnet in sonnets:
        for sentence in sonnet:
            s |= set([word.lower() for word in sentence])
    return(list(s))

dictionary = getUniqueWords(sonnet)
#print((dictionary))
# i th state = dictionary[i] word
# converting each word into integer to do HMM.
convert = dict(zip(dictionary, [i for i in range(len(dictionary))]))

# converting every input data into integer array. 
number_sonnet = [[[0 for m in range(len(sonnet[k][j]))] for j in range(len(sonnet[k]))] for k in range(154)]
for k in range(154):
    for j in range(len(sonnet[k])):
        for m in range(len(sonnet[k][j])):
            number_sonnet[k][j][m] = convert[sonnet[k][j][m].lower()]

# make sonnet into one list to define meter dictionary
def flatten(list):
    return [x for sublist in list for x in sublist]

# inverse function of convert
invert = dict(zip([i for i in range(len(dictionary))], dictionary))



# measuring the frequency of each word in the total input data
frequency = [0 for i in range(len(dictionary))]
for k in range(154):
    for j in range(len(sonnet[k])):
        for m in range(len(sonnet[k][j])):
            frequency[number_sonnet[k][j][m]] +=1


# counting syllables. 
import pronouncing

# this is the elementary way to syllable counting. Only apply to the case the word is not in the nltk library. 
def easy_syllable_count(text):
    count = 0
    vowels = 'aeiouy'
    text = text.lower().strip(".:;?!)(")
    if text[0] in vowels:
        count += 1
    for index in range(1, len(text)):
        if text[index] in vowels and text[index-1] not in vowels:
            count += 1
    if text.endswith('e'):
        count -= 1
    if text.endswith('le'):
        count += 1
    if text.endswith('es'):
        count -= 1
    if count == 0:
        count += 1
    count = count - (0.1*count)
    return (round(count))


def syllable_count(word):
    test = pronouncing.phones_for_word(word)
    if(test==[]):
        return easy_syllable_count(word)
    else:
        return pronouncing.syllable_count(test[0]) # using nltk library.
        


# determining the stress states if we know the starting stress and the number of syllables.

# this determines the starting state is stress or unstress for each word.
def start(k,j,m):
    save = int(sum(number[k][j][0:m]))
    if (save//2)*2 == save: return 'x'
    else: return '/'



# if you put word starting condition and total length, it will give the stress states.
# if you put '/' and '3', it gives '/x/'
def function(start, number):
    answer = []
    number = int(number)
    if start == '/':
        for i in range(number):
            if (i//2)*2 ==i: answer.append('/')
            else : answer.append('x')
    else:
        for i in range(number):
            if (i//2)*2 ==i: answer.append('x')
            else : answer.append('/')
    x = ''.join([str(xi) for xi in answer])
    return x
    
# saving the number of syllables for each word
number = [[[0 for m in range(len(sonnet[k][j]))] for j in range(len(sonnet[k]))] for k in range(154)]

# saving stress states on each syllable. 
syllable = [[['' for m in range(len(sonnet[k][j]))] for j in range(len(sonnet[k]))] for k in range(154)]

for k in range(154):
    for j in range(len(sonnet[k])):
        for m in range(len(sonnet[k][j])):
            number[k][j][m] = syllable_count(sonnet[k][j][m])
            syllable[k][j][m] = function(start(k,j,m), number[k][j][m])




sonnet_flat = flatten(flatten(sonnet))
stress_flat = flatten(flatten(syllable))
low_sonnet = [x.lower() for x in sonnet_flat]
# if you put a word in sonnet, it will give meter status, '/', 'x', '/x/' etc..
stress_dictionary = dict(zip(low_sonnet, stress_flat))


