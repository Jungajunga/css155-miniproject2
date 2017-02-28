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

def getUniqueWords(sonnets):
    s = set()
    for sonnet in sonnets:
        for sentence in sonnet:
            s |= set([word.lower() for word in sentence])
    return(list(s))

dictionary = getUniqueWords(sonnet)
#print((dictionary))
# i th state = dictionary[i] word
    
convert = dict(zip(dictionary, [i for i in range(len(dictionary))]))

number_sonnet = [[[0 for m in range(len(sonnet[k][j]))] for j in range(len(sonnet[k]))] for k in range(154)]
for k in range(154):
    for j in range(len(sonnet[k])):
        for m in range(len(sonnet[k][j])):
            number_sonnet[k][j][m] = convert[sonnet[k][j][m].lower()]

def flatten(list):
    return [x for sublist in list for x in sublist]
#print(flatten(number_sonnet))

from HMM_stop import unsupervised_HMM

def unsupervised_learning(n_states, n_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    # Train the HMM.
    HMM = unsupervised_HMM(flatten(number_sonnet), n_states, n_iters)

    # Print the transition matrix.
    print("Transition Matrix:")
    print('#' * 70)
    for i in range(len(HMM.A)):
        print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    print('')
    print('')

    # Print the observation matrix. 
    print("Observation Matrix:  ")
    print('#' * 70)
    for i in range(len(HMM.O)):
        print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    print('')
    print('')
    return(HMM.A,HMM.O)

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Question 2H"))
    print('#' * 70)
    print('')
    print('')

    A,O = unsupervised_learning(15, 0.001)
    print(A,O)


