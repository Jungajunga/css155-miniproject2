########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#            FOR MINIPROJECT 2
# Author:       Andrew Kang
# Description:  Set 5 solutions
########################################


##### generating emission function
### after getting, learned A,O, we used this HMM to generate.

import random

import numpy as np
import copy

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

def getUniqueWords(sonnets):
    s = set()
    for sonnet in sonnets:
        for sentence in sonnet:
            s |= set([word.lower() for word in sentence])
    return(list(s))


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


sonnet     = loadShakespeareSonnets()    #3-dim array of sonnets
dictionary = getUniqueWords(sonnet)   #list of words appearing in all sonnets

# saving the number of syllables for each word
number = [[[0 for m in range(len(sonnet[k][j]))] for j in range(len(sonnet[k]))] for k in range(154)]

# saving stress states on each syllable.
syllable = [[['' for m in range(len(sonnet[k][j]))] for j in range(len(sonnet[k]))] for k in range(154)]

for k in range(154):
    for j in range(len(sonnet[k])):
        for m in range(len(sonnet[k][j])):
            number[k][j][m] = syllable_count(sonnet[k][j][m])
            syllable[k][j][m] = function(start(k,j,m), number[k][j][m])


def StressToNumber(stress_seq):
    if stress_seq[0] == 'x':
        return 0
    else:
        return 1

# stress_count[w][0] = number of times word w is unstressed
stress_count = np.zeros((3232,2))
for s in range(154):
    for l in range(len(sonnet[s])):
        for w in range(len(sonnet[s][l])):
            stress_count[dictionary.index(sonnet[s][l][w].lower())][StressToNumber(syllable[s][l][w])] += 1

unstressed_prob = [float(p)/float(p+q) for [p,q] in stress_count]






class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state. 

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
                        
            Sonnets:    3-dim array of Shakespeare's sonnets
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]





    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Note that alpha_j(0) is already correct for all j's.
        # Calculate alpha_j(1) for all j's.
        for curr in range(self.L):
            alphas[1][curr] = self.A_start[curr] * self.O[curr][x[0]]

        # Calculate alphas throughout sequence.
        for t in range(1, M):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible previous states to accumulate
                # the probabilities of all paths from the start state to
                # the current state.
                for prev in range(self.L):
                    prob += alphas[t][prev] \
                            * self.A[prev][curr] \
                            * self.O[curr][x[t]]

                # Store the accumulated probability.
                alphas[t + 1][curr] = prob

            if normalize:
                norm = sum(alphas[t + 1])
                for curr in range(self.L):
                    alphas[t + 1][curr] /= norm

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize initial betas.
        for curr in range(self.L):
            betas[-1][curr] = 1

        # Calculate betas throughout sequence.
        for t in range(-1, -M - 1, -1):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible next states to accumulate
                # the probabilities of all paths from the end state to
                # the current state.
                for nxt in range(self.L):
                    if t == -M:
                        prob += betas[t][nxt] \
                                * self.A_start[nxt] \
                                * self.O[nxt][x[t]]

                    else:
                        prob += betas[t][nxt] \
                                * self.A[curr][nxt] \
                                * self.O[nxt][x[t]]

                # Store the accumulated probability.
                betas[t - 1][curr] = prob

            if normalize:
                norm = sum(betas[t - 1])
                for curr in range(self.L):
                    betas[t - 1][curr] /= norm

        return betas

    def unsupervised_learning(self, X, iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        for iteration in range(iters):
            print("Iteration: " + str(iteration))

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    
                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        if t != M:
                            A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][x[t - 1]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][x[t]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]

    def generate_emission(self):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''

        emission = []
        
        rand_var = random.uniform(0, 1)
        next_state = 0
        
        while rand_var > 0 and next_state<len(self.A_start):
            rand_var -= self.A_start[next_state]
            next_state += 1
        
        next_state -= 1
        state       = next_state
        
        num_syll = 0# syllable_count(dictionary)
        
        while num_syll<10:
            repeat = True
            while repeat:
                O_modified = copy.deepcopy(self.O[state])
                for w in range(3232):
                    if num_syll%2 == 0:
                        O_modified[w] *= unstressed_prob[w]
                    elif num_syll%2 == 1:
                        O_modified[w] *= (1-unstressed_prob[w])
                sum_ = sum(O_modified)
                for w in range(3232):
                    O_modified[w] /= sum_
        
                # Sample next observation.
                rand_var = random.uniform(0, 1)
                next_obs = 0

                while rand_var > 0 and next_obs<3232:
                    rand_var -= O_modified[next_obs]
                    next_obs += 1

                next_obs -= 1
                
                if num_syll+syllable_count(dictionary[next_obs]) <= 10:
                    emission.append(next_obs)
                    num_syll += syllable_count(dictionary[next_obs])
                    repeat = False

            # Sample next state.
            rand_var = random.uniform(0, 1)
            next_state = 0

            while rand_var > 0 and next_state<len(self.A[state]):
                rand_var -= self.A[state][next_state]
                next_state += 1

            next_state -= 1
            state = next_state

        return emission


def unsupervised_HMM(X, n_states, n_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, n_iters)

    return HMM




'''A = [[random.uniform(0, 1) for j in range(5)]    for i in range(5)]
O = [[random.uniform(0, 1) for j in range(3232)] for i in range(5)]

print('Reached point 1')

for i in range(5):
    sumA = sum(A[i])
    sumO = sum(O[i])
    for j in range(5):
        A[i][j] /= sumA
    for j in range(3232):
        O[i][j] /= sumO

HMM = HiddenMarkovModel(A, O)

print('Reached point 2')

x = HMM.generate_emission()

print('Reached point 3')

counter = 0
while counter<len(x):
    print(dictionary[x[counter]]),
    print(' ')
    counter += 1'''
