### python 3

test1 =pronouncing.phones_for_word("art's")
test2 =pronouncing.phones_for_word('pronounce') 

# function that returns the last phoneme containing the last vowel 
def rhyme_check(word): 
    length = len(word[0])
    start_idx = -1
    i = 1
    while(start_idx == -1):
        # inspecting phoneme backward 
        # if the function detects number which corresponds to a vowel, we save that index 
        if word[0][length - i].isdigit() == True:
            end_idx = length -i
            start_idx = end_idx -1
            # from the saved end index, we go back until we find the start of the corresponding phoneme.
            while((word[0][start_idx] != ' ') & (start_idx != 0) ):
                start_idx = start_idx -1
            # if the detected phoneme is in the first block, you get start_idx 
            # if the detected phonem is not placed at the first place, you get start_idx +1
            if start_idx != 0:
                start_idx = start_idx +1
        i = i + 1
            
    
    return word[0][start_idx: end_idx+1]    
        



##############################################
# function that returns the last phoneme containing the last vowel 
###################################################
def rhyme_check(word): 
    length = len(word[0])
    start_idx = -1
    i = 1
    while(start_idx == -1):
        # inspecting phoneme backward 
        # if the function detects number which corresponds to a vowel, we save that index 
        if word[0][length - i].isdigit() == True:
            end_idx = length -i
            start_idx = end_idx -1
            # from the saved end index, we go back until we find the start of the corresponding phoneme.
            while((word[0][start_idx] != ' ') & (start_idx != 0) ):
                start_idx = start_idx -1
            # if the detected phoneme is in the first block, you get start_idx 
            # if the detected phonem is not placed at the first place, you get start_idx +1
            if start_idx != 0:
                start_idx = start_idx +1
        i = i + 1
            
    
    return word[0][start_idx: end_idx+1]   


####################
# first generate a b c d e f g and save the end-word from each line.
####################
end_word = []
first_poem =[]
count =1
while(len(end_word) != 7):
    
    x=HMM.generate_emission()
    
    idx = x[len(x)-1]
    if pronouncing.phones_for_word(invert[x[len(x)-1]]) != []:
        first_poem.append(x)
        end_word.append(invert[x[len(x)-1]])
    #print(count)
    count +=1

######################
# obtain rhyme  
#####################    
#print(end_word, len(end_word))
rhyme_block =[] 
for i in range(7):
    #print(i)
    end_word_phonemes = pronouncing.phones_for_word(end_word[i])
    #print(end_word_phonemes)
    rhyme_block.append(rhyme_check(end_word_phonemes))
    
#print(rhyme_block)

#############################
# Now add rhyme in the poem.
#############################
second_poem = []
count = 0
while(len(second_poem) != 7):
    
    x=HMM.generate_emission()
    endword = invert[x[len(x)-1]]
    #print(endword)
    step_a = pronouncing.phones_for_word(endword)
    #print(step_a)
    
    if step_a != []:
        step_b = rhyme_check(step_a)
        #print(step_b)
        if step_b == rhyme_block[count]:
            second_poem.append(x)
            count = count +1
            #print(count)
            
            
            
############################
# Now we have complete poem and need to combine them
############################

for i in range(0,3):
    counter = 0 
    while counter <len(first_poem[2*i]):
        print(invert[first_poem[2*i][counter]], end=" ")
        counter += 1
    
    print(" ")

    counter = 0 
    while counter <len(first_poem[2*i+1]):
        print(invert[first_poem[2*i+1][counter]], end=" ")
        counter += 1

    print(" ")
    counter = 0
    while counter <len(second_poem[2*i]):
        print(invert[second_poem[2*i][counter]], end=" ")
        counter += 1
    
    print(" ")

    counter = 0
    while counter <len(second_poem[2*i+1]):
        print(invert[second_poem[2*i+1][counter]], end=" ")
        counter += 1

    print(" ")
    
counter = 0 
while counter <len(first_poem[6]):
    print(invert[first_poem[6][counter]], end=" ")
    counter += 1

print(" ")
counter = 0
while counter <len(second_poem[6]):
    print(invert[second_poem[6][counter]], end=" ")
    counter += 1
    
print(" ")

