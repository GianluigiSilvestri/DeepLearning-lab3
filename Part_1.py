import numpy as np

def read_names_and_labels(file_name):
    all_names=[]
    labels=[]
    with open(file_name) as fp:
        line = fp.readline()
        while line:
            surname=""
            for word in line.split():
                if not word.isdigit():
                    if surname=="":
                        surname+=word
                    else:
                        surname+=" "
                        surname += word
                else:
                    labels.append(int(word))

            all_names.append(surname.lower())
            line = fp.readline()

    return names_to_onehot(all_names),labels

def names_to_onehot(all_names):
    ascii_names=[]
    min_ch=127###maximum possible ascii value
    max_lenght=0 ###We need this for the onehot matrix dimension
    max_ch=0 #this too
    for name in all_names:
        len_n=len(name)
        if len_n>max_lenght:
            max_lenght=len_n
        ascii_name=[ord(ch) for ch in name]
        min_in_name=min(x for x in ascii_name)
        if min_in_name<min_ch:
            min_ch=min_in_name
        ascii_names.append(np.array(ascii_name))

    ###Transform words --> from 0 to max value, so we can transform to one-hot
    for i in range(len(ascii_names)):
        ascii_names[i]-=min_ch
        max_n=np.max(ascii_names[i])
        if max_n>max_ch:
            max_ch=max_n

    input=word_to_onehot(ascii_names[0],max_ch,max_lenght)
    a=0

def word_to_onehot(word,n_ch,max_lenght):
    '''Function that given a word made of integers returns the vectorizes one hot representation
        n_ch --> is the length of each onehot vector representing a character
        max_lenght --> is the lenght of the longest word, we append zero vectors at the end to fill the shortest words
    '''
    oh_word=np.zeros((n_ch,max_lenght),dtype=int)
    for i,ch in enumerate(word):
        oh_word[ch,i]=1
    return oh_word.flatten('F').reshape(-1,1)

read_names_and_labels('ascii_names.txt')