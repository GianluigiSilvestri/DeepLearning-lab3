import numpy as np

class ConvNet():
    def __init__(self):
        '''Initially the convnet will be empty. Filters and fully connected layers can be added through methods'''
        self.F_layers=[]#list of convolutional layers
        self.W_layers=[]#list of fully connected layers

    def F(self,input_size,f_width,f_num):
        '''
        Function to add a convolutional layer.
        :param input_size: size of the input matrix , (d,n_len) for first conv layer
        :param f_width: width of the filters (in the assignment we use filters of same height as input)
        :param f_num: number of filters applied at this level
        :return:
        '''

        if not self.F_layers:
            self.F_layers.append(np.random.normal(0,0.001,(input_size,f_width,f_num))) #random init todo: check better ones
        else:
            self.F_layers.append(np.random.normal(0,np.sqrt(2/input_size),(input_size,f_width,f_num))) #initialize with HE

    def W(self,output_size,input_size):
        '''
        Function to add a fully connected layer
        :param output_size: dimension of outputs
        :param input_size: dimension of inputs
        :return:
        '''
        self.W_layers.append(np.random.normal(0, np.sqrt(2 / input_size), (output_size,input_size))) # HE initialization


def evaluate_classifier(X_batch, MFs, W):
    '''

    :param X_batch: dimension of input batch
    :param MFs: list of MF matrices
    :param W: Weights for fully connected layer
    :return: output probabilities P, output for each convolutional layer
    '''
    X_batch_1 = np.maximum(np.dot(MFs[0], X_batch), 0) # Assume ReLu activation
    X_batch_2 = np.maximum(np.dot(MFs[1], X_batch_1), 0)
    S = np.dot(W, X_batch_2)
    P = softmax(S)
    return P,X_batch_1,X_batch_2

def compute_loss(X_batch, Y_batch,MFs, W):
    '''

    :param X_batch: dimension of input batch
    :param Y_batch: output corresponding to the input batch
    :param MFs: list of MF matrices
    :param W: Weights for fully connected layer
    :return: loss value
    '''
    N = np.shape(X_batch)[1]
    P , X_batch_1,X_batch_2= evaluate_classifier(X_batch,MFs, W)
    # The loss function is computed by element-wise product between Y and P, and summing column by column
    return np.sum(-np.log(np.sum(Y_batch * P, axis=0))) / N

def make_MF_Matrix(F,n_len):
    '''
    Function to transform the weights in order to compute the convolution as dot product
    :param F: weights for convolutional layer
    :param n_len: length of the longest word
    :return: result Matrix
    '''
    f_rows=np.shape(F)[0]
    f_cols=np.shape(F)[1]
    n_filters=np.shape(F)[2]
    MF=np.zeros(((n_len-f_cols+1)*n_filters,n_len*f_rows))
    for i in range(n_len-f_cols+1):
        for filter in range(n_filters):
            MF[(i*n_filters)+filter,f_rows*i:f_rows*f_cols+f_rows*i]=F[:,:,filter].flatten('F')

    return MF

def make_MX_Matrix(x_input,n_len,f_rows,f_cols,n_filters):
    '''
    Function to transform one Input into matrix to compute the convolution as dot product
    :param x_input: input as vector
    :param n_len: length of the longest word
    :param f_rows: rows of the filter matrix
    :param f_cols: coulmns of the filter matrix
    :param n_filters: number of filters
    :return: result Matrix
    '''

    MX=np.zeros((((n_len-f_cols+1)*n_filters),f_cols*f_rows*n_filters))
    X=np.zeros(f_cols*f_rows)
    for i in range(n_len-f_cols+1):
        for fc in range(f_cols):
            X[f_rows*fc:f_rows*(fc+1)]=x_input[:,i+fc]
        for filter in range(n_filters):
            MX[i*n_filters+filter,filter*f_cols*f_rows:(filter+1)*f_cols*f_rows]=X
    return MX


def read_names_and_labels(file_name):
    '''
    Read the input file.
    :param file_name: file containing labelled surnames
    :return: matrix containing one_hot vectorized words, matrix of one_hot labels, vector of labels,
     length of longest name, number of characters.
    '''
    all_names=[]
    labels=[]
    characters=[]
    n_len=0
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

            surname=surname.lower()
            all_names.append(surname)
            if len(surname)>n_len:
                n_len=len(surname)
            for character in surname:
                if not character in characters:
                    characters.append(character)
            line = fp.readline()
        characters=sorted(characters)

    return names_to_onehot(all_names,characters,n_len),labels_to_onehot(labels),np.array(labels),n_len,len(characters)

def softmax(S):
    return np.exp(S)/np.sum(np.exp(S))

def labels_to_onehot(labels):
    '''
    Take as input the vector of labels and return the
    :param labels: vector of labels
    :return: one_hot matrix of dimensions (K,N), K=number of classes, N=number of labeled inputs
    '''

    len=np.max(labels)
    Y=np.zeros((len,np.size(labels)),dtype=int)
    for i in range(np.size(labels)):
        Y[labels[i]-1,i]=1
    return Y

def names_to_onehot(all_names,characters,n_len):
    '''
    Transform all the names in a one_hot matrix
    :param all_names: list of all the names
    :param characters: list of all possible characters in the names
    :param n_len: length of the longest name
    :return: one_hot matrix of all the names, each column is a vector of one_hot words
    '''
    X=np.zeros((len(characters)*n_len,len(all_names)),dtype=int)
    for i,name in enumerate(all_names):
        X[:,i]=word_to_onehot(name,characters,n_len)

    return X


def word_to_onehot(word,characters,n_len):
    '''
    transform a word into its one_hot representation. If shorter than n_len, fill with all zeroes letters
    :param word: the input word
    :param characters: the list containing all the possible characters in the words
    :param n_len: lenght of the longest word
    :return: the one_hot word
    '''
    oh_word=np.zeros((len(characters),n_len),dtype=int)
    for i,ch in enumerate(word):
        oh_word[characters.index(ch),i]=1
    return oh_word.flatten('F')


### READING DATA
X,Y,y,n_len,d=read_names_and_labels('ascii_names.txt')# n_len: length of longest word, d: lenght of each one_hot letter vector
with open('Validation_Inds.txt') as fp:
    valid=np.fromstring(fp.read(),dtype=int,sep=' ')

### SPLIT INTO TRAINING AND VALIDATION
X_valid=X[:,valid]
Y_valid=Y[:,valid]
y_valid=y[valid]
X=np.delete(X,valid,1)
Y=np.delete(Y,valid,1)
y=np.delete(y,valid)

K=np.shape(Y)[0] #number of classes

## HYPERPARAMETERS ##
n_1=5 # number of filters for layer 1
k_1=5 # width of filters for layer 1
n_2=5 # number of filters for layer 2
k_2=5 # width of filters for layer 2
eta=0.001 # learning rate
rho=0.9 # momentum rate

n_len_1=n_len-k_1+1 # size of each output from layer 1
n_len_2=n_len_1-k_2+1 # size of each output from layer 2

## INITIALIZE NETWORK ##
cNet=ConvNet()
cNet.F(d,k_1,n_1)
cNet.F(n_1,k_2,n_2)
cNet.W(K,n_2*n_len_2)

MF1=make_MF_Matrix(cNet.F_layers[0],n_len)
MF2=make_MF_Matrix(cNet.F_layers[1],n_len_1)
x_input=X[:,0].reshape(n_len,d).T
mx=make_MX_Matrix(x_input,n_len,d,k_1,n_1)

loss=compute_loss(X,Y,[MF1,MF2],cNet.W_layers[0])
a=0
