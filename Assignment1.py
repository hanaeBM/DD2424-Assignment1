import pickle
import numpy as np

# EXERCICE 1

# question 1

def LoadBatch(filename):
    
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    X = data[b'data'].astype(np.float64) / 255.0   
    X = X.T                                         

    y = data[b'labels']                              

    K = 10
    n = len(y)
    Y = np.zeros((K, n), dtype=np.float64)
    Y[y, np.arange(n)] = 1.0                         

    return X, Y, y

# question 2


def Normalize(trainX,valX,testX):
    d=trainX.shape[0]
    mean_X = np.mean(trainX, axis=1).reshape(d, 1)   
    std_X  = np.std(trainX, axis=1).reshape(d, 1)    

    trainX = (trainX - mean_X) / std_X
    valX   = (valX   - mean_X) / std_X
    testX  = (testX  - mean_X) / std_X
    print(f"Mean trainig data= {mean_X}")
    print(f"Standard Deviation trainig data= {std_X}")
    print("Preprocess Done")

    return trainX, valX, testX

# question 3 : in main

# question 4

def Softmax(s):
    return np.exp(s)/(np.sum(np.exp(s), axis=0, keepdims=True))

def ApplyNetwork(X,network):
    W=network['W']
    b=network['b']

    K=W.shape[0]
    n=W.shape[1]

    s=W@X +b
    P=Softmax(s)
    

    return P
        

# question 5

def ComputeLoss(P,y):

    D=P.shape[1]
    n=P.shape[1]
    y = np.array(y, dtype=int)
    correct_p = P[y, np.arange(n)]
    
    l_cross=-np.log(correct_p)
    L=np.sum(l_cross)/D


    return L


# question 6

def ComputeAccuracy(P,y):
    predicted_y=np.argmax(P,axis=0)

    acc=np.sum(y==predicted_y)
    return acc/P.shape[1]

# question 7

def BackwardPass(X, Y, P, network, lam):
    grads={}
    G_batch=-(Y-P)
    n=X.shape[1]
    In=np.ones((n,1))

    grads["W"]= G_batch@X.T/n +2*lam*network["W"]
    grads["b"]=G_batch@In/n

    return grads 

# question 8

# EXERCICE 2



    


    


# main Top Level

if __name__ == '__main__':

    # question 1 
    cifar_dir = 'data/cifar-10-batches-py/'

    trainX, trainY, train_y = LoadBatch(cifar_dir + 'data_batch_1')
    valX,   valY,   val_y   = LoadBatch(cifar_dir + 'data_batch_2')
    testX,  testY,  test_y  = LoadBatch(cifar_dir + 'test_batch')

    print("Question 1")
    print("Training data:   ", trainX.shape, trainY.shape, len(train_y))
    print("Validation data: ", valX.shape,   valY.shape,   len(val_y))
    print("Test data:       ", testX.shape,  testY.shape,  len(test_y))

    # question 2 
    print("Question 2")
    trainX, valX, testX=Normalize(trainX,valX,testX)

    # question 3
    print("Question 3")
    d=trainX.shape[0]
    K= 10 # nb class
    rng = np.random.default_rng()
    
    BitGen = type(rng.bit_generator) # get the BitGenerator used by default_rng
    
    seed = 42 # use the state from a fresh bit generator
    
    rng.bit_generator.state = BitGen(seed).state
    init_net = {}
    init_net['W'] = .01*rng.standard_normal(size = (K, d))
    init_net['b'] = np.zeros((K, 1))

    print(f"W={init_net['W']}")
    print(f"b={init_net['b']}")
    print("W and b initialized")

    # question 4
    print("Question 4")
    P = ApplyNetwork(trainX[:, 0:100], init_net)
    print("P computed")

    # question 5
    print("Question 5")
    L=ComputeLoss(P,train_y[0:100])

    print(f"mean cross-entropy loss = {L}")

    # question 6
    print("Question 6")
    Acc=ComputeAccuracy(P,train_y[0:100])
    print(f"Accuracy= {Acc}")

    # question 7
    print("Question 7")
    lam=0
    grads=BackwardPass(trainX[:, 0:100],train_y[0:100],P,init_net,lam)
    print(f"Gradients :{grads}")

    # question 8
    print("Question 8")







   

