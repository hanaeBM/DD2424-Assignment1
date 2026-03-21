import pickle
import numpy as np


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





   

