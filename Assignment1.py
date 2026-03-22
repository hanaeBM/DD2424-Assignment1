import pickle
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt 

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

# check gradient 

def ComputeGradsWithTorch(X, y, network_params,lam):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    

    cost = loss + lam * torch.sum(torch.multiply(W, W))
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads    

def EqualGrad(grads1,grads2):
    eps=10**(-10)
    diff_W=np.abs(grads1["W"]-grads2["W"])/np.maximum(eps, np.abs(grads1["W"]) + np.abs(grads2["W"]))
    diff_b=np.abs(grads1["b"]-grads2["b"])/np.maximum(eps, np.abs(grads1["b"]) + np.abs(grads2["b"]))

    return np.max(diff_W)<=10**(-6) and np.max(diff_b)<=10**(-6)



# question 8
def ComputeCost(P,y,network,lam):
    loss=ComputeLoss(P,y)
    W=network["W"]
    
    return loss + lam*np.sum(W**2)

def MiniBatchGD(X, Y,y,X_val,y_val,GDparams, init_net, lam,seed=42):
    np.random.seed(seed)
    trained_net = copy.deepcopy(init_net)
    eta=GDparams['eta']
    n_batch=int(GDparams['n_batch'])
    n_epochs=int(GDparams['n_epochs'])
    n=X.shape[1]

    C_train=[]
    L_train=[]
    C_val=[]
    L_val=[]


    for e in range(n_epochs):
        idx=np.random.permutation(n)
        
        # create minibatch
        for j in range(n//n_batch):
            j_start=idx[j*n_batch]

            batch_idx = idx[j*n_batch:(j+1)*n_batch]
            Xbatch = X[:, batch_idx]
            Ybatch = Y[:, batch_idx]

            # Forward Pass
            P=ApplyNetwork(Xbatch,trained_net)
            # Backward Pass
            grads=BackwardPass(Xbatch, Ybatch, P, trained_net, lam)
            # Update
            trained_net["W"]=trained_net["W"]-eta*grads["W"]
            trained_net["b"]=trained_net["b"]-eta*grads["b"]
        # training
        P= ApplyNetwork(X, trained_net)
        c_t=ComputeCost(P,y,trained_net,lam)
        l_t=ComputeLoss(P,y)
        L_train.append(l_t)
        C_train.append(c_t)
        # validation
        P_v= ApplyNetwork(X_val, trained_net)
        c_v=ComputeCost(P_v,y_val,trained_net,lam)
        l_v=ComputeLoss(P_v,y_val)
        L_val.append(l_v)
        C_val.append(c_v)


    #Cost
    plt.plot(C_train,label=" Training Cost")
    plt.plot(C_val,label=" Validation Cost")
    
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title(f"Cost - eta={eta}, lam={lam}")

    plt.legend()
    #plt.show()
    plt.savefig(f"plots/cost_eta{eta}_lam{lam}.png")
    plt.close()

    #Loss
    plt.plot(L_train,label=" Training Loss")
    plt.plot(L_val,label=" Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title(f"Lost - eta={eta}, lam={lam}")
    plt.legend()
    #plt.show()
    plt.savefig(f"plots/lost_eta{eta}_lam{lam}.png")
    plt.close()
    


    return trained_net

def PlotImg(trained_net, eta, lam):
    Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    
    fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        axs[i].imshow(w_im_norm)
        axs[i].axis('off')
    
    plt.savefig(f"plots/weights_eta{eta}_lam{lam}.png")
    plt.close()
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
    lam=0.1
    grads = BackwardPass(trainX[:, 0:100], trainY[:, 0:100], P, init_net, lam)
    print(f"Gradients :{grads}")

    print("Check equal Gradient")
    grads2=ComputeGradsWithTorch(trainX[:, 0:100],train_y[0:100],init_net,lam)

    bool=EqualGrad(grads,grads2)
    print(bool)

    # question 8
    import os
    os.makedirs("plots", exist_ok=True)

    print("Question 8")
    GDparams={'n_batch':100,
              'eta':0.001,
              'n_epochs':20}
    trained_net=MiniBatchGD(trainX, trainY,train_y, valX,val_y,GDparams, init_net, lam=0,seed=42)
     
    configs = [
    {'lam': 0,   'n_epochs': 40, 'n_batch': 100, 'eta': 0.1},
    {'lam': 0,   'n_epochs': 40, 'n_batch': 100, 'eta': 0.001},
    {'lam': 0.1, 'n_epochs': 40, 'n_batch': 100, 'eta': 0.001},
    {'lam': 1,   'n_epochs': 40, 'n_batch': 100, 'eta': 0.001},
]
    
    for config in configs:
        trained_net = MiniBatchGD(trainX, trainY, train_y, valX,val_y,config, init_net, lam=config['lam'], seed=42)
        PlotImg(trained_net, config['eta'], config['lam'])
    







   

