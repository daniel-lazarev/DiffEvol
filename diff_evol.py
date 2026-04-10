# Computing the transition matrix W
def trans_mat(m,V):     # m = avg mutation rate, V = set of G sequences, each of length L
    import numpy as np
    import scipy
    G = len(V)        # number of sequences
    L = len(V[0])     # length of each sequence (assumed to be the same for all sequences)
    M = np.zeros((G,G))
    
    for i in range(G):
        for j in range(G):
            h = np.count_nonzero(np.array(V[i])!= np.array(V[j]))    # Hamming distance between seq i and j
            M[i][j] =  (L-h)*(np.log(1-m)) + h*np.log(m) 
        
    W = scipy.special.softmax(M, axis=1)      # Transition matrix, G x G
    return W


# Computing the (T x G) constraint matrix after time T
def constraint(freq,W,T):        #freq = frequency matrix, W = transition matrix, T = time point up to which constraint matrix is measured               
    import numpy as np
    G = W.shape[0]
    K = []         # Constraint matrix, N x G. Every row is the constraint function/vector for time n

    for t in range(T):
        K.append( freq[t] / (freq[0] @ np.linalg.matrix_power(W,t))  )

    for t in range(T):        #normalization 
        K[t] = K[t] / np.sum(K[t])
    K = np.reshape(K,(T,G))
    return K

# This is for simulations for which initial freqency f0 does NOT equal freq[0]
# Ignore otherwise and use 'constraint'
# Computing the (T x G) constraint matrix after time T
def constraint2(freq,W,T,f0):        #freq = frequency matrix, W = transition matrix, T = time point up to which constraint matrix is measured
                                            # f0 = frequency at time = 0 (only if different from freq[0]
    import numpy as np
    G = W.shape[0]
    K = []         # Constraint matrix, N x G. Every row is the constraint function/vector for time n

    for t in range(T):
        K.append( freq[t] / (f0 @ np.linalg.matrix_power(W,t))  )

    for t in range(T):        #normalization 
        K[t] = K[t] / np.sum(K[t])
    K = np.reshape(K,(T,G))
    return K

# using a heat kernel to compute the (T x G) constraint matrix after time T 
def constraint_ker(freq,W,T):        #freq = frequency matrix, W = transition matrix, T = time point up to which constraint matrix is measured        
    import numpy as np
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    G = W.shape[0]
    K = []         # Constraint matrix, N x G. Every row is the constraint function/vector for time n

    for t in range(T):
        K.append( freq[t] / (freq[0] @ U@np.exp(t*np.diag(S))@U.T  )  )

    for t in range(T):        #normalization 
        K[t] = K[t] / np.sum(K[t])
    K = np.reshape(K,(T,G))
    return K