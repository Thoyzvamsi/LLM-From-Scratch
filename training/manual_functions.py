"""
    --- Just Trying to implement the same functions used in nanoGPT manully ---
"""

import numpy as np

class LinearLayer:
    def __init__(self,in_features,out_features):
        # To make them small , multiplying with 0.02
        self.wei = np.random.randn(in_features,out_features) * 0.02
        self.bias = np.zeros(out_features)

    def forward(self,X):
        return np.matmul(X,self.wei) + self.bias
    
class Embedding:
    def __init__(self,vocab_size,n_embd):
        self.embedding_table_weights = np.random.randn(vocab_size,n_embd)
    
    def forward(self,token_ids):
        return self.embedding_table_weights[token_ids]

class LayerNorm:
    # layernorm doesn't stop after the normalization its learns through gemma and beta
    def __init__(self,n_embd,eps=1e-5):
        self.gamma = np.ones(n_embd)
        self.beta = np.zeros(n_embd)
        self.eps = eps
    
    def forward(self,x):
        mean = np.mean(x,axis=-1,keepdims=True)
        var = np.var(x,axis=-1,keepdims=True)
        x_hat = (x - mean)/np.sqrt(var + self.eps)

        return x_hat @ self.gamma + self.beta
    
class MyFunctions:
    def Softmax(x):
        exp = np.exp(x)
        return exp/np.sum(np.exp(x),axis=-1,keepdims=True)
    
    def cross_entropy(probs,target):
        batch_size = probs.shape[0]
        # Gives the probabilities of the correct predictions
        correct_probs = probs[np.arange(batch_size),target]
        loss = -np.log(correct_probs)
        return np.mean(loss)