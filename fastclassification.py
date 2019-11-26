import torch
import random
import numpy as np
from random import shuffle
from collections import Counter
import pandas as pd
import re
import argparse

def Reader(data_name, train=True):
    data = pd.read_csv(data_name, header=None)
    X = data.iloc[:, 1] + ' ' + data.iloc[:, 2]
    X = list(X.to_numpy())
    alphabet_filter = re.compile('\w+')
    corpus = []; bigram = []
    y = data.iloc[:,0]
    for i in range(len(X)):
        X[i] = [j.lower() for j in alphabet_filter.findall(X[i])]
        corpus.extend(X[i])
        temp = [X[i][j] + ' ' + X[i][j+1] for j in range(len(X[i])-1)]
        X[i] += temp
        bigram.extend(temp)
    if train:
        return X, y, corpus, bigram
    else:
        return X, y

def fnv(string, start_index, bucket=2100000, seed=0):
    prime = 16777619
    basis = 2166136261
    hashing = basis + seed
    for cash in string:
        hashing = hashing ^ ord(cash)
        hashing *= prime
    return int(start_index + 1 + (hashing % bucket))

def Layer(X, y, inputMatrix, outputMatrix, mode='train'):
    z1 = torch.mean(inputMatrix[X, :], dim=0, keepdim=True) # 1 * D
    z2 = torch.mm(z1, outputMatrix) # D * C
    e = np.exp(z2)
    softmax = e / torch.sum(e, dim=1, keepdim=True) # 1 * C
    if mode == 'train':
        loss = -np.log(softmax[:, y] + 1e-7)
        softmax_grad = softmax
        softmax_grad[:, y] -= 1
        grad_out = torch.mm(z1.T, softmax_grad)
        grad_emb = torch.mm(softmax_grad, outputMatrix.T)
        return loss, grad_emb, grad_out
    else:
        return torch.argmax(softmax, 1)

def fasttext_trainer(X_train, y_train, corpus, word2ind, dimension=10, learning_rate=0.01, iteration=10000, bucket=2100000):
# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind) + bucket, dimension) / (dimension**0.5) # V * D
    W_out = torch.randn(dimension, 4) / (dimension**0.5) # D * 4
    window_size = 5
    losses=[]; batch = 0
    print('iter: {}'.format(iteration))

    for i in range(iteration):
        #Training fasttext using SGD
        X = [word2ind[j] for j in X_train[batch]]
        y = int(y_train[batch]) - 1
        L, G_emb, G_out = Layer(X, y, W_emb, W_out)
        W_emb[X] -= learning_rate*G_emb
        W_out -= learning_rate*G_out
        losses.append(L.item())
        batch += 1
        if batch == len(X_train):
            print('epoch finished')
            batch = 0

        if i%10000==0:
            avg_loss=sum(losses)/len(losses)
            print("Loss : %f" %(avg_loss,))
            losses=[]

    return W_emb, W_out

def testing(X_test, y_test, word2ind, emb, out):
    answer = []
    acc = 0
    for t in range(len(X_test)):
        X = []
        for j in X_test[t]:
            try:
                X.append(word2ind[j])
            except:
                pass
        y_pred = Layer(X, [], emb, out, mode='test')
        answer.append(y_pred.items())
        if int(y_test[t]) == y_pred + 1:
            acc += 1
    print('test_acc: {}'.format(acc/len(X_test)))
    return answer

def main():
    parser = argparse.ArgumentParser(description='Fast Text Classification')
    args = parser.parse_args()
    
    #Load and tokenize corpus
    print("loading...")
    print("Read the train data and tokenizing")
    X_train, y_train, corpus, bigram = Reader('./ag_news_csv/train.csv', train=True)
    print("Read the test data and tokenizing")
    X_test, y_test = Reader('./ag_news_csv/test.csv', train=False)
    word2ind = {}
    word2ind[" "]=0
    vocabulary = set(corpus)
    i=1
    for word in vocabulary:
        word2ind[word] = i
        i+=1

    print('Hashing the Bigram')
    for words in bigram:
        word2ind[words] = fnv(words, len(word2ind), bucket=210000)

    print("Vocabulary size")
    print(len(word2ind), len(corpus))
    print()

    #Training section
    emb, out = fasttext_trainer(X_train, y_train, corpus, word2ind, dimension=10, learning_rate=0.01, iteration=len(X_train))
    print("1 epoch finished")
    answer = testing(X_test, y_test, word2ind, emb, out)
    
main()