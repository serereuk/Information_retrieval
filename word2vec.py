import torch
import random
import numpy as np
from random import shuffle
from collections import Counter
from huffman import HuffmanCoding
import argparse


def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)

def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix, mode2):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    z1 = inputMatrix[centerWord, :].view(1, -1) # 1 X D
    z2 = torch.mm(z1, outputMatrix.T) # (1 X D) * (D X V)
    e = np.exp(z2)
    if mode2 == "HS" or mode2 == "NS":
        sigmoid = e / (e + 1)
    else:
        softmax = e / (torch.sum(e, dim=1, keepdim = True) + 1e-7)
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################
    if mode2 =="HS":
        loss = 0
        sigmoid_grad = sigmoid
        for i in range(len(contextWord)):
            if contextWord[i] > 0:
                loss -= np.log(sigmoid_grad[:, i])
                sigmoid_grad[:, i] -= 1
            else:
                loss -= np.log(1 - sigmoid_grad[:, i])
        grad_out = torch.mm(sigmoid_grad.view(-1, 1), z1)
        grad_emb = torch.mm(sigmoid_grad, outputMatrix) 

    elif mode2 == "NS":
        sigmoid_grad = sigmoid
        loss = torch.sum(-np.log(sigmoid_grad[:, :(outputMatrix.shape[0]-2)] + 1e-7), axis=1) - np.log(1-sigmoid_grad[:, outputMatrix.shape[0]-1] + 1e-7)
        sigmoid_grad[:, outputMatrix.shape[0]-1] -= 1
        grad_out = torch.mm(sigmoid_grad.view(-1, 1), z1)
        grad_emb = torch.mm(sigmoid_grad, outputMatrix)

    else:
        loss = -np.log(softmax[:, contextWord] + 1e-7)
        softmax_grad = softmax
        softmax_grad[:, contextWord] -= 1
        grad_out = torch.mm(softmax_grad.view(-1, 1), z1)   # ((V X 1) * (1 * D))
        grad_emb = torch.mm(softmax_grad, outputMatrix)  # 1XV * VXD
    
    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix, mode2):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    z1 = torch.sum(inputMatrix[contextWords, :], dim=0, keepdim=True).view(1, -1) # 1 X D
    z2 = torch.mm(z1, outputMatrix.T) # 1 X D * D X V
    e = np.exp(z2)
    if mode2 == "HS" or mode2 == "NS":
        sigmoid = e / (e+1)
    else:
        softmax = e / (torch.sum(e, dim=1, keepdim=True))
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################
    if mode2 =="HS":
        loss = 0
        sigmoid_grad = sigmoid
        for i in range(len(centerWord)):
            if centerWord[i] > 0:
                loss -= np.log(sigmoid_grad[:, i])
                sigmoid_grad[:, i] -= 1
            else:
                loss -= np.log(1 - sigmoid_grad[:, i])
        grad_out = torch.mm(sigmoid_grad.view(-1, 1), z1)
        grad_emb = torch.mm(sigmoid_grad, outputMatrix) /10

    elif mode2 == "NS":
        sigmoid_grad = sigmoid
        loss = torch.sum(-np.log(1-sigmoid_grad[:, :(outputMatrix.shape[0]-2)]), axis=1) - np.log(sigmoid_grad[:, outputMatrix.shape[0]-1])
        sigmoid_grad[:, outputMatrix.shape[0]-1] -= 1
        grad_out = torch.mm(sigmoid_grad.view(-1, 1), z1)
        grad_emb = torch.mm(sigmoid_grad, outputMatrix) /10
        
    else:
        loss = -np.log(softmax[:, centerWord] + 1e-7)
        softmax_grad = softmax
        softmax_grad[:, centerWord] -= 1
        grad_out = torch.mm(softmax_grad.view(-1, 1), z1)   # ((V X 1) * (1 * D))
        grad_emb = torch.mm(softmax_grad, outputMatrix)  # 1XV * VXD

    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2ind, mode="CBOW", mode2="HS", mode3=None, dimension=100, learning_rate=0.025, iteration=50000):

# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    window_size = 5
    if mode2 == "HS":
        word_code, code_index = HuffmanCoding().build(Counter(corpus))
    elif mode2 == "NS":
        temp = np.array(list(Counter(corpus).values()))
        prob = temp ** (3/4) / sum(temp ** (3/4))
        table = []
        for index,pr in enumerate(prob):
            table.extend([index] * int(pr * 1e7))
        table = np.array(table)

    if mode3 == "SB":
        freq = Counter(corpus)
        t = 1e-5
        for word in corpus:
            prob = 1-np.sqrt(t/freq[word])
            if np.random.random() < prob:
                corpus.remove(word)
    
    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)

        if mode2 == "HS" and mode == "CBOW":
            code = word_code[centerword]
            centerInd = [list(code_index[code[:j]])[0] if code[j] == '0' else -list(code_index[code[:j]])[0] for j in range(len(code))]
        else:
            centerInd = word2ind[centerword]

        if mode2 == "HS" and mode == "SG":
            contextInds = []
            for ii in context:
                code = word_code[ii]
                temp = [list(code_index[code[:j]])[0] if code[j] == '0' else -list(code_index[code[:j]])[0] for j in range(len(code))]
                contextInds.append(temp)
        else:
            contextInds = [word2ind[i] for i in context]
        
        if mode=="CBOW":
            if mode2 == "HS":
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[np.abs(centerInd), :], mode2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out[np.abs(centerInd), :] -= learning_rate*G_out
                losses.append(L.item())

            elif mode2 == "NS":
                temp = []
                while True:
                    if len(temp) == 5:
                        break
                    rand = np.random.randint(0, len(table), 1)
                    if table[rand] not in temp and table[rand] != centerInd:
                        temp.append(table[rand].squeeze())
                list_k = np.array(temp + [centerInd])
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[list_k, :], mode2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out[list_k, :] -= learning_rate*G_out
                losses.append(L.item())

            else:
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out, mode2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out -= learning_rate*G_out
                losses.append(L.item())
            

        elif mode=="SG":
            for contextInd in contextInds:
                if mode2 == "HS":
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[np.abs(contextInd),:], mode2)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out[np.abs(contextInd),:] -= learning_rate*G_out
                    losses.append(L.item())
                elif mode2 == "NS":
                    temp = []
                    while True:
                        if len(temp) == 5:
                            break
                        rand = np.random.randint(0, len(table), 1)
                        if table[rand] not in temp and table[rand] != contextInd:
                            temp.append(table[rand].squeeze())
                    list_k = np.array(temp + [contextInd])
                    L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[list_k, :], mode2)
                    W_emb[contextInds] -= learning_rate*G_emb
                    W_out[list_k, :] -= learning_rate*G_out
                    losses.append(L.item())
                else:
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out, mode2)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out -= learning_rate*G_out
                    losses.append(L.item())

        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
            avg_loss=sum(losses)/len(losses)
            print("Loss : %f" %(avg_loss,))
            losses=[]

    return W_emb, W_out


def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('mode2', metavar='mode2', type=str,
                        help='"HS" for Hierarchical Softmax, "NS" for Negative Sampling')
    parser.add_argument('mode3', metavar='mode3', type=str, help='"SB" for subsampling, default is None')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    mode2 = args.mode2
    mode3 = args.mode3

    #Load and tokenize corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus)
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>4:
            processed.append(word)

    vocabulary = set(processed)

    #Assign an index number to a word
    word2ind = {}
    word2ind[" "]=0
    i = 1
    for word in vocabulary:
        word2ind[word] = i
        i+=1
    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Vocabulary size")
    print(len(word2ind))
    print()

    #Training section
    emb,_ = word2vec_trainer(processed, word2ind, mode=mode, mode2=mode2, mode3=mode3, dimension=300, learning_rate=0.02, iteration=150000)
    
    #Print similar words
    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    for tw in testwords:
        sim(tw,word2ind,ind2word,emb)

main()