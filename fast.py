import torch
import random
import numpy as np
from random import shuffle
from collections import Counter
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

def slicing(data, i_list=[3, 4, 5, 6]):
    temp = []
    data = '<' + data + '>'
    for i in i_list:
        temp += [data[:i-1]] + [data[j:j+i] for j in range(len(data)-i+1)] + [data[len(data)-i+1:]] 
    return temp

def fnv(string, start_index, bucket=2100000, seed=0):
    prime = 16777619
    basis = 2166136261
    hashing = basis + seed
    for cash in string:
        hashing = hashing ^ ord(cash)
        hashing *= prime
    return int(start_index + 1 + (hashing % bucket))

def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix, mode2):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    #print(inputMatrix[centerWord, :].shape)
    z1 = torch.sum(inputMatrix[centerWord, :], dim = 0, keepdim=True).view(1, -1) # 1 X D
    z2 = torch.mm(z1, outputMatrix.T) # (1 X D) * (D X V)
    e = np.exp(z2)
    if mode2 == "NS":
        sigmoid = e / (1 + e)
    else:
        softmax = e / (torch.sum(e, dim=1, keepdim = True) + 1e-7)
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################
    if mode2 == "NS":
        sigmoid_grad = sigmoid
        loss = torch.sum(-np.log(1-sigmoid_grad[:, :(outputMatrix.shape[0]-2)] + 1e-7), axis=1) - np.log(sigmoid_grad[:, outputMatrix.shape[0]-1] + 1e-7)
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
    if mode2 == "NS":
        sigmoid = e / (e+1)
    else:
        softmax = e / (torch.sum(e, dim=1, keepdim=True))
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################
    
    if mode2 == "NS":
        sigmoid_grad = sigmoid
        loss = torch.sum(-np.log(1-sigmoid_grad[:, :(outputMatrix.shape[0]-2)] + 1e-7), axis=1) - np.log(sigmoid_grad[:, outputMatrix.shape[0]-1] + 1e-7)
        sigmoid_grad[:, outputMatrix.shape[0]-1] -= 1
        grad_out = torch.mm(sigmoid_grad.view(-1, 1), z1)
        grad_emb = torch.mm(sigmoid_grad, outputMatrix) /5
        
    else:
        loss = -np.log(softmax[:, centerWord] + 1e-7)
        softmax_grad = softmax
        softmax_grad[:, centerWord] -= 1
        grad_out = torch.mm(softmax_grad.view(-1, 1), z1)   # ((V X 1) * (1 * D))
        grad_emb = torch.mm(softmax_grad, outputMatrix)  # 1XV * VXD

    return loss, grad_emb, grad_out


def fasttext_trainer(corpus, word2ind, mode="SG", mode2="NS", mode3=None, dimension=300, learning_rate=0.05, iteration=10000, bucket=2100000):

# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind) + bucket, dimension) / (dimension**0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    window_size = 5
    
    if mode2 == "NS":
        temp = np.array(list(Counter(corpus).values()))
        prob = temp ** (3/4) / sum(temp ** (3/4))
        table = []
        for index,pr in enumerate(prob):
            table.extend([index] * int(pr * 1e7))
        table = np.array(table)
        
    if mode3 == "SB":
        freq = Counter(corpus)
        sum_freq = sum(freq.values())
        t = 0.0005
        rand_table = np.random.random(size = len(corpus))
        temp = []
        for idx, word in enumerate(corpus):
            prob = 1-np.sqrt(t/(freq[word]/sum_freq))
            if rand_table[idx] < prob:
                temp.append(idx)
            if idx % 1000000 == 0:
                print('now {} words finished'.format(idx))
        corpus = list(np.delete(corpus, temp))
        print('Subsampling finished')
        print('size : {}'.format(len(corpus)))
        iteration = len(corpus)

    print('making n_grams')
    ind_slice_index = {}
    for i, j in zip(word2ind.keys(), word2ind.values()):
        ind_slice_index[j] = [fnv(t, len(word2ind)) for t in slicing(i, [3, 4, 5, 6])] + [j]
    
    losses=[]
    print('iter: {}'.format(iteration))

    for i in range(iteration):
        #Training fasttext using SGD
        centerword, context = getRandomContext(corpus, window_size)
        centerInd = word2ind[centerword]
        contextInds = [word2ind[i] for i in context]
        
        if mode=="CBOW":
            if mode2 == "NS":
                temp = []
                while True:
                    if len(temp) == 16:
                        break
                    rand = np.random.randint(0, len(table), 1)
                    if table[rand] not in temp and table[rand] != centerInd:
                        temp.append(table[rand].squeeze())
                list_k = np.array(temp + [centerInd])
                temp2 = []
                for i in contextInds:
                    temp2.extend(ind_slice_index[i])
                contextInds = temp2
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
            centerInd = ind_slice_index[centerInd]
            for contextInd in contextInds:
                if mode2 == "NS":
                    temp = []
                    while True:
                        if len(temp) == 5:
                            break
                        rand = np.random.randint(0, len(table), 1)
                        if table[rand] not in temp and table[rand] != contextInd:
                            temp.append(table[rand].squeeze())
                    list_k = np.array(temp + [contextInd])
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[list_k, :], mode2)
                    W_emb[centerInd] -= learning_rate*G_emb
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

        if i%1000000==0:
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

def testing(vec, ind2word, matrix):
    vec_length = torch.sum((vec*vec))**0.5
    length = (matrix*matrix).sum(1)**0.5
    vec_normed = vec.reshape(1, -1)/vec_length
    sim = (vec_normed@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    result = []
    for i in indices:
        result.append(ind2word[i.item()])
    return result

def sim2(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    try:
        wi = word2ind[testword] + [fnv(t, len(word2ind)) for t in slicing(testword, [3, 4, 5, 6])]
    except:
        wi = [fnv(t, len(word2ind)) for t in slicing(testword, [3, 4, 5, 6])]
    inputVector = torch.sum(matrix[wi], dim=0, keepdim=True).reshape(1,-1)/torch.sum(length[wi], dim = 0, keepdim=True)
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze()[:len(word2ind)].topk(5)

    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()

def sim3(testword, word2ind, ind2word, matrix):
    temp = torch.randn(len(word2ind), 300)
    for i in word2ind.keys():
        wi = [word2ind[i]] + [fnv(t, len(word2ind)) for t in slicing(i)]
        temp[word2ind[i]] = torch.sum(matrix[wi], dim =0, keepdim=True)
    try:
        wi = word2ind[testword] + [fnv(t, len(word2ind)) for t in slicing(testword, [3, 4, 5, 6])]
    except:
        wi = [fnv(t, len(word2ind)) for t in slicing(testword, [3, 4, 5, 6])]
    length = (matrix*matrix).sum(1)**0.5
    inputVector = torch.sum(matrix[wi], dim=0, keepdim=True).reshape(1,-1)/torch.sum(length[wi], dim = 0, keepdim=True)
    sim = (inputVector@temp.t())[0]/((temp * temp).sum(1)**0.5)
    values, indices = sim.squeeze().topk(5)

    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()    




def main():
    parser = argparse.ArgumentParser(description='Fast Text')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('mode2', metavar='mode2', type=str,
                        help='"HS" for Hierarchical Softmax, "NS" for Negative Sampling')
    parser.add_argument('mode3', metavar='mode3', type=str, help='"SB" for subsampling, default is None')
    parser.add_argument('test', metavar='test', type=str, help='learn and testing')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    mode2 = args.mode2
    mode3 = args.mode3
    test = args.test

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
    print(len(word2ind))
        
    print("Vocabulary size")
    print(len(word2ind), len(processed))
    print()

    #Training section
    emb,_ = fasttext_trainer(processed, word2ind, mode=mode, mode2=mode2, mode3=mode3, dimension=300, learning_rate=0.025, iteration=len(processed)*2)
    
    print("1 epoch finished")
    
    
    testwords = ["narrow-mindedness", "imperfection", "department",
                 "principality", "campfires", "abnormal", "knowing",
                  "secondary", "urbanize", "ungraceful"]
    for tw in testwords:
        sim3(tw,word2ind,ind2word,emb)

main()