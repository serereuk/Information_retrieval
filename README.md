# Information_retrieval

* word2vec.py  
  word2vec implementation, You can use Hierarchical softmax, Negative Sampling, Subsampling (Skip_gram, CBOW)

* fast.py   
  fast_text implementation, You can use Negative Sampling, Subsampling (Skip_gram, CBOW)

* fastclassification.py   
  fast text based classification the result of AG datasets is 0.90 almost same as paper presented 92.5(h = 10, bigram).
  The difference of accuracy is from batch or lr setting (In my opinion).
  
* textCNN.ipynb    
  textCNN-rand(2014, kim) implementation in tensorflow2.0. The accuracy is 74.7 which is similar to 76.1 in paper. The accuracy is increasing when I use another initializers. In this implementation, I use default one.

