'''
資料預處理
利用 word2vec 取得每則貼文/回覆的詞向量(shape: #貼文, #回覆, #回覆字詞, #嵌入向量) --> HENIN, GCN 
利用 doc2vec 取得每則貼文/回覆的文本向量 (shape: #貼文, #回復, #嵌入向量) --> LSTM, GRU, RNN
利用 doc2vec 取得每則貼文的文本向量 (shape: #貼文, #嵌入向量) --> RF, LR, XGB, DF
'''
from collections import Counter
from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import itertools
from keras.preprocessing import sequence
import logging
import math
import numpy as np
import networkx as nx
from nltk.tokenize import word_tokenize
import nltk
import os 
import pandas as pd
import pickle
import re
import random
import spacy
from time import time

from utils import *
nltk.download('punkt')

# 載入原始資料
filePath = '../data/Insta_labeled_data/'
all_dict = generate_dict(filePath)

# 收集資料集中所有的文字
reviews = []
for i in range(len(all_dict)):
    reviews.append(all_dict[list(all_dict.keys())[i]].text)
reviews_freq = len(list(itertools.chain.from_iterable(reviews)))
print('reviews count in whole data set:', reviews_freq) # reviews count in whole data set: 159277
df = pd.DataFrame({'spoken_words': list(itertools.chain.from_iterable(reviews))})

'''
執行 word2vec 模型前，先對所收集到的文字做基本的清理，包含
(a).移除不同語言產生之亂碼或是表情符號所導致的無法辨識的字串或是停用詞;
(b).處理雙字母組(Bigrams)
'''
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
# python3 -m spacy download en 
nlp = spacy.load('en', disable=['ner', 'parser'])

## (a). Cleaning 
def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # if a sentence is only one or two words long, 
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)
    
brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(row).lower()) for row in df['spoken_words']]
# taking advantages of spacy .pipe() attribute to speed-up the cleaning process
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=1)]
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
print('df_clean.shape:', df_clean.shape) # reviews count in whole data set after cleaning: (95163, 1)

## (b). Bigrams
# using Gensim Phrases package to automatically detect common phrases(bigrams)
# e.g. 'mr_burns' or 'bart_simpson'
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

'''
貼文及回覆的基本統計量
'''
sess_review_cnt = [len(messages['text']) for pid, messages in all_dict.items()]
print('max review length:', max(sess_review_cnt)) # max review length: 193
print('min review length:', min(sess_review_cnt)) # min review length: 15
print('mean review length: {:.3f}'.format(np.mean(sess_review_cnt))) # mean review length: 72.038
print('median review length: {:.3f}'.format(np.median(sess_review_cnt))) # median review length: 52.0
review_word_len = [len(x.split()) for x in df_clean['clean']]
print('max review word length:', max(review_word_len)) # max review word length: 211
print('min review word length:', min(review_word_len)) # min review word length: 2
print('mean review word length: {:.3f}'.format(np.mean(review_word_len))) # mean review word length: 8.293
print('median review word length: {:.3f}'.format(np.median(review_word_len))) # median review word length: 5.0

'''
訓練 word2vec 模型
'''
import multiprocessing
from gensim.models import Word2Vec

# 設定 word2vec 的模型參數
w2v_model = Word2Vec(min_count=20, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=10)
# 建立字典
w2v_model.build_vocab(sentences, progress_per=10000)
# 訓練 word2vec 模型
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
# 如果沒有要進一步訓練模型，可以使用 .init_sims()，讓模型的儲存更高效
w2v_model.init_sims(replace=True)

'''
利用訓練好的 word2vec 模型，產生每則貼文/回覆的詞向量，相關參數為 
model: 預訓練好的 word2vec 模型, i.e., w2v_model
dictDat: 原始資料集並整理成 dictionary 的形式
MAX_REV_WORD_LEN: 將不同長度的貼文/回覆的字詞 padding 成相同長度，值參考自基本統計量中得到的平均值, i.e., 8.293 約等於 10
MAX_REV_LEN: 將每則貼文的不同回覆數padding成相同長度, 值參考至基本統計量得到的平均值, i.e., 72.038 約等於 75
--> As Input for HENIN, GCN model
'''
def get_each_review_emb(model, dictDat, MAX_REV_WORD_LEN, MAX_REV_LEN):
    
    y = []
    try:
        del all_reviews
    except:
        pass
        
    for i, (pid, messages) in enumerate(dictDat.items()):
        progress = divmod(i+1, int(len(dictDat)*0.1))
        if progress[1] == 0:
            print('%s %% datasets processing!'%(progress[0]*10))
        
        try:
            del review_vec
        except:
            pass
        
        df = pd.DataFrame({'review_word': messages.text})
        brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(row).lower()) for row in df['review_word']]
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=1)]
        df_clean = pd.DataFrame({'clean': txt})
        df_clean = df_clean.dropna().drop_duplicates()


        for review in df_clean['clean']:
            vocab = list(model.wv.vocab.keys())
            one_review_vec = np.array([model.wv.__getitem__([wrd]) for wrd in review.split() if wrd in vocab])
        
            if len(one_review_vec) != 0:    
                one_review_vec = np.transpose(one_review_vec, (1, 0, 2))
                # mean word count in review is '8.293' 
                # one_review_rec: [#review(?), #words(MAX_REV_WORD_LEN), #emb_dim(300)]
                one_review_vec = sequence.pad_sequences(one_review_vec, maxlen=MAX_REV_WORD_LEN, padding='post', dtype='float32')
        
                try:
                    review_vec = np.concatenate((review_vec, one_review_vec), axis=0)
                except:
                    review_vec = one_review_vec
        
        if len(review_vec) != 0:
            review_vec = review_vec[np.newaxis, :, :]
            # mean reviews count in post is '72.038'
            # review_vec: [#post(1), #reviews(MAX_REV_LEN), #words(MAX_REV_WORD_LEN), #emb_dim(300)]
            review_vec = sequence.pad_sequences(review_vec, maxlen=MAX_REV_LEN, padding='post', dtype='float32')
            try:
                all_reviews = np.concatenate((all_reviews, review_vec), axis=0)
            except:
                all_reviews = review_vec
                
            y += [1 if messages.label[0] == 'bullying' else 0]
    
    return all_reviews, y

w2v_vec_all, y_all = get_each_review_emb(model=w2v_model, dictDat=all_dict, MAX_REV_WORD_LEN=10, MAX_REV_LEN=75)
y_all = np.array(y_all)
print('w2v_vec_all.shape: ', w2v_vec_all.shape) # w2v_vec_all.shape: (2211, 75, 10, 300)
print('y_all.shape: ', y_all.shape) # y_all.shape: (2211, 1)

'''
訓練 doc2vec 模型
收集每則貼文的所有回覆(經清理過)作為該則貼文的文本來訓練 doc2vec 模型
'''
def get_sentences(dictDat):
    
    setences = []
    for i, (pid, messages) in enumerate(dictDat.items()):
        print('%s-th event is processing...'%(i+1))
        df = pd.DataFrame({'review': messages.text})
        brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(row).lower()) for row in df['review']]
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=1)]
        df_clean = pd.DataFrame({'txt': txt})
        doc = ' '.join(df_clean.dropna().drop_duplicates().txt)
        taggedDoc = TaggedDocument(words = word_tokenize(doc), tags = [str(pid)])
        setences.append(taggedDoc)
        
    return setences

# 收集每則貼文的所有回覆
sentence = get_sentences(all_dict)
# 設定 doc2vec 模型參數
doc_vectorizer = Doc2Vec(min_count=1, windows=10, vector_size=300, sample=1e-4, negative=5, workers=8)
# 建立字典
doc_vectorizer.build_vocab(sentence)
# 訓練 doc2vec 模型
doc_vectorizer.train(sentence, total_examples=doc_vectorizer.corpus_count, epochs=30)

'''
利用訓練好的 doc2vec 模型，產生每則貼文/回覆的文本向量，相關參數為 
model: 預訓練好的 doc2vec 模型, i.e., doc_vectorizer
dictDat: 原始資料集並整理成dictionary的形式
MAX_REV_LEN: 將每則貼文的不同回覆數padding成相同長度, 值參考至基本統計量得到的平均值, i.e., 72.038 約等於 75
--> As Input for RNN, LSTM, GRU model
'''
def get_textFeat(model, dictDat, MAX_REV_LEN):
    
    textFeat = []
    for i, (pid, messages) in enumerate(dictDat.items()):
        print('%s-th event is processing...'%(i+1))
        one_textFeat = [] # single post text features
        df = pd.DataFrame({'review': messages.text})    
        brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(row).lower()) for row in df['review']]
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=1)]
        df_clean = pd.DataFrame({'txt': txt})
        df_review = df_clean.dropna().drop_duplicates().txt
        for rev in df_review:
            one_bin_textFeat = model.infer_vector(word_tokenize(rev)).tolist()
            one_textFeat.append(one_bin_textFeat)
        textFeat.append(one_textFeat)
            
    # padding sequence to equal size
    textFeat = sequence.pad_sequences(textFeat, maxlen=MAX_REV_LEN, truncating='post', padding='post', dtype='float32', value=0.0)
    
    return textFeat
    
textFeat_all = get_textFeat(model=doc_vectorizer, dictDat=all_dict, MAX_REV_LEN=75)
print('textFeat_all.shape: ', textFeat_all.shape) # textFeat_all.shape: (2211, 75, 300)

'''
利用訓練好的 doc2vec 模型，產生每則貼文的文本向量，相關參數為 
model: 預訓練好的 doc2vec 模型, i.e., doc_vectorizer
dictDat: 原始資料集並整理成dictionary的形式
--> As Input for RF, LR, XGB DF model
'''
def get_textFeat_clf(model, dictDat):
    
    textFeat = []
    for eid, messages in dictDat.items():
        df = pd.DataFrame({'review': messages.text})
        brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(row).lower()) for row in df['review']]
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=1)]
        df_clean = pd.DataFrame({'txt': txt})
        doc = ' '.join(df_clean.dropna().drop_duplicates().txt)
        one_textFeat = model.infer_vector(word_tokenize(doc)).tolist()        
        textFeat.append(one_textFeat)
            
    return np.array(textFeat)

textFeat_clf_all = get_textFeat_clf(model=doc_vectorizer, dictDat=all_dict)
print('textFeat_clf_all.shape: ', textFeat_clf_all.shape) # textFeat_clf_all.shape (2211, 300)

# 儲存預處理後的資料
Dat4Model = {'w2v_vec_all': w2v_vec_all, 'textFeat_all': textFeat_all, 'textFeat_clf_all': textFeat_clf_all, 'y_all': y_all}
with open('preprocessData/Dat4Model.pickle', 'wb') as f:
    pickle.dump(Dat4Model, f)
    

'''
生成 貼文-用戶 相連的圖形
'''
def PostUserGraph(dictDat):
    # 貼文 id --> pid
    # 用戶 id --> uid
    
    # 將 pid 重新編碼, ex. 679604281 --> 1
    pid2gid = dict(zip(list(all_dict.keys()), range(1, len(all_dict)+1)))
    
    ## 紀錄每則貼文中每則回覆的用戶 uid
    # i.e, pid, uid
    #      1, 'cmacdaddyy'
    for i, (pid, messages) in enumerate(all_dict.items()):
        print('%s -th session processing!'%(i))
        df = pd.DataFrame({'review': messages.text, 'uid': messages.uid})
        brief_cleaning = [re.sub("[^A-Za-z]+", ' ', str(row).lower()) for row in df['review']]
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=1)]
        df_clean = pd.DataFrame({'pid': pid2gid[pid], 'clean': txt, 'uid': df.uid})
        df_clean = df_clean.dropna().drop_duplicates()
        g = df_clean[['pid', 'uid']]
    
        try:
            G_dt = G_dt.append(g)
        except:
            G_dt = g
    G_dt = G_dt.reset_index(drop=True)
    print('There have %s users'%(len(list(set(G_dt.uid))))) # There have 72176 users
    
    ## 以正整數重新編碼 uid, 為使圖形中每個點的 id 唯一，所以 uid 以 '999' 為開頭用以區別於 pid
    # i.e., pid: 1-2211; uid: 9991-...
    uid = list(set(G_dt.uid))
    uix = ['999' + str(i) for i in range(1, len(list(set(G_dt.uid)))+1)]
    uname2uid = dict(zip(uid, uix))
    G_dt['uix'] = [uname2uid[i] for i in G_dt.uid]
    
    PUGraph_dt = G_dt[['pid', 'uix']]
    PUGraph_dt = PUGraph_dt.drop_duplicates()
    PUGraph_dt = PUGraph_dt.astype(str)
    print('Complete User-Post Graph shape is (remove duplicates edges): ', PUGraph_dt.shape) # Complete User-Post Graph shape is (remove duplicates edges):  (75659, 2)
    # transform dataframe object to graph object
    PUGraph = nx.from_pandas_edgelist(PUGraph_dt, 'pid', 'uix', create_using=nx.Graph()) 
    print('There have %s nodes in the User-Post graph.'%(len(PUGraph.nodes()))) # There have 74387 nodes in the User-Post graph.
    
    return PUGraph_dt

PUGraph_dt = PostUserGraph(dictDat=all_dict)

'''
每則貼文用戶的 multi-hot encoding
ex.   u1 u2 u3...
   p1 0  1  1 ...
   p2 0  0  0 ... (N x K), N: amount of posts; K: amount of users
'''
def MultiHotUsers(GraphDt):
    # Need input Post-User Graph
    
    GraphDt = GraphDt.astype(int)
    Graph = nx.from_pandas_edgelist(GraphDt, 'pid', 'uix', create_using=nx.Graph()) 
    multi_hot_users = nx.adjacency_matrix(Graph, nodelist=sorted(Graph.nodes()))
    
    post_nodes = [x for x in Graph.nodes() if not bool(re.match(r"999\d", str(x)))]
    multi_hot_users = multi_hot_users[:len(post_nodes), len(post_nodes):]
    
    return multi_hot_users

multi_hot_users = MultiHotUsers(GraphDt=PUGraph_dt)
print('multi_hot_users.shape: ', multi_hot_users.shape) # multi_hot_users.shape: (2211, 72176)    

# 儲存 Multi-Hot Users
with open('preprocessData/multi_hot_users.pickle', 'wb') as f:
    pickle.dump(multi_hot_users, f)
   
print('All Process Done.')    
