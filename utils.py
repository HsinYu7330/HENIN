import re
import pandas as pd
import pickle
from datetime import datetime
import math
import os 
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 載入原始資料
def load_oriDat(filePath):
    sessFiles = [x for x in os.listdir(filePath) if 'sessions' in x]
    sessDat = pd.DataFrame()
    for i in range(len(sessFiles)):
        dat = pd.read_csv(filePath + sessFiles[i], encoding = 'ISO-8859-1')
        sessDat = sessDat.append(dat, ignore_index = True, sort= False)
        
    # imputue 'empty' to owner comment is nan
    sessDat.loc[sessDat.owner_cmnt.isna(), 'owner_cmnt'] = 'empty'
    
    # Description of datasets
    print('The shape of all session data: ', sessDat.shape) # The shape of all session data: [2218, 215]
    print('The counts of unique session id', len(set(sessDat.id))) # The counts of unique session id: 2218
    print(Counter(sessDat.question2)) # non Bully: 1540, Bullying: 678
             
    return sessDat

# 將每則貼文整理成 dictionary，包含回覆者(uid), 回覆時間(timestamp), 回覆文字(text), 該則貼文是否有霸凌現象(label)等資訊
# dictionary 中的第一則文字為原始貼文
def generate_dict(filePath):
    
    sessDat = load_oriDat(filePath)
        
    '''
    create input dictionary data

    {'652910876': {'uid': [xxx, yyy]}, 
                  {'timestamp':[111, 222]},
                  {'text': ['hi', 'good']},
                  {'label': ['bully', 'nonebull']},
                  {'likes': [751, 124]},
                  {'follows': [789, 458]},
                  {'followed_by':[7892, 4875]}
              
    }

    '''

    clmn = [x for x in sessDat.columns if 'clmn' in x] # column of comments
    TAG_RE = re.compile(r'<[^>]+>')
    ownerID = [TAG_RE.sub('', x)[:-4] for x in sessDat.owner_id] # original post id
    ownerCn = sessDat['owner_cmnt'] # original post text
    
    '''
    some post time unusual
    ex. Media posted at 014-06-09 22:20:18
    ex. Media posted at 2013-05-21 21:54:37\r\r
    '''
    ownerTime = [x.replace('Media posted at ', '') for x in sessDat.cptn_time] # original post time
    ownerTime = ['2' + x.replace('\r\r', '') if x[0] != '2' else x.replace('\r\r', '') for x in ownerTime]
    
    # create dictionary data
    all_dict = {}
    for i in range(sessDat.shape[0]):
        eid = sessDat._unit_id[i]
        
        try:
            del uid, timestamp, text
        except:
            pass
         
        comments = sessDat.loc[i, clmn]
        comments = [x for x in comments if 'empety' not in x] # remove empety elements
        comments = [TAG_RE.sub('', x) for x in comments] # remove html strings
        
        uid = [ownerID[i]] + [x.split('   ')[0] for x in comments] # responses id
        timestamp = [ownerTime[i]] + [x[-20:-1] for x in comments] # comment time
        text = [ownerCn[i]] + [x.split('   ')[1][:-34] for x in comments] # comment text
        
        messages = {'uid': uid, 'timestamp': timestamp, 'text': text, 'label': sessDat.loc[i, 'question2'], 
                    'likes': sessDat.loc[i, 'likes'][:-8], 'follows': sessDat.loc[i, 'follows'], 'followed_by': sessDat.loc[i, 'followed_by']}
        
        messages = pd.DataFrame(messages)
        messages['timestamp'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in messages.timestamp]
        messages = messages.sort_values(by='timestamp').reset_index(drop=True)
        
        if len(messages.text) >= 15:
            all_dict[eid] = messages
       
    return all_dict

def metrics(y, pred):
    
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

def Mask_y(y, train_ix, test_ix): 
    
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[train_ix] = y[train_ix]
    y_test[test_ix] = y[test_ix]
    train_mask = sample_mask(train_ix, y.shape[0])
    
    return y_train, y_test, train_mask

def genAdjacencyMatrix(X, metrics):
    
    if metrics == 'cosine':
        A = cosine_similarity(X)
        A = np.exp(-A**2)
    elif metrics == 'jaccard':
        A = cdist(X, X, 'jaccard')
        A = np.exp(-A**2)
    elif metrics == 'euclidean':
        A = cdist(X, X, 'euclidean')
        A = np.exp(-A**2)
        
    A_ = csr_matrix(A)
    
    return A_


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

def rescale_laplacian(laplacian):
    try:
        # print('Calculating largest eigenvalue of normalize graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except:
        # print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2
    
    scaled_laplacian = (2. / largest_eigval)*laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian

def chebyshev_polynomial(X, k):
    '''Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices.'''
    # print('Calculating Chebyshev polynomials up to order {}...'.format(k))
    
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)
    
    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2*X_.dot(T_k_minus_one) - T_k_minus_two
    
    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
        
    return T_k

def genGCNgraph(adjacency, X):
    
    ''' Generate graph object with chebysheve filter as GCN input'''
    MAX_DEGREE = 2
    L = normalized_laplacian(adj=adjacency, symmetric=True)
    L_scaled = rescale_laplacian(laplacian=L)
    T_k = chebyshev_polynomial(X=L_scaled, k=MAX_DEGREE)
    graph = [X] + T_k
    
    return graph


