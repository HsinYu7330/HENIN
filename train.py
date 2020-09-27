from collections import defaultdict
from keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D, GRU, Bidirectional
from keras.layers import GlobalMaxPooling1D, LSTM, Dropout, SimpleRNN, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras.layers import concatenate
from keras import activations, initializers, constraints
from keras import regularizers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

from layers import *
from utils import *


## The proposed model, HENIN
def HENIN(GCNXss_shape, GCNXpp_shape, n_head=8, size_per_head=8, MAX_REV_LEN=75, MAX_REV_WORD_LEN=10, support=3):
    
    '''
    Comment Encoding
    '''
    
    ''' Capture reviews context correlation'''
    ## word-level encoding
    word_input = Input(shape=(None, 300), dtype='float32')
    word_sa = Self_Attention(n_head, size_per_head)(word_input)
    word_avg = GlobalAveragePooling1D()(word_sa)
    wordEncoder = Model(word_input, word_avg)
    
    ## review-level encoding
    content_input = Input(shape=(MAX_REV_LEN, MAX_REV_WORD_LEN, 300), dtype='float32')
    content_word_encode = TimeDistributed(wordEncoder, name='word_seq_encoder')(content_input)
    content_sa = Self_Attention(n_head, size_per_head)(content_word_encode)
    contentSA_avg_pool = GlobalAveragePooling1D()(content_sa) # session embedding
    
    ''' Capture Post-Comment co-attention'''
    post_words_input = Input(shape=(None, 300), dtype='float32')
    post_lstm = Bidirectional(GRU(32, return_sequences=True))(post_words_input)
    coAtt_vec = CoAttLayer(MAX_REV_LEN)([content_word_encode, post_lstm])
    
    '''
    GCN
    Session-Session Interaction Extractor
    Adjacency: session-session
    '''
    G_ss = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(3)]
    
    X_ss = Input(shape=(GCNXss_shape,))
    X_ss_emb = Dense(16, activation='relu')(X_ss)
    
    # Define GCN model architecture
    H_ss = Dropout(0.2)(X_ss_emb)
    H_ss = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H_ss]+G_ss)
    H_ss = GraphConvolution(8, support, activation='softmax', kernel_regularizer=l2(5e-4))([H_ss]+G_ss)
    
    '''
    GCN
    Post-Post Interaction Extractor
    Adjacency: post-post
    '''
    G_pp = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(3)]
    
    X_pp = Input(shape=(GCNXpp_shape,))
    X_pp_emb = Dense(16, activation='relu')(X_pp)
    
    # Define GCN model architecture
    H_pp = Dropout(0.2)(X_pp_emb)
    H_pp = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H_pp]+G_pp)
    H_pp = GraphConvolution(8, support, activation='softmax', kernel_regularizer=l2(5e-4))([H_pp]+G_pp)
     
    '''
    Concatenate Comment Encoding & GCN Embedding
    '''
    H = concatenate([contentSA_avg_pool, coAtt_vec, H_ss, H_pp])
    Y = Dense(1, activation='sigmoid')(H)
    
    # Compile model
    model = Model(inputs=[content_input]+[post_words_input]+[X_ss]+G_ss+[X_pp]+G_pp, outputs=Y)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))
    model.summary()
    
    return model


'''
Load data
'''
# load preprocessed data
with open('preprocessData/Dat4Model.pickle', 'rb') as f:
    Dat4Model = pickle.load(f)
    
# load multi-hot user vectors of each session
with open('preprocessData/multi_hot_users.pickle', 'rb') as f:
    multi_hot_users = pickle.load(f)  
    
w2v_vec_all = Dat4Model['w2v_vec_all'] # features for HENIN
y_all = Dat4Model['y_all'] # target for HENIN
textFeat_all = Dat4Model['textFeat_all']

MAX_REV_WORD_LEN = w2v_vec_all.shape[2]
MAX_REV_LEN = w2v_vec_all.shape[1]

# word embedding of posted text
postEmb = pad_sequences(w2v_vec_all[:,0,:,:], maxlen=MAX_REV_LEN, dtype='float32', padding='post') 

## cross validating for HENIN model
def HENIN_cv(graph, y, A, model, epochs):
    
    skf = StratifiedKFold(n_splits=5, random_state=9999, shuffle=True)
    iters = 0
    
    for train_index, test_index in skf.split(range(len(y)), y):
        
        y_train, y_test, train_mask = Mask_y(y=y, train_ix=train_index, test_ix=test_index)
        clf = model
        for epoch in range(epochs):
            clf.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1)
        preds = (clf.predict(graph, batch_size=A.shape[0])[:,0] >= 0.5).astype(int)
        
        completePerform = metrics(y, preds) # Complete set performance
        generalPerform = metrics(y[test_index], preds[test_index]) # test set performance
        
        try:
            if iters == 1:
                CP = {k: v + [completePerform[k]] for k, v in CP.items()}
                GP = {k: v + [generalPerform[k]] for k, v in GP.items()}
            else:  
                CP = {k: [v] + [completePerform[k]] for k, v in CP.items()}
                GP = {k: [v] + [generalPerform[k]] for k, v in GP.items()}
                iters += 1
        except:
            CP = completePerform
            GP = generalPerform
    
    AvgCP = {k: '{:.3f}'.format(np.mean(v)) for k, v in CP.items()}
    AvgGP = {k: '{:.3f}'.format(np.mean(v)) for k, v in GP.items()}
    
    return AvgCP, AvgGP

OurComResult = {}
OurGenResult = {}

## HENIN
ppA = genAdjacencyMatrix(textFeat_all[:,0,:], 'cosine')
ssA = genAdjacencyMatrix(multi_hot_users, 'cosine')

graph_ss = genGCNgraph(ssA, multi_hot_users)
graph_pp = genGCNgraph(ppA, textFeat_all[:,0,:])

graph = [w2v_vec_all]+[postEmb]+graph_ss+graph_pp

clf = HENIN(GCNXss_shape=multi_hot_users.shape[1], 
	        GCNXpp_shape=textFeat_all[:,0,:].shape[1], 
	        n_head=8, size_per_head=8, MAX_REV_LEN=MAX_REV_LEN, 
	        MAX_REV_WORD_LEN=MAX_REV_WORD_LEN, support=3)

AvgCP, AvgGP = HENIN_cv(graph=graph, y=y_all, A=ppA, model=clf, epochs=10)
print(AvgGP)

