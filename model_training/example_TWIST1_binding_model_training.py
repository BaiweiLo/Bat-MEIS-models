import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, InputLayer, Input, GlobalAvgPool1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Concatenate, PReLU, TimeDistributed 
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
sys.path.append('/project/bat_analysis/baiwei/software/Neural_Network_DNA_Demo/')
from helper import IOHelper, SequenceHelper # from https://github.com/const-ae/Neural_Network_DNA_Demo


import random
random.seed(1234)


## import fasta sequences
path='/project/bat_analysis/data/enhancer/prox_dist/mouse/twist1/deepLearning'

input_fasta_data = IOHelper.get_fastas_from_file(path + '/train.pos.fasta', uppercase=True)
X_train_pos = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)
input_fasta_data = IOHelper.get_fastas_from_file(path + '/train.neg.fasta', uppercase=True)
X_train_neg = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)
input_fasta_data = IOHelper.get_fastas_from_file(path + '/valid.pos.fasta', uppercase=True)
X_valid_pos = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)
input_fasta_data = IOHelper.get_fastas_from_file(path + '/valid.neg.fasta', uppercase=True)
X_valid_neg = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)
input_fasta_data = IOHelper.get_fastas_from_file(path + '/test.pos.fasta', uppercase=True)
X_test_pos = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)
input_fasta_data = IOHelper.get_fastas_from_file(path + '/test.neg.fasta', uppercase=True)
X_test_neg = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)
## concatenate sequences
X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
X_valid=np.concatenate((X_valid_pos,X_valid_neg),axis=0)
X_test=np.concatenate((X_test_pos,X_test_neg),axis=0)

## create labels
y_train=np.concatenate((np.ones(X_train_pos.shape[0]),
                        np.zeros(X_train_neg.shape[0])))
y_valid=np.concatenate((np.ones(X_valid_pos.shape[0]),
                        np.zeros(X_valid_neg.shape[0])))
y_test=np.concatenate((np.ones(X_test_pos.shape[0]),
                        np.zeros(X_test_neg.shape[0])))


## model construction
cl2_1fc_model=Sequential() 
cl2_1fc_model.add(Conv1D(filters=64,kernel_size=(10),padding="same", input_shape=X_train.shape[1::]))
cl2_1fc_model.add(Activation('relu'))
cl2_1fc_model.add(BatchNormalization(axis=-1))


cl2_1fc_model.add(Conv1D(filters=32,kernel_size=(7),padding="same", input_shape=X_train.shape[1::]))
cl2_1fc_model.add(Activation('relu'))
cl2_1fc_model.add(BatchNormalization(axis=-1))


cl2_1fc_model.add(MaxPooling1D(pool_size=(35)))

cl2_1fc_model.add(Flatten())
cl2_1fc_model.add(Dense(64))
cl2_1fc_model.add(Activation('relu'))
cl2_1fc_model.add(Dropout(0.3))
cl2_1fc_model.add(Dense(1))
cl2_1fc_model.add(Activation("sigmoid"))

##compile the cl2_1fc_model, specifying the Adam optimizer, and binary cross-entropy loss. 
cl2_1fc_model.compile(optimizer='adam',
                               loss='binary_crossentropy',  metrics =[AUC(name='auc')] )

cl2_1fc_model.summary()

## training the model
history_cl2_1fc_model = cl2_1fc_model.fit(x=X_train,
                                  y=y_train,
                                  batch_size=128,
                                  epochs=150,
                                  verbose=1,
                                  callbacks=[EarlyStopping(patience=3,restore_best_weights=True),
                                            History()], validation_data=(X_valid,y_valid))

## evaluate the model on test set
cl2_1fc_model.evaluate(X_test, y_test)
