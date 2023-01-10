
###############################
from google.colab import drive
drive.mount('/content/drive')

###############################
#!pip install -q keras
#!pip install --upgrade -q tensorflow
#!pip install keras==2.4.3
import keras
import tensorflow

print(keras.__version__)
print(tensorflow.__version__)

###############################
!pip install bert-for-tf2 >> /dev/null
!pip install sentencepiece >> /dev/null
#!pip install git+https://github.com/KingsleyNA/NLP-on-a-ktrain
!pip install keras-self-attention
!pip install --upgrade grpcio >> /dev/null
!pip install tqdm  >> /dev/null
!pip install keras-multi-head

###############################
try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers
import bert

###############################
import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


###############################
#import ktrain
#from ktrain import text as txt

from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead

import numpy as np
import os
import sys

import wave
import copy
import math

import keras

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import GRU, LSTM, Input, Flatten, Concatenate, Embedding, Convolution1D, Dropout, Bidirectional, Conv2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
from keras.layers import *




import tensorflow as tf

###############################
%cd /content/drive/MyDrive/PROJECT/codes

###############################
from features import *
from helper import *

###############################
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

###############################
code_path = "/content/drive/MyDrive/PROJECT/data"
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = code_path + "/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

###############################
import pickle
with open(data_path + '/'+'data_collected.pickle', 'rb') as handle:
    data2 = pickle.load(handle)

###############################    
text = []

for ses_mod in data2:
    text.append(ses_mod['transcription'])

reviews = []

for i in text:
    reviews.append(i)
    
len(reviews)

###############################
Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])
    
#Y = label_binarize(Y,emotions_used)

#Y.shape
#type(Y)
Y[0], Y[1], Y[2], Y[3], Y[4]

###############################
x_train = reviews[:3838]
x_test = reviews[3838:]
y_train = Y[:3838]
y_test = Y[3838:]

###############################
train = pd.DataFrame(
    {'review': x_train,
     'emotion': y_train
    })

test = pd.DataFrame(
    {'review': x_test,
     'emotion': y_test
    })

###############################    
train.head()

###############################
chart = sns.countplot(train.emotion, palette=HAPPY_COLORS_PALETTE)
plt.title("No. of reviews per emotion train data")
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');

###############################
chart = sns.countplot(test.emotion, palette=HAPPY_COLORS_PALETTE)
plt.title("No. of reviews per emotion test data")
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');

###############################
!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
os.makedirs("model", exist_ok=True)
!mv uncased_L-12_H-768_A-12/ model


###############################
bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


###############################
#**Preprocessing**
class IntentDetectionData:
  DATA_COLUMN = "review"
  LABEL_COLUMN = "emotion"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)


###############################    
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
tokenizer.tokenize("I can't wait to visit Bulgaria again!")
tokens = tokenizer.tokenize("I can't wait to visit Bulgaria again!")
tokenizer.convert_tokens_to_ids(tokens)


###############################
def calculate_features(frames, freq, options):
    window_sec = 0.2
    window_n = int(freq * window_sec)

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        
        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f

###############################        
x_train_speech = []

counter = 0
for ses_mod in data2:
    x_head = ses_mod['signal']
    st_features = calculate_features(x_head, framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
    x_train_speech.append( st_features.T )
    counter+=1
    if(counter%100==0):
        print(counter)
    
x_train_speech = np.array(x_train_speech)
x_train_speech.shape
x_train_mocap = []
counter = 0
for ses_mod in data2:
    x_head = ses_mod['mocap_head']
    if(x_head.shape != (200,18)):
        x_head = np.zeros((200,18))   
    x_head[np.isnan(x_head)]=0
    x_hand = ses_mod['mocap_hand']
    if(x_hand.shape != (200,6)):
        x_hand = np.zeros((200,6))   
    x_hand[np.isnan(x_hand)]=0
    x_rot = ses_mod['mocap_rot']
    if(x_rot.shape != (200,165)):
        x_rot = np.zeros((200,165))   
    x_rot[np.isnan(x_rot)]=0
    x_mocap = np.concatenate((x_head, x_hand), axis=1)
    x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
    x_train_mocap.append( x_mocap )
    
x_train_mocap = np.array(x_train_mocap)
x_train_mocap = x_train_mocap.reshape(-1,200,189,1)
x_train_mocap.shape
# k = 1

###############################
xtrain_sp = x_train_speech[:3838]
xtest_sp = x_train_speech[3838:]
ytrain_sp = Y[:3838]
ytest_sp = Y[3838:]


#xtrain_sp = np.array(xtrain_sp)
#xtest_sp = np.array(xtest_sp)
#ytrain_sp = np.array(ytrain_sp)
#ytest_sp = np.array(ytest_sp)

from pandas.core.common import flatten

xtrain_sp_ch = []
xtest_sp_ch = []

for i in xtrain_sp:
  a = list(flatten(i))
  xtrain_sp_ch.append(a)


for i in xtest_sp:
  a = list(flatten(i))
  xtest_sp_ch.append(a)

print(type(xtrain_sp_ch))

xtrain_sp_ch = numpy.array(xtrain_sp_ch)
xtest_sp_ch = numpy.array(xtest_sp_ch)

print(type(xtrain_sp_ch))


###############################
# **Text + Speech**
def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=256, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  #logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits) # BERT model created
  model.summary()
  


  # Speech model
  speech_inputs = Input(shape=(3400))
  
  model_speech = Dense(512, activation="relu")(speech_inputs)
  #model_speech = SeqSelfAttention(attention_activation='relu')(model_speech)
  model_speech = Dense(256, activation="relu")(model_speech)
  #model_speech = SeqSelfAttention(attention_activation='relu')(model_speech)
  model_speech = Dense(256, activation="relu")(model_speech)
  
  speech_outputs = Dense(256)(model_speech)

  model_speech = Model(speech_inputs, speech_outputs) # Speech model created
  model_speech.summary()
  
  
  
  # Concatenation
  
  concatenated = concatenate([logits, speech_outputs])
  out = Dense(256, activation='relu')(concatenated)
  out = Dense(4, activation='softmax', name='output_layer')(out)

  merged_model = Model([input_ids, speech_inputs], out)
  merged_model.summary()
  

  model.build(input_shape=(None, max_seq_len))
  load_stock_weights(bert, bert_ckpt_file)
  #merged_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  




  
  

        
  return merged_model


###############################  
#**Training**
classes = train.emotion.unique().tolist()

data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
data.train_x.shape
data.train_x[0]
data.train_y[0]
data.max_seq_len
model = create_model(data.max_seq_len, bert_ckpt_file)

###############################
model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

###############################
log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

###############################
history = model.fit( x=[data.train_x, xtrain_sp_ch], y=data.train_y,
                    validation_split=0.2,
                    batch_size=72,
                    shuffle=True,
                    epochs=50,
                    callbacks=[tensorboard_callback]
                  )

###############################                  
#**Evaluation**
import tensorboard
%reload_ext tensorboard
%tensorboard --logdir log
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.show();
train_acc = model.evaluate([data.train_x, xtrain_sp_ch], data.train_y)
test_acc = model.evaluate([data.test_x, xtest_sp_ch], data.test_y)

print("train acc", train_acc)
print("test acc", test_acc)
y_pred = model.predict([data.test_x, xtest_sp_ch]).argmax(axis=-1)
print(classification_report(data.test_y, y_pred, target_names=classes))
cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
#print(cm) 
#print(df_cm)

# Normalize CM
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(norm_cm)

df_norm_cm = pd.DataFrame(norm_cm, index=classes, columns=classes)
#print(df_norm_cm)
hmap = sns.heatmap(df_norm_cm, annot=True, fmt="0.1f")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label');
sentences = [
  "What's your age?",
  "you are an awful individual"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict([data.test_x, xtest_sp_ch]).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()



###############################  
#**Text + Mocap**
x_train_mocap = []
counter = 0
for ses_mod in data2:
    x_head = ses_mod['mocap_head']
    if(x_head.shape != (200,18)):
        x_head = np.zeros((200,18))   
    x_head[np.isnan(x_head)]=0
    x_hand = ses_mod['mocap_hand']
    if(x_hand.shape != (200,6)):
        x_hand = np.zeros((200,6))   
    x_hand[np.isnan(x_hand)]=0
    x_rot = ses_mod['mocap_rot']
    if(x_rot.shape != (200,165)):
        x_rot = np.zeros((200,165))   
    x_rot[np.isnan(x_rot)]=0
    x_mocap = np.concatenate((x_head, x_hand), axis=1)
    x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
    x_train_mocap.append( x_mocap )
    
x_train_mocap = np.array(x_train_mocap)
x_train_mocap = x_train_mocap.reshape(-1,200,189,1)
x_train_mocap.shape
x_train_mocap2 = x_train_mocap.reshape(-1,200,189)
xtrain_mo = x_train_mocap2[:3838]
xtest_mo = x_train_mocap2[3838:]
def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=256, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  #logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits) # BERT model created
  model.summary()
  
  
  #model_mocap = Sequential()
  mocap_inputs = Input(shape=(200, 189))
  model_mocap = GRU(512, return_sequences=True)(mocap_inputs)
  model_mocap = SeqSelfAttention(attention_activation='relu')(model_mocap)
  model_mocap = GRU(256, return_sequences=True)(model_mocap)
  model_mocap = SeqSelfAttention(attention_activation='relu')(model_mocap)
  model_mocap = GRU(128, return_sequences=False)(model_mocap)
  mocap_outputs = Dense(256)(model_mocap)

  model_mocap = Model(mocap_inputs, mocap_outputs)
  model_mocap.summary()
    
  # Concatenation
  
  concatenated = concatenate([logits, mocap_outputs])
  out = Dense(256, activation='relu')(concatenated)
  out = Dense(4, activation='softmax', name='output_layer')(out)

  merged_model1 = Model([input_ids, mocap_inputs], out)
  merged_model1.summary()
  

  model.build(input_shape=(None, max_seq_len))
  load_stock_weights(bert, bert_ckpt_file)
  #merged_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  




  
  

        
  return merged_model1
classes = train.emotion.unique().tolist()

data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
data.train_x.shape
data.train_x[0]
data.train_y[0]
data.max_seq_len
model = create_model(data.max_seq_len, bert_ckpt_file)
model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit( x=[data.train_x, xtrain_mo], y=data.train_y,
                    validation_split=0.2,
                    batch_size=50,
                    shuffle=True,
                    epochs=50,
                    callbacks=[tensorboard_callback]
                  )
#**Evaluation**
%load_ext tensorboard
%tensorboard --logdir=log/intent_detection/
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.show();
train_acc = model.evaluate([data.train_x, xtrain_mo], data.train_y)
test_acc = model.evaluate([data.test_x, xtest_mo], data.test_y)

print("train acc", train_acc)
print("test acc", test_acc)
y_pred = model.predict([data.test_x, xtest_mo]).argmax(axis=-1)
print(classification_report(data.test_y, y_pred, target_names=classes))
cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
#print(cm) 
#print(df_cm)

# Normalize CM
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(norm_cm)

df_norm_cm = pd.DataFrame(norm_cm, index=classes, columns=classes)
#print(df_norm_cm)
hmap = sns.heatmap(df_norm_cm, annot=True, fmt="0.1f", cmap="YlGnBu")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label');
sentences = [
  "What's your age?",
  "you are an awful individual"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict([data.test_x, xtest_mo]).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()


###############################  
# **Text + Mocap 2**

x_train_mocap = []
counter = 0
for ses_mod in data2:
    x_head = ses_mod['mocap_head']
    if(x_head.shape != (200,18)):
        x_head = np.zeros((200,18))   
    x_head[np.isnan(x_head)]=0
    x_hand = ses_mod['mocap_hand']
    if(x_hand.shape != (200,6)):
        x_hand = np.zeros((200,6))   
    x_hand[np.isnan(x_hand)]=0
    x_rot = ses_mod['mocap_rot']
    if(x_rot.shape != (200,165)):
        x_rot = np.zeros((200,165))   
    x_rot[np.isnan(x_rot)]=0
    x_mocap = np.concatenate((x_head, x_hand), axis=1)
    x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
    x_train_mocap.append( x_mocap )
    
x_train_mocap = np.array(x_train_mocap)
x_train_mocap = x_train_mocap.reshape(-1,200,189,1)
x_train_mocap.shape
def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=256, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  #logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits) # BERT model created
  model.summary()
  
  
  #model_mocap = Sequential()
  mocap_inputs = Input(shape=(200, 189, 1))
  model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same') (mocap_inputs)
  model_mocap = Dropout(0.2)(model_mocap)
  model_mocap = Activation('relu')(model_mocap)
  model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same')(model_mocap)
  model_mocap = Dropout(0.2)(model_mocap)
  model_mocap = Activation('relu')(model_mocap)
  model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same')(model_mocap)
  model_mocap = Dropout(0.2)(model_mocap)
  model_mocap = Activation('relu')(model_mocap)
  model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same')(model_mocap)
  model_mocap = Dropout(0.2)(model_mocap)
  model_mocap = Activation('relu')(model_mocap)
  model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same')(model_mocap)
  model_mocap = Dropout(0.2)(model_mocap)
  model_mocap = Activation('relu')(model_mocap)
  model_mocap = Flatten()(model_mocap)
  mocap_outputs = Dense(256)(model_mocap)

  model_mocap = Model(mocap_inputs, mocap_outputs)
  model_mocap.summary()
    
  # Concatenation
  
  concatenated = concatenate([logits, mocap_outputs])
  out = Dense(256, activation='relu')(concatenated)
  out = Dense(4, activation='softmax', name='output_layer')(out)

  merged_model2 = Model([input_ids, mocap_inputs], out)
  merged_model2.summary()
  

  model.build(input_shape=(None, max_seq_len))
  load_stock_weights(bert, bert_ckpt_file)
  #merged_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  




  
  

        
  return merged_model2
classes = train.emotion.unique().tolist()

data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
data.train_x.shape
data.train_x[0]
data.train_y[0]
data.max_seq_len
model = create_model(data.max_seq_len, bert_ckpt_file)
model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
xtrain_mo = x_train_mocap[:3838]
xtest_mo = x_train_mocap[3838:]
log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit( x=[data.train_x, xtrain_mo], y=data.train_y,
                    validation_split=0.2,
                    batch_size=45,
                    shuffle=True,
                    epochs=50,
                    callbacks=[tensorboard_callback]
                  )


#**Evaluation**
%load_ext tensorboard
%tensorboard --logdir=log/intent_detection/
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.show();
train_acc = model.evaluate([data.train_x, xtrain_mo], data.train_y)
test_acc = model.evaluate([data.test_x, xtest_mo], data.test_y)

print("train acc", train_acc)
print("test acc", test_acc)
y_pred = model.predict([data.test_x, xtest_mo]).argmax(axis=-1)
print(classification_report(data.test_y, y_pred, target_names=classes))
cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
#print(cm) 
#print(df_cm)

# Normalize CM
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(norm_cm)

df_norm_cm = pd.DataFrame(norm_cm, index=classes, columns=classes)
#print(df_norm_cm)
hmap = sns.heatmap(df_norm_cm, annot=True, fmt="0.1f", cmap="cool")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label');
sentences = [
  "What's your age?",
  "you are an awful individual"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict([data.test_x, xtest_mo]).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()


###############################  
# **Speech + Mocap**
x_train_mocap = []
counter = 0
for ses_mod in data2:
    x_head = ses_mod['mocap_head']
    if(x_head.shape != (200,18)):
        x_head = np.zeros((200,18))   
    x_head[np.isnan(x_head)]=0
    x_hand = ses_mod['mocap_hand']
    if(x_hand.shape != (200,6)):
        x_hand = np.zeros((200,6))   
    x_hand[np.isnan(x_hand)]=0
    x_rot = ses_mod['mocap_rot']
    if(x_rot.shape != (200,165)):
        x_rot = np.zeros((200,165))   
    x_rot[np.isnan(x_rot)]=0
    x_mocap = np.concatenate((x_head, x_hand), axis=1)
    x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
    x_train_mocap.append( x_mocap )
    
x_train_mocap = np.array(x_train_mocap)
x_train_mocap = x_train_mocap.reshape(-1,200,189,1)
x_train_mocap.shape
x_train_mocap2 = x_train_mocap.reshape(-1,200,189)
xtrain_mo = x_train_mocap2[:3838]
xtest_mo = x_train_mocap2[3838:]
ytest_mo = Y[3838:]
from tensorflow.keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import*
import numpy as np

#from tensorflow.keras.layers import Embedding
# Speech model
speech_inputs = Input(shape=(100, 34))

model_speech = GRU(512, return_sequences=True, activation="relu")(speech_inputs)
model_speech = SeqSelfAttention(attention_activation='relu')(model_speech)
model_speech = GRU(256, return_sequences=True, activation="relu")(model_speech)
model_speech = SeqSelfAttention(attention_activation='relu')(model_speech)
model_speech = GRU(128, return_sequences=False, activation="relu")(model_speech)

speech_outputs = Dense(256)(model_speech)

model_speech = Model(speech_inputs, speech_outputs) # Speech model created
model_speech.summary()


#model_mocap = Sequential()
mocap_inputs = Input(shape=(200, 189))
model_mocap = GRU(512, return_sequences=True)(mocap_inputs)
model_mocap = SeqSelfAttention(attention_activation='relu')(model_mocap)
model_mocap = GRU(256, return_sequences=True)(model_mocap)
model_mocap = SeqSelfAttention(attention_activation='relu')(model_mocap)
model_mocap = GRU(128, return_sequences=False)(model_mocap)
mocap_outputs = Dense(256)(model_mocap)

model_mocap = Model(mocap_inputs, mocap_outputs)
model_mocap.summary()

# Concatenation

concatenated = concatenate([speech_outputs, mocap_outputs])
out = Dense(256, activation='relu')(concatenated)
out = Dense(4, activation='softmax', name='output_layer')(out)

merged_model3 = Model([speech_inputs, mocap_inputs], out)
merged_model3.summary()



merged_model3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


#model.compile(
merged_model3.summary()

print("Model3 Built")
hist = merged_model3.fit([np.array(xtrain_sp), xtrain_mo], np.array(ytrain_sp), 
                 batch_size=100, epochs=2, verbose=1, shuffle=True, 
                 validation_data=([np.array(xtest_sp), xtest_mo], np.array(ytest_sp)))

                 
#**Evaluation**
%load_ext tensorboard
%tensorboard --logdir log
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.show();
train_acc = model.evaluate([data.train_x, xtrain_sp_ch,xtrain_mo], data.train_y)
test_acc = model.evaluate([data.test_x, xtest_sp_ch, xtest_mo], data.test_y)

print("train acc", train_acc)
print("test acc", test_acc)
y_pred = model.predict([data.test_x, xtest_sp_ch, xtest_mo]).argmax(axis=-1)
print(classification_report(data.test_y, y_pred, target_names=classes))
cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
#print(cm) 
#print(df_cm)

# Normalize CM
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(norm_cm)

df_norm_cm = pd.DataFrame(norm_cm, index=classes, columns=classes)
#print(df_norm_cm)
hmap = sns.heatmap(df_norm_cm, annot=True, fmt="0.1f", cmap="YlGnBu")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label');
sentences = [
  "What's your age?",
  "you are an awful individual"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict([data.test_x, xtest_sp_ch, xtest_mo]).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()
xtrain_mo = x_train_mocap[:3838]
xtest_mo = x_train_mocap[3838:]
import keras
from tensorflow.keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import*
from tensorflow.keras.layers import Embedding

speech_inputs = Input(shape=(100, 34))
model_speech = GRU(128, activation='relu', return_sequences=True)(speech_inputs)
model_speech = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(model_speech)
model_speech = GRU(128, activation='relu', return_sequences=True)(model_speech)
model_speech = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(model_speech)
model_speech = GRU(128, activation='relu', return_sequences=False)(model_speech)
model_speech = Dropout(0.2)(model_speech)
speech_outputs = Dense(256)(model_speech)

model_speech = Model(speech_inputs, speech_outputs)

#model_mocap = Sequential()
mocap_inputs = Input(shape=(200, 189, 1))
model_mocap = Conv2D(32, 3, strides=(2, 2), padding='same') (mocap_inputs)
model_mocap = Dropout(0.2)(model_mocap)
model_mocap = Activation('relu')(model_mocap)
model_mocap = Conv2D(64, 3, strides=(2, 2), padding='same')(model_mocap)
model_mocap = Dropout(0.2)(model_mocap)
model_mocap = Activation('relu')(model_mocap)
model_mocap = Conv2D(64, 3, strides=(2, 2), padding='same')(model_mocap)
model_mocap = Dropout(0.2)(model_mocap)
model_mocap = Activation('relu')(model_mocap)
model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same')(model_mocap)
model_mocap = Dropout(0.2)(model_mocap)
model_mocap = Activation('relu')(model_mocap)
model_mocap = Conv2D(128, 3, strides=(2, 2), padding='same')(model_mocap)
model_mocap = Dropout(0.2)(model_mocap)
model_mocap = Activation('relu')(model_mocap)
model_mocap = Flatten()(model_mocap)
mocap_outputs = Dense(256)(model_mocap)

model_mocap = Model(mocap_inputs, mocap_outputs)

concatenated = concatenate([speech_outputs, mocap_outputs])

model_combined = Activation('relu')(concatenated)
out = Dense(256)(model_combined)
model_combined = Activation('relu')(out)

out = Dense(4, activation='softmax')(model_combined)

speech_mocap = Model([speech_inputs, mocap_inputs], out)
speech_mocap.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



#model.compile()
model_speech.summary()
model_mocap.summary()
speech_mocap.summary()

print("Speech + Mocap")
hist = speech_mocap.fit([xtrain_sp, xtrain_mo], ytrain_sp, 
                 batch_size=100, epochs=50, verbose=1,
                 validation_split=0.2)
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

acc = hist.history["accuracy"]
val_acc = hist.history["val_accuracy"]
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]

epochs = range(len(acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 5])
ax1.plot(epochs, acc, 'k')
ax1.plot(epochs, val_acc, 'm')
ax1.set_title('Model speech + mocap accuracy')
ax1.legend(['Model accuracy','Model Val accuracy'])

ax2.plot(epochs, loss, 'k')
ax2.plot(epochs, val_loss, 'm')
ax2.set_title('Model speech + mocap loss')
ax2.legend(['Model loss','Model Val loss'])

plt.show()
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn import metrics
loss, acc = speech_mocap.evaluate([xtrain_sp,xtrain_mo], ytrain_sp,verbose = 0)
print("Training Loss {:.5f} and Training Accuracy {:.2f}%".format(loss,acc*100))

loss, acc = speech_mocap.evaluate([xtest_sp,xtest_mo], ytest_sp,verbose = 0)
print("Validation Loss {:.5f} and Validation Accuracy {:.2f}%".format(loss,acc*100))
from sklearn.metrics import classification_report
# CR untuk Training Data
print("model speech + mocap")
pred = speech_mocap.predict([xtrain_sp,xtrain_mo])
labels = (pred > 0.5).astype(np.int)

print(classification_report(ytrain_sp, labels, target_names = emotions_used))
# CR untuk Validation Data
print("model speech + mocap")
pred = speech_mocap.predict([xtest_sp,xtest_mo])
labels = (pred > 0.5).astype(np.int)

print(classification_report(ytest_sp, labels, target_names = emotions_used))
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.BuPu):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
      print('Confusion matrix, without normalization')

    cm =np.around(cm, decimals=2)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
import itertools
Y_pred = speech_mocap.predict([xtest_sp,xtest_mo])
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(ytest_sp,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
print("Model speech + mocap")
plot_confusion_matrix(confusion_mtx, classes = emotions_used) 

