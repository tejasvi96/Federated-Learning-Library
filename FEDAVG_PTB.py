# This code is adapted from TensorFlowFederated  and some modifications have been made

import os
from loguru import logger
logger.add("logs.log")

from torchnlp.datasets import penn_treebank_dataset
logger.info("Started the experiment")
# Load the ptb training dataset
train = penn_treebank_dataset(train=True)


params={}
params['num_workers']=50
params['fraction']=0.1
params['hidden_size']=500
params['fraction']=0.1
params['max_length']=22
params['embedding_size']=300
params['buffer_size']=100
params['batch_size']=4
params['num_rounds']=50
logger.info("params")
logger.info(params)

datalist=[]
sent=[]
for i in (train):
    if i=="</s>":
        datalist.append(sent)
        sent=[]
    else:
        sent.append(i)

import unicodedata,re
SOS_token = 1
EOS_token = 2
pad_token = 0
unk_token=3
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "SOS", 2: "EOS",0:"<pad>",3:"<unk>"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
eng=Lang("English")
for i,sent in enumerate(datalist):
    sent=" ".join(i for i in sent)
    datalist[i]=normalizeString(sent)

for i in datalist:
    eng.addSentence(i)
from torchnlp.word_to_vector import GloVe
import numpy as np
pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in eng.word2index.keys())
embedding_weights = np.zeros((eng.n_words, pretrained_embedding.dim))
for i, token in eng.index2word.items():
    embedding_weights[i] = pretrained_embedding[token]


def word_tokenize(sent,lang):
    tokenized_sent=[]
    tokenized_sent.append("SOS")
    for word in sent.split():
        if word in lang.word2index.keys():
            tokenized_sent.append(word)
        else:
            tokenized_sent.append("<unk>")
    tokenized_sent.append("EOS")
    l=len(tokenized_sent)
    if(l!=params['max_length']):
        if(l<params['max_length']):
            tokenized_sent=tokenized_sent+(["<pad>"]*(params['max_length']-l))
        else:
            tokenized_sent=tokenized_sent[:params['max_length']-1]+["EOS"]
    return tokenized_sent
    
    

# Creating actual mini clients from the PTB dataset
import collections
tokenized_sents=[word_tokenize(i,eng) for i in datalist]
client_train_dataset = collections.OrderedDict()
data_per_worker=int(len(tokenized_sents)/params['num_workers'])
for i in range(1, params['num_workers']+1):
    client_name = "client_" + str(i)
    start = data_per_worker * (i-1)
    end = data_per_worker * i
    data = collections.OrderedDict((('x', tokenized_sents[start:end]),('y',tokenized_sents[start:end])))
    client_train_dataset[client_name] = data


# Federated Learning Pipeline
import collections
import functools
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
nest_asyncio.apply()

np.random.seed(0)

# Test the TFF is working:
# tff.federated_computation(lambda: 'Hello, World!')()
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Masking

def build_model(vocab_size,embedding_size, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, params['embedding_size'],weights=[embedding_weights],
                                  batch_input_shape=[batch_size, None],trainable=False),
        tf.keras.layers.LSTM(rnn_units,return_sequences=True,input_shape=[params['max_length'],params['embedding_size']]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            ),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model=build_model(eng.n_words,params['embedding_size'],params['hidden_size'],params['batch_size'])


# Sample function to generate text and test our model
def generate_text(model, start_string):
    # From https://www.tensorflow.org/tutorials/sequences/text_generation
    num_generate = 200
    input_eval = [eng.word2index[s] if s in eng.word2index.keys() else pad_token for s in start_string.split()]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(eng.index2word[predicted_id])

    return (start_string + ' '.join(text_generated))

generate_text(model,"hello there shakespeare ")


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='accuracy', dtype=tf.float32):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, eng.n_words, 1])
        return super().update_state(y_true, y_pred, sample_weight)


# Lookup for indexes of words
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=list(eng.word2index.keys()), values=tf.constant(list(eng.word2index.values()),
                                       dtype=tf.int64)),
    default_value=0)

# Function to tokenize a sentence
def to_ids(x):
    s=tf.reshape(x['x'],shape=[22])
    chars = tf.strings.split(s).values
    ids=table.lookup(chars)
    return ids



def split_input_target(chunk):
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    target_text = tf.map_fn(lambda x: x[1:], chunk)
    return (input_text, target_text)    
# Dataset preprocessing
def preprocess(dataset):
    return (

      dataset.map(to_ids)


      .batch(params['batch_size'], drop_remainder=True)
      .map(split_input_target))

train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)


sample_data=preprocess(train_dataset.create_tf_dataset_for_client('client_1'))
# Creates a model taking reference from a KEras model
def create_tff_model():
  # TFF uses an `input_spec` so it knows the types and shapes
  # that your model expects.
    input_spec = sample_data.element_spec
    keras_model_clone = keras.models.clone_model(keras_model)
    return tff.learning.from_keras_model(
      keras_model_clone,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[FlattenedCategoricalAccuracy()])

keras_model = build_model(eng.n_words,params['embedding_size'],params['hidden_size'],batch_size=params['batch_size'])
keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()])




import tensorflow_federated as tff


import keras
fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5))
    
# How many datapoints to take out from each client
def data(client, source=train_dataset):
    return preprocess(source.create_tf_dataset_for_client(client)).take(800)

clients = [
    'client_1', 'client_2','client_3','client_4','client_5','client_6','client_7','client_8','client_9','client_10',
]
train_datasets = [data(client) for client in clients]

test_clients=['client_12','client_13']
test_dataset = tf.data.Dataset.from_tensor_slices(
    [data(client, train_dataset) for client in test_clients]).flat_map(lambda x: x)


# data preprocessing done here

# The state of the FL server, containing the model and optimization state.
state = fed_avg.initialize()

# Load the pre-trained Keras model weights into the global model state.
state = tff.learning.state_with_new_model_weights(
    state,
    trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in keras_model.non_trainable_weights
    ])


def keras_evaluate(state, round_num):
  # Take our global model weights and push them back into a Keras model to
  # use its standard `.evaluate()` method.
    keras_model = build_model(eng.n_words,params['embedding_size'],params['hidden_size'],params['batch_size'])
    keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[FlattenedCategoricalAccuracy()])
    state.model.assign_weights_to(keras_model)
    loss, accuracy = keras_model.evaluate(sample_data, steps=2, verbose=0)
    print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))

# Performing the actual training FL rounds
for round_num in range(params['num_rounds']):
    print('Round {r}'.format(r=round_num))
    keras_evaluate(state, round_num)
    state, metrics = fed_avg.next(state, train_datasets)
    train_metrics = metrics['train']
    print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
      l=train_metrics['loss'], a=train_metrics['accuracy']))
    logger.info(train_metrics['loss'])
    logger.info(train_metrics['accuracy'])  
    
# Add code to save the model as well