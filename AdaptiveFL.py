#!/usr/bin/env python
# coding: utf-8

import collections
from loguru import logger
logger.add("Log_file_afl2.log")
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
tff.backends.reference.set_reference_context();
import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math
import json
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

@tff.federated_computation
def hello_world():
    return 'Hello, World!'

hello_world()

params={}
params['NUM_CLIENTS']=4
params['BATCH_SIZE']=100
params['num_rounds']=2
params['tau']=10
params['input_size']=60
params['hidden_size']=200
params['target_size']=10
params['sampled_clients']=100
params['learning_rate'] = 0.01
params['init_alpha']=0.01
params['dataset']='synthetic'
params['loaded_synthetic']=0
train_path = "./mytrain.json"
test_path = "./mytest.json"

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta, iid,NUM_USER):

    dimension = 60
    NUM_CLASS = 10
    
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        print(mean_x[i])

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_split, y_split



def gen_data_synthetic(NUM_USER = 100):


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_path = "./mytrain.json"
    test_path = "./mytest.json"

    #X, y = generate_synthetic(alpha=0, beta=0, iid=0)     # synthetiv (0,0)
#     X, y = generate_synthetic(alpha=0.5, beta=0.5, iid=0) # synthetic (0.5, 0.5)
    X, y = generate_synthetic(alpha=1, beta=1, iid=0,NUM_USER=NUM_USER)     # synthetic (1,1)
    #X, y = generate_synthetic(alpha=0, beta=0, iid=1)      # synthetic_IID


    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    for i in trange(NUM_USER, ncols=120):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)


    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

    return train_data,test_data
    
def preprocess_synthetic(dataset):

    def batch_format_fn(element_x,element_y):
        return collections.OrderedDict(
        x=element_x,
        y=tf.reshape(element_y, [-1, 1]))
    return [batch_format_fn(dataset['x'][i*params['BATCH_SIZE']:(i+1)*params['BATCH_SIZE']],dataset['y'][i*params['BATCH_SIZE']:(i+1)*params['BATCH_SIZE']]) for i in range( max(1,int(len(dataset['x'])/(params['BATCH_SIZE']))))]

params['NUM_EPOCHS'] = 5
params['SHUFFLE_BUFFER'] = 100
params['PREFETCH_BUFFER'] = 10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.dtypes.cast(tf.reshape(1-element['pixels'], [-1, 784]),tf.float64),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(params['NUM_EPOCHS']).shuffle(params['SHUFFLE_BUFFER']).batch(
      params['BATCH_SIZE']).map(batch_format_fn).prefetch(params['PREFETCH_BUFFER'])



if params['dataset']=='synthetic':
    if params['loaded_synthetic']==1:
        with open(train_path,'r') as outfile:
            train_data=json.load( outfile)
    
        with open(test_path,'r') as outfile:
            test_data=json.load( outfile)
    else:
        train_data,test_data=gen_data_synthetic(params['NUM_CLIENTS'])
    initializer = tf.initializers.GlorotUniform(seed=0)
    
    initial_model = collections.OrderedDict(
    weights_1=tf.dtypes.cast(tf.Variable(initializer(shape=[60,10],dtype=tf.float32)),tf.float32).numpy(),
    bias_1=np.zeros([10], dtype=np.float32),
    )
#     train_data,test_data=gen_data_synthetic(params['NUM_CLIENTS'])
    client_train_data=[preprocess_synthetic(train_data['user_data'][i]) for i in train_data['users']]
    client_test_data=[preprocess_synthetic(test_data['user_data'][i]) for i in test_data['users']]
    def forward_pass(variables,batch):
        y=tf.nn.sigmoid(tf.matmul(batch['x'],variables['weights_1'])+variables['bias_1'])
        predictions = tf.argmax(y, 1)

        flat_labels = tf.reshape(batch['y'], [-1])

        loss = -tf.reduce_mean(
          tf.reduce_sum(tf.one_hot(tf.cast(flat_labels,dtype=tf.uint8), 10) * tf.math.log( y ), axis=[1]))

        return predictions,loss
    def batch_loss(model, batch):
        return forward_pass(model, batch)
# print(batch_loss(global_model_vars, sample_batch))
    
    layer_names="weights_1 bias_1"
    layers=layer_names.split()
elif params['dataset']=='EMNIST':
    initializer = tf.initializers.GlorotUniform(seed=0)
    
    initial_model = collections.OrderedDict(
    weights_1=tf.dtypes.cast(tf.Variable(initializer(shape=[784,200],dtype=tf.float64)),tf.float64).numpy(),
#     weights_1=    
    bias_1=np.zeros([200], dtype=np.float64),
    weights_2=tf.dtypes.cast(tf.Variable(initializer(shape=[200,200],dtype=tf.float32)),tf.float64).numpy(),
    bias_2=np.zeros([200], dtype=np.float64),
    weights_3=tf.dtypes.cast(tf.Variable(initializer(shape=[200,10],dtype=tf.float32)),tf.float64).numpy(),
    bias_3=np.zeros([10], dtype=np.float64),
    )
    train_dataset,test_dataset=tff.simulation.datasets.emnist.load_data(
    only_digits=True, cache_dir=None
)
    client_train_data=[preprocess(client_dataset[i]) for i in range(len(client_dataset))]
    client_test_data=[preprocess(client_test_dataset[i]) for i in range(len(client_test_dataset))]
    client_test_data=client_test_data[:params['NUM_CLIENTS']]
    client_train_data=client_train_data[:params['NUM_CLIENTS']]
    def forward_pass(variables, batch):
        y = tf.nn.relu(tf.matmul(batch['x'], variables['weights_1']) + variables['bias_1'])
        y = tf.nn.relu(tf.matmul(y,variables['weights_2'])+variables['bias_2'])
        y = tf.nn.softmax(tf.matmul(y,variables['weights_3'])+variables['bias_3'])
        predictions = tf.cast(tf.argmax(y, 1), tf.int32)

        flat_labels = tf.reshape(batch['y'], [-1])
        loss = -tf.reduce_mean(
          tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(tf.cast(y,dtype=tf.float32)), axis=[1]))
        accuracy = tf.reduce_mean(
          tf.cast(tf.equal(predictions, flat_labels), tf.float32)) 
        num_examples = tf.cast(tf.size(batch['y']), tf.float32)

        return  predictions,loss

    def batch_loss(model, batch):
        return forward_pass(model, batch)
    layer_names="weights_1 bias_1 weights_2 bias_2 weights_3 bias_3"
    layers=layer_names.split()
# print(batch_loss(global_model_vars, sample_batch))
else:
    print("Not implemented on this")
    exit()

def batch_train(initial_model, batch, learning_rate,flag):
  # Define a group of model variables and set them to `initial_model`. Must
  # be defined outside the @tf.function.
    global_model_vars = initial_model['global']
    private_model_vars=initial_model['private']
    personalized_model_vars=initial_model['personalized']
    global_model_vars = collections.OrderedDict([
      (name, tf.Variable(name=name, initial_value=value ))
      for name,value in global_model_vars.items()])
    private_model_vars = collections.OrderedDict([
      (name, tf.Variable(name=name, initial_value=value))
      for name,value in private_model_vars.items()])
    personalized_model_vars = collections.OrderedDict([
      (name, tf.Variable(name=name, initial_value=value))
      for name,value in personalized_model_vars.items()])
    
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    @tf.function
    def _train_on_batch(model_vars, batch,flag=0,model_vars_opt=None):
        loss_sum=tf.constant(0,tf.float32)
        with tf.GradientTape() as tape:
            predictions,loss = forward_pass(model_vars, batch)
        loss_sum += loss* tf.cast(params['BATCH_SIZE'], tf.float32)
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
        zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars_opt)))
        return model_vars_opt,loss_sum,grads

    if flag==0:
        return _train_on_batch(global_model_vars, batch,flag,global_model_vars)
    else:
        return _train_on_batch(personalized_model_vars, batch,flag,private_model_vars)
    

def local_train(model, learning_rate, all_batches,alph):

    alphaval=alph
    def batch_fn(model, batch,flag):
        return batch_train(model, batch, learning_rate,flag)
    l_local=[];
    l_global=[]
    flag=0
    k=0
    for i in all_batches:
        
        flag=1
        model['global'],losses,grads=batch_fn(model,i,0)
        l_global.append(losses)
        model['private'],losses,grads=batch_fn(model,i,1)
        l_local.append(losses)
        d={}
#     Commenting
        if alphaval>1:
            alphaval=1
        if alphaval<0:
            alphaval=0
        for lname in layers:
            s=model['private'][lname]
            t=model['global'][lname]
            model['personalized'][lname]=s*alphaval+(1-alphaval)*t
        v=0.0
        for lname in layers:
            s=model['private'][lname]
            t=model['global'][lname]
            g=grads[lname]
            diff=s-t
            g=tf.reshape(g,-1,1)
            diff=tf.reshape(diff,-1,1)
            val=tf.tensordot(g,diff,axes=1).numpy()
            v+=val
        alphaval=alphaval - learning_rate*v
    return model,l_global,l_local,alphaval

def federated_train(models, client_lrs, data,client_ids):
    cur_loss_global=[];
    cur_loss_local=[]
    gradlist=[]
    n_items=0
    #Change code here to add scheuling logic per client
    client_lrs=[params['learning_rate'] for i in range(params['NUM_CLIENTS'])]
    for i in client_ids:
        print(alphas[i])
        models[i],loss_global,loss_local,alphas[i]=local_train(models[i],client_lrs[i],data[i],alphas[i])
        print(alphas[i])
        
        cur_loss_local.append(sum(loss_local))
        cur_loss_global.append(sum(loss_global))

    return models,sum(cur_loss_global)/len(client_ids),sum(cur_loss_local)/len(client_ids)

models=[]
# Init all to samew weights initially
for i in range(params['NUM_CLIENTS']):
    model={}
    model['global']=initial_model
    model['private']=initial_model
    model['personalized']=initial_model
    models.append(model)

alphas=[]
v=np.float64(params['init_alpha'])
for i in range(params['NUM_CLIENTS']):
    alphas.append(v)
    
client_ids=[i for i in range(params['NUM_CLIENTS'])]

#  Can add smapling logic too
sampled_clients=params['NUM_CLIENTS']
logger.info(params)
logger.info("Started FedAdaptive on Synthetic Dataset")
client_lrs=[params['learning_rate'] for i in range(params['NUM_CLIENTS'])]
prev_loss=0
lossValues=[]
lossValueslocal=[]
for round_num in range(params['num_rounds']):
    if((round_num+1)%params['tau']==0):
        for i in tqdm(range(len(client_ids))):
            if i==0:
                global_model=models[i]['global']
            else:
                for lname in layers:
                    s=global_model[lname]
                    t=models[i]['global'][lname]
                    global_model[lname]=s+t
        for lname in layers:
            global_model[lname]/=sampled_clients
        for i in range(len(client_ids)):
            models[i]['global']=global_model
    else:
        models,prev_loss_global,prev_loss_local=federated_train(models, client_lrs, client_train_data,client_ids)
        
    logger.info('round {}, loss={}'.format(round_num, prev_loss_global))
    lossValues.append(prev_loss_global)
    lossValueslocal.append(prev_loss_local)
    
    
"""
Optional to visualize results


len(lossValues)

lv2=[i for i in lossValueslocal]
lv=[i for i in lossValues]
fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.set_size_inches=(10,10)
fig.set_size_inches(10,5)
fig.suptitle('Comparison of FedAvg and AdaptiveFL on IID Synthetic dataset')
ax1.plot(l,color='r',label='FedAvg')
ax1.set_title("Training Loss vs Rounds")
ax1.set_xlabel("Training Rounds")
ax1.set_ylabel("Loss")

ax1.plot(lv,color='g',label='AdaptiveFL(tau=10)')
ax1.legend()
ax2.set_title("Distribution of Alphas(starting from 0.01)")
ax2.plot(plt.hist(alphas))

plt.title("Synthetic(1,1)")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.plot(l,color='r',label='FedAvg')
plt.plot(lv,color='b',label='AdaptiveFL')
# plt.plot(lv2,color='g',label='Local AdaptiveFL')
plt.legend()
plt.show()
"""
