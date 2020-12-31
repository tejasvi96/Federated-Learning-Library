#refer paper
# https://arxiv.org/pdf/1812.07108.pdf
# Implementing on the MNIST MLP network
import collections
from loguru import logger
logger.add("Log_file.log")
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
tff.backends.reference.set_reference_context()

@tff.federated_computation
def hello_world():
    return 'Hello, World!'

hello_world()

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

params={}
params['NUM_EXAMPLES_PER_USER']=1000
params['BATCH_SIZE']=100
params['threshold']=2
params['num_rounds']=50

params['num_clients']=10
params['learning_rate'] = 0.2

# Mix of digits
def get_data_iid(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if i<(digit+1)*params['NUM_EXAMPLES_PER_USER'] and i>=digit*(params['NUM_EXAMPLES_PER_USER']) ]
    for i in range(0, min(len(all_samples), params['NUM_EXAMPLES_PER_USER']), params['BATCH_SIZE']):
        batch_samples = all_samples[i:i + params['BATCH_SIZE']]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence



# Each client acts as non iid here as a different digit is assigned
def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples),params['NUM_EXAMPLES_PER_USER']), params['BATCH_SIZE']):
        batch_samples = all_samples[i:i + params['BATCH_SIZE']]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence


federated_train_data = [get_data_iid(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_iid(mnist_test, (d)) for d in range(10)]

BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)

str(BATCH_TYPE)
MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)

print(MODEL_TYPE)

initial_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))

sample_batch = federated_train_data[5][-1]
@tf.function
def forward_pass(model, batch):
    predicted_y = tf.nn.softmax(
      tf.matmul(batch['x'], model['weights']) + model['bias'])
    return predicted_y,-tf.reduce_mean(
      tf.reduce_sum(
          tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1]))

# @tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)
batch_loss(initial_model, sample_batch)

# @tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
  # Define a group of model variables and set them to `initial_model`. Must
  # be defined outside the @tf.function.
    model_vars = collections.OrderedDict([
      (name, tf.Variable(name=name, initial_value=value))
      for name, value in initial_model.items()
  ])
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
    # Perform one step of gradient descent using loss from `batch_loss`.
        loss_sum=tf.constant(0,tf.float32)
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        loss_sum += loss * tf.cast(params['BATCH_SIZE'], tf.float32)
        grads = tape.gradient(loss, model_vars)
    
        optimizer.apply_gradients(
        zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
        return model_vars,loss_sum,grads

    return _train_on_batch(model_vars, batch)

LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

# @tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):

  # Mapping function to apply to each batch.
#   @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)
    l=[];
    g=[]
#   print("D_lengths",len(all_batches))  
    for i in all_batches:
        model,losses,grads=batch_fn(initial_model,i)
        l.append(losses)
        g.append(grads)
    return model,l,g
#   return tff.sequence_reduce(all_batches, initial_model, batch_fn)

@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):

    return tff.sequence_sum(
      tff.sequence_map(
          tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
          all_batches))

SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)
CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)

@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
      tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))

def calc_acc():
    ele=0
    out=0
    for i in range(0,10):
        elems=0
        outs=0    
        for j in federated_test_data[i]:
            elems+=int(j['x'].shape[0])
            sm,outputs=batch_loss(model,j)
            ans=[bool(k) for k in tf.argmax(sm,1)==j['y']]
            temp=sum([1 for k in ans if k is True])
            outs+=temp
        ele+=elems
        out+=outs
    return(out/ele)    


SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)
gradlist=[]
g1=[]
def federated_train(model, learning_rate, data,prev_loss):
    cur_loss=[];
    weights_mse=[]
    bias_mse=[]
    weights=[]
    bias=[]
    optimizer_bias = tf.keras.optimizers.SGD(learning_rate)
    optimizer_weights = tf.keras.optimizers.SGD(learning_rate)
#   layerwise optimizer but a common learning rate
    bias_gradlist=[]
    weights_gradlist=[]
    n_items=0  
    for i in data:
        outs,loss,grads=local_train(model,learning_rate,i)
        n_items+=(len(loss))
        weights_mse.append(tf.norm(outs['weights']-model['weights'],ord=1))
        bias_mse.append(tf.norm(outs['bias']-model['bias'],ord=1))
        weights.append(outs['weights'])
        bias.append(outs['bias'])
        cur_loss.append(sum(loss))
        #       Shannon
      
      #todo change this to KL divergence
    model_vars = collections.OrderedDict([
    (name, tf.Variable(name=name, initial_value=value))
    for name, value in model.items()
  ])

    print('use fedattn')
    weights_mse=tf.constant(np.array(weights_mse).reshape(1,-1))    
    probs=tf.keras.activations.softmax(weights_mse)
    bias_mse=tf.constant(np.array(bias_mse).reshape(1,-1))
    b_probs=tf.keras.activations.softmax(bias_mse)
    for nums in range(num_clients):
        optimizer_bias.apply_gradients(zip(tf.nest.flatten(b_probs[0][nums]*(model_vars['bias']-bias[nums])),tf.nest.flatten(model_vars['bias'])))
        optimizer_weights.apply_gradients(zip(tf.nest.flatten(probs[0][nums]*(model_vars['weights']-weights[nums])),tf.nest.flatten(model_vars['weights'])))
    flag=0  
    acc=calc_acc()  
    print(acc)  
    return model_vars,sum(cur_loss)/n_items,flag


model = initial_model
prev_loss=0
# scale the gradeints and apply the layerwise gradeints
lossValues=[]
for round_num in range(params['num_rounds']):
    model,prev_loss,flag=federated_train(model, params['learning_rate'], federated_train_data,prev_loss)
    if flag==1:
        break
    print('round {}, loss={}'.format(round_num, prev_loss))
    lossValues.append(prev_loss)
    if flag==1:
        break
