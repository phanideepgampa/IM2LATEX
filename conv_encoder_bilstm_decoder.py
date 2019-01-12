from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
import abc
from decoder import embedding_attention_decoder
from tensorflow.python.ops import array_ops
from collections import namedtuple
from pydoc import locate

import six
import random, time, os
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
#img = np.array(Image.open('/home/deeksha/Deep_Learning/im2latex-master/data/images_processed/60ee748793.png').convert('L'))
vocab = open('/home/ec2-user/files/latex_vocab.txt').read().split('\n')
vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
formulas = open('/home/ec2-user/files/formulas.norm.lst').read().split('\n')

# four meta keywords
# 0: START
# 1: END
# 2: UNKNOWN
# 3: PADDING

inp=tf.placeholder(tf.float32)
label=tf.placeholder(tf.int64)
pad=tf.placeholder(tf.int32)
prob=0.9
is_training=tf.placeholder(tf.int32)
learning_rate = tf.placeholder(tf.float32)


vocab_size=0
max_decode_length=0
emb_size=128       #change
batch_size=16
max_w=0
max_h=0

def cross_entropy_sequence_loss(logits, targets, sequence_length ,max_size_labels=max_decode_length):
  """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape `[T, B, vocab_size]`
    targets: Target classes of shape `[T, B]`
    sequence_length: An int32 tensor of shape `[B]` corresponding
      to the length of each input

  Returns:
    A tensor of shape [T, B] that contains the loss per example, per time step.
  """
  with tf.name_scope("cross_entropy_sequence_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)

    # Mask out the losses we don't care about
    loss_mask = tf.sequence_mask(tf.to_int32(sequence_length), tf.to_int32(max_decode_length))
    losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

    return losses


def _transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(
          ([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t
def formula_to_indices(formula):
  formula = formula.split(' ')
  res = [0]
  for token in formula:
    if token in vocab_to_idx:
      res.append( vocab_to_idx[token] + 4 )
    else:
      res.append(2)
  res.append(1)
  return res

formulas = map( formula_to_indices, formulas)

train = open('/home/ec2-user/files/train_filter.lst').read().split('\n')[:-1]
val = open('/home/ec2-user/files/validate_filter.lst').read().split('\n')[:-1]
test = open('/home/ec2-user/files/test_filter.lst').read().split('\n')[:-1]
train_list=[]
test_list=[]
val_list=[]
for i in range(0,len(train)):
  x=train[i].split(' ')
  if(os.path.exists("/home/ec2-user/images_(160_500)/"+str(x[0]))):
    img = np.array(Image.open("/home/ec2-user/images_(160_500)/"+str(x[0])).convert('L'))
    #train(img, formulas[ int(train[1]) ])
    if(len(formulas[int(x[1])])<=60):
      train_list.append([img, formulas[ int(x[1])]])


print(len(train_list))
for i in range(0,len(val)):

  y=val[i].split(' ')
  if(os.path.exists("/home/ec2-user/images_(160_500)/"+str(y[0]))):
    img = np.array(Image.open("/home/ec2-user/images_(160_500)/"+str(y[0])).convert('L'))
    #train(img, formulas[ int(train[1]) ])
    if(len(formulas[int(y[1])])<=60):
      val_list.append([img, formulas[ int(y[1])]])
print(len(val_list))
for i in range(0,len(test)):
  z=test[i].split(' ')
  if(os.path.exists("/home/ec2-user/images_(160_500)/"+str(z[0]))):
    img = np.array(Image.open("/home/ec2-user/images_(160_500)/"+str(z[0])).convert('L'))
    #train(img, formulas[ int(train[1]) ])
    if(len(formulas[int(z[1])])<=60 and len(formulas[int(z[1])])>=4):
      test_list.append([img, formulas[ int(z[1])]])
print(len(test_list))
def batchify(data, batch_size):
  # group by image size
  res = {}
  for datum in data:
    if datum[0].shape not in res:
      res[datum[0].shape] = [datum]
    else:
      res[datum[0].shape].append(datum)
  batches = []
  for size in res:
    # batch by similar sequence length within each image-size group -- this keeps padding to a
    # minimum
    group = sorted(res[size], key= lambda x: len(x[1]))
    for i in range(0, len(group), batch_size):
      images = map(lambda x: np.expand_dims(np.expand_dims(x[0],0),3), group[i:i+batch_size])
      batch_images = np.concatenate(images, 0)
      seq_len = max([ len(x[1]) for x in group[i:i+batch_size]])
      def preprocess(x):
        arr = np.array(x[1])
        pad = np.pad( arr, (0, seq_len - arr.shape[0]), 'constant', constant_values = 3)
        return np.expand_dims( pad, 0)
      labels = map( preprocess, group[i:i+batch_size])
      batch_labels = np.concatenate(labels, 0)
      too_big = [(160,400),(100,500),(100,360),(60,360),(50,400),\
                (100,800), (200,500), (800,800), (100,600)] # these are only for the test set
      if batch_labels.shape[0] == batch_size\
          and not (batch_images.shape[1],batch_images.shape[2]) in too_big:
        batches.append( (batch_images, batch_labels) )
  #skip the last incomplete batch for now
  return batches


################# PREPROCESSING ########################

print("Loading Data")

#print(len(train_list))
#train, val, test = load_data()he


for image1,label1 in train_list:
  length=len(label1)
  
  for i in range(length):
    x=label1[i]
    if(x>vocab_size):
      vocab_size=x
  if(length>max_decode_length):
    max_decode_length=length
  h=len(image1)
  w=len(image1[0])
  if(max_h<h):
    max_h=h
  if(max_w<w):
    max_w=w

for image1,label1 in val_list:
  length=len(label1)
  for i in range(length):
    x=label1[i]
    if(x>vocab_size):
      vocab_size=x
  if(length>max_decode_length):
    max_decode_length=length
  h=len(image1)
  w=len(image1[0])
  if(max_h<h):
    max_h=h
  if(max_w<w):
    max_w=w
for image1,label1 in test_list:
  length=len(label1)
  
  #print(length)
  for i in range(length):
    x=label1[i]
    if(x>vocab_size):
      vocab_size=x
  if(length>max_decode_length):
    max_decode_length=length
  h=len(image1)
  w=len(image1[0])
  if(max_h<h):
    max_h=h
  if(max_w<w):
    max_w=w

#print(max_decode_length)
vocab_size=vocab_size+1
input_train = batchify(train_list, batch_size)

input_val=batchify(val_list,batch_size)
input_test=batchify(test_list,batch_size)

i=0
for image2,label2 in input_train:
  h_diff=160-image2.shape[1]
  w_diff=500-image2.shape[2]
  s_diff=max_decode_length-label2.shape[1]
  label2=np.pad(label2,((0,0),(0,s_diff)),'constant',constant_values=3)
  #image2=np.pad(image2,((0,0),(h_diff/2,h_diff/2),(w_diff/2,w_diff/2),(0,0)),'constant',constant_values=0)
  input_train[i]=image2,label2
  i=i+1

i=0
for image2,label2 in input_val:
  h_diff=160-image2.shape[1]
  w_diff=500-image2.shape[2]
  s_diff=max_decode_length-label2.shape[1]
  label2=np.pad(label2,((0,0),(0,s_diff)),'constant',constant_values=3)
  #image2=np.pad(image2,((0,0),(h_diff/2,h_diff/2),(w_diff/2,w_diff/2),(0,0)),'constant',constant_values=0)
  input_val[i]=image2,label2
  i=i+1
i=0
for image2,label2 in input_test:
  h_diff=160-image2.shape[1]
  w_diff=500-image2.shape[2]
  s_diff=max_decode_length-label2.shape[1]
  label2=np.pad(label2,((0,0),(0,s_diff)),'constant',constant_values=3)
  #image2=np.pad(image2,((0,0),(h_diff/2,h_diff/2),(w_diff/2,w_diff/2),(0,0)),'constant',constant_values=0)
  input_test[i]=image2,label2
  i=i+1
#random.shuffle(input_train)

####################### MODEL ##########################

embed=tf.get_variable("embedding",initializer=tf.truncated_normal([vocab_size,emb_size],stddev=0.1),trainable=True)
def max_pool(
  name,
  l_input,
  k,
  s
  ):
  """
  Max pooling operation with kernel size k and stride s on input with NCHW data format

  :parameters:
      l_input: input in NCHW data format
      k: tuple of int, or int ; kernel size
      s: tuple of int, or int ; stride value
  """

  if type(k)==int:
      k1=k
      k2=k
  else:
      k1 = k[0]
      k2 = k[1]
  if type(s)==int:
      s1=s
      s2=s
  else:
      s1 = s[0]
      s2 = s[1]
  return tf.nn.max_pool(l_input, ksize=[1, k1, k2, 1], strides=[1, s1, s1, 1],
                        padding='SAME', name=name)

def conv2d_weight_norm(inputs,name, in_dim,out_dim,dropout=1.0, padding="SAME"):
  with tf.variable_scope(name):
    V = tf.get_variable('V'+name, shape=[3,3, in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0*1.0/(3*in_dim))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=[0,1])  # V shape is M*eN*k,  V_norm shape is k  
    g = tf.get_variable('g'+name, dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b'+name, shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

    # use weight normalization (Salimans & Kingma, 2016)
    W = tf.reshape(g, [1,1,in_dim,out_dim])*tf.nn.l2_normalize(V,[0,1])
    inputs = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input=inputs, filter=V, strides=[1,1,1,1], padding=padding), b))
  return inputs

def linear_mapping_weightnorm(inputs,name,out_dim,f, dropout=1.0):
  with tf.variable_scope(name,reuse=None) as scope:
    if(f==1):
      scope.reuse_variables()
    input_shape = inputs.get_shape().as_list()  
      # static shape. may has None
    in_dim = int(inputs.get_shape()[-1])
    input_shape_tensor = tf.shape(inputs)    
    # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
    V = tf.get_variable('V_linear'+name, shape=[int(input_shape[-1]), out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/int(input_shape[-1]))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
    g = tf.get_variable('g_linear'+name, dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b_linear'+name, shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)   # weightnorm bias is init zero
    scope.reuse_variables()
    #assert len(input_shape) == 3
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])
    inputs = tf.matmul(inputs, V)
    inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
    #inputs = tf.matmul(inputs, V)    # x*v
    
    scaler = tf.div(g, tf.norm(V, axis=0))   # g/2-norm(v)
    inputs = tf.reshape(scaler,[1, out_dim])*inputs + tf.reshape(b,[1, out_dim])   # x*v g/2-norm(v) + b
  return inputs 
#inputs=next_layer, name=str(i),out_dim=256, kernel_size=3,f, padding="SAME", dropout=1.0
def conv1d_weightnorm(inputs, name,out_dim,f,kernel_size=3,padding="SAME", dropout=1.0): #padding should take attention

  with tf.variable_scope(name) as scope:
    if(f==1):
      scope.reuse_variables() 
    in_dim = int(inputs.get_shape()[-1])
    V = tf.get_variable('V_conv1'+name, shape=[kernel_size, in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0*dropout/(kernel_size*in_dim))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=[0,1])  # V shape is M*eN*k,  V_norm shape is k  
    g = tf.get_variable('g_conv1'+name, dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b_conv1'+name, shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    scope.reuse_variables()
    # use weight normalization (Salimans & Kingma, 2016)
    W = tf.reshape(g, [1,1,out_dim])*tf.nn.l2_normalize(V,[0,1])
    inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)   
  return inputs
  '''
  #V = get_variable("enc_weight"+str(i),shape= [3,3,256,256],dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0*1.0/(kernel_size*in_dim))), trainable=True) #change5
  #V_norm = tf.norm(W.initialized_value(), axis=[0,1,2])
  #g = tf.get_variable('g'+str(i), dtype=tf.float32, initializer=V_norm, trainable=True)
  #b = bias_variable("enc_bias"+str(i), [256]) #change6
  '''

  #next_layer=tf.pad(next_layer, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')
def weight_variable(name,shape):
  initial = tf.contrib.layers.xavier_initializer(uniform=True)
  return tf.get_variable(name + "_weights", shape=shape,initializer= initial,trainable=True)

def bias_variable(name, shape):
  initial = tf.zeros_initializer()
  return tf.get_variable(name + "_bias",shape=shape, initializer= initial,trainable=True)
def init_cnn(inp):
  inp=inp-128.
  inp=inp/128.



  
  W_conv1 = weight_variable("conv1", [3,3,1,64])
  b_conv1 = bias_variable("conv1", [64])
  h_conv1 = tf.nn.relu(conv2d(inp,W_conv1) + b_conv1)
  h_pool1=max_pool('pool1', h_conv1, k=2, s=2)

  W_conv2 = weight_variable("conv2", [3,3,64,128])
  b_conv2 = bias_variable("conv2", [128])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool('pool2', h_conv2, k=2, s=2)

  W_conv3 = weight_variable("conv3", [3,3,128,256])
  b_conv3 = bias_variable("conv3", [256])
  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  h_bn3  = tf.contrib.layers.batch_norm(h_conv3)

  W_conv4 = weight_variable("conv4", [3,3,256,256])
  b_conv4 = bias_variable("conv4", [256])
  h_conv4 = tf.nn.relu(conv2d(h_bn3, W_conv4) + b_conv4)
  h_pool4=max_pool('pool4', h_conv4, k=(1,2), s=(1,2))

  W_conv5 = weight_variable("conv5", [3,3,256,512])
  b_conv5 = bias_variable("conv5", [512])
  h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
  h_bn5   = tf.contrib.layers.batch_norm(h_conv5)
  h_pool5 = max_pool('pool5', h_bn5, k=(2,1), s=(2,1))

  W_conv6 = weight_variable("conv6", [3,3,512,512])
  b_conv6 = bias_variable("conv6", [512])
  h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
  h_bn6   = tf.contrib.layers.batch_norm(h_conv6)

  W_conv7 = weight_variable("conv7", [3,3,512,128])
  b_conv7 = bias_variable("conv7", [128])
  h_conv7 = tf.nn.relu(conv2d(h_bn6, W_conv7) + b_conv7)
  h_bn7   = tf.contrib.layers.batch_norm(h_conv7)
  return h_bn7


next_layer=init_cnn(inp)
#next_layer=linear_mapping_weightnorm(inputs=next_layer,name="next_layer",out_dim=128, in_dim=512, dropout=1.0)
sz=tf.shape(next_layer)

################# ENCODER ######################

initial_res=next_layer


initial_res=next_layer 

for i in range(0,4):
  res=next_layer
  #W = weight_variable("enc_weight"+str(i), [3,3,128,256]) #change5
  #b = bias_variable("enc_bias"+str(i), [256]) #change6
  #next_layer=tf.nn.relu(conv2d(next_layer, W) + b)           #change
  next_layer=conv2d_weight_norm(inputs=next_layer,name=str(i),in_dim=128,out_dim=256,dropout=1.0,padding="SAME")

  A=next_layer[:,:,:,0:128]
  B=next_layer[:,:,:,128:256]
  B=tf.sigmoid(B)
  next_layer=tf.multiply(A,B)
  next_layer=(next_layer+res)*tf.sqrt(0.5) 

cnn_c=(next_layer+initial_res)*tf.sqrt(0.5) 
sh=tf.shape(cnn_c)

cnn_c=tf.reshape(cnn_c,[batch_size,-1,emb_size])########################to be changed############################
cnn_a=tf.reshape(next_layer,[batch_size,-1,emb_size])
attention_states=tf.stack([cnn_c,cnn_a])
attention_states=tf.reshape(attention_states,[batch_size,-1,2*emb_size])
####################  DECODER  ########################



loss=0
embedding_size=128
dec_lstm_dim = 512     #################### CHECK KRLE YE 512 KYU H.......512 ke alawa kuch me bhi error aata h
dec_lstm_cell = tf.contrib.rnn.LSTMCell (dec_lstm_dim)
dec_init_shape = [batch_size, dec_lstm_dim]                       ############### ye truncated normal h ################
dec_init_state = tf.contrib.rnn.LSTMStateTuple( tf.truncated_normal(dec_init_shape),tf.truncated_normal(dec_init_shape) )
init_words = np.zeros([batch_size,1,vocab_size])
(output,state) = embedding_attention_decoder(dec_init_state,attention_states,dec_lstm_cell,vocab_size,
	max_decode_length,batch_size,embedding_size,feed_previous=True)
temp_logits=_transpose_batch_time(output)
temp_label=_transpose_batch_time(label)

sequence_length=tf.tile(pad,[batch_size])
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=label))
loss=tf.reduce_mean(cross_entropy_sequence_loss(logits=temp_logits,targets=temp_label,sequence_length=sequence_length ,max_size_labels=max_decode_length))

tf.summary.histogram("loss",loss)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.to_int64(tf.argmax( output, 2)), label)

                                                      
                                                       

saver = tf.train.Saver()

sess=tf.Session()
list_of_variables = tf.all_variables()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("output", sess.graph)
variable_names= [v.name for v in tf.trainable_variables()]
print(variable_names)
uninitialized_variables = list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables(list_of_variables)))
sess.run(tf.variables_initializer(uninitialized_variables))
print(sess.run(tf.report_uninitialized_variables(list_of_variables)))




#images,labels=input_train[0]
#print((sh.eval({inp:images,label:labels})))





last_val_acc=0
reduce_lr=0
t=0

with sess.as_default():

	merged=tf.summary.merge_all()
	train_writer=tf.summary.FileWriter("train_loss",sess.graph)
	#saver.restore(sess,"/home/ec2-user/ole/saved_models/model-epoch30/model")  ##########commented
	last_val_acc=0
	reduce_lr = 0
	epochs=100
	#raw_input()
	print("start")
	for epoch in range(0,epochs):
		random.shuffle(input_train)
		l=0
		#print(len(input_train))
		for i in range(0,len(input_train)):
			images,labels=input_train[i]	
			max_true=0
			for r in range(0,batch_size):
				s=max_decode_length-1
				while(labels[r][s]==3):
					s=s-1
				max_true=max(max_true,s)
      #print(sess.run(sz.eval({inp:images,label:labels})))
			print(train_step.run({inp:images,label:labels,pad:[max_true],is_training:[1]}))
			pred=np.array_split((np.argmax(output.eval({inp:images,label:labels,pad:[max_true],is_training:[0]}),2)),batch_size)

			train_labels=open("epoch-"+str(epoch)+"train_labels.txt","a")
			for j in range(0,batch_size):
				train_labels.write("label  ")
				#print("labels")
				#print(" ".join(str(x) for x in labels[j]))
				train_labels.write(" ".join(str(x) for x in labels[j])+"\n")
				train_labels.write("pred   ")
				#print("pred")
				#print(" ".join(str(x) for x in pred[j][0]))
				train_labels.write(" ".join(str(x) for x in pred[j][0])+'\n')		
			train_labels.close()						
			loss_i,summary=sess.run([loss,merged],feed_dict={inp:images,label:labels,pad:[max_true],is_training:[0]})
			#loss_i=loss.eval({inp:images,label:labels})
			#loss_i=sess.run(loss,feed_dict={inp:images,label:labels,pad:[max_true],is_training:[0]})
			train_writer.add_summary(summary,t)
			t=t+1
			l=l+loss_i
			print("loss " +str(i)+" "+str(loss_i)+"\n")

			if(i%2750==0 and i >0 ):                                              ########commented
				id = "/home/ec2-user/epoch4/saved_models/model-epoch"+str(epoch)+" "+str(i)
				os.mkdir(id)
				save_path = saver.save(sess, id+'/model' )
			zx=len(input_train)/5
			if(i%zx==0 and i>0):
				random.shuffle(input_val)
				l_val=0
				for  z in range(0,20):
					images,labels=input_val[z]
					max_true_v=0
					for rv in range(0,batch_size):
						sv=max_decode_length-1
						while(labels[rv][sv]==3):
							sv=sv-1
						max_true_v=max(max_true_v,sv)
					loss_val=sess.run((loss),feed_dict={inp:images,label:labels,pad:[max_true_v],is_training:[0]})
					l_val=l_val+loss_val
				val=open("val.txt","a")
				val.write("epoch-"+str(epoch)+"-"+str(i)+" "+str(l_val/20)+"\n")
				val.close()
			batch_loss=open("batch_loss.txt","a")
			batch_loss.write("loss " +str(i)+" "+str(loss_i)+"\n")
			batch_loss.close()			
		#print(l/len(input_train))
		epoch_loss=open("batch_loss.txt","a")
		
		id = "/home/ec2-user/bilstm_decoder/saved_models/model-epoch"+str(epoch) ########commented
		os.mkdir(id)
		save_path = saver.save(sess, id+'/model' )
		epoch_loss.write("epoch "+str(epoch)+"-"+str((l/len(input_train)))+"\n")
		epoch_loss.close()
		
 		
		print("test_data")
		k=0
		l1=0
		for i in range(0,len(input_test)):
			images,labels=input_test[i]
			max_true_t=0
			for rt in range(0,batch_size):
				st=max_decode_length-1
				while(labels[rt][st]==3):
					st=st-1
				max_true_t=max(max_true_t,st)
			pred=np.array_split((np.argmax(output.eval({inp:images,label:labels,pad:[max_true],is_training:[0]}),2)),batch_size)
			for i1 in range(0,batch_size):
				for j in range(0,len(pred[i1][0])):
					if labels[i1][j]==pred[i1][0][j] and labels[i1][j] !='3' :
						k=k+1
					if labels[i1][j]!='3':
						l1=l1+1
			acc=(float(k)*100)/float(l1)
			test_labels=open("epoch-"+str(epoch)+"test_labels.txt","a")
			for j in range(0,batch_size):
				test_labels.write("label  ")
				#print("labels")
				#print(" ".join(str(x) for x in labels[j]))
				test_labels.write(" ".join(str(x) for x in labels[j])+"\n")
				test_labels.write("pred   ")
				#print("pred")
				#print(" ".join(str(x) for x in pred[j][0]))
				test_labels.write(" ".join(str(x) for x in pred[j][0])+'\n')		
			test_labels.close()
		#print(k)
		#print(l)
		
		print("test "+str(acc)+"\n")
		test_acc=open("test_acc.txt","a")
		test_acc.write("test-epoch-"+str(epoch)+"-"+str(acc)+"\n")
		test_acc.close()
	train_writer.close()

						
		
