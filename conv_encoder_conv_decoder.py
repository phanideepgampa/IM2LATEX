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
from collections import namedtuple
from pydoc import locate

import six
import random, time, os
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
#img = np.array(Image.open('/home/deeksha/Deep_Learning/im2latex-master/data/images_processed/60ee748793.png').convert('L'))

vocab = open('/home/maneeshita/Downloads/im2latex-master/data/latex_vocab.txt').read().split('\n')
vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
formulas = open('/home/maneeshita/Downloads/im2latex-master/data/formulas.norm.lst').read().split('\n')

# four meta keywords
# 0: START
# 1: END
# 2: UNKNOWN
# 3: PADDING

inp=tf.placeholder(tf.float32)
label=tf.placeholder(tf.int64)



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

train = open('/home/maneeshita/Downloads/im2latex-master/data/valid.lst').read().split('\n')[:-1]
val = open('/home/ec2-user/files/validate_filter.lst').read().split('\n')[:-1]
test = open('/home/ec2-user/files/test_filter.lst').read().split('\n')[:-1]


train_list=[]
test_list=[]
val_list=[]
for i in range(0,len(train)):
	x=train[i].split(' ')
	if(os.path.exists('/home/maneeshita/Downloads/im2latex-master/data/images_processed2/'+str(x[0]))):
		img = np.array(Image.open('/home/maneeshita/Downloads/im2latex-master/data/images_processed2/'+str(x[0])).convert('L'))
		#train(img, formulas[ int(train[1]) ])
		if(len(formulas[int(x[1])])<=60):
			train_list.append([img, formulas[ int(x[1])]])



for i in range(0,len(val)):

	y=val[i].split(' ')
	if(os.path.exists('/home/ec2-user/images_(160_500)/'+str(y[0]))):
		img = np.array(Image.open('/home/ec2-user/images_(160_500)/'+str(y[0])).convert('L'))
		#train(img, formulas[ int(train[1]) ])
		if(len(formulas[int(y[1])])<=60):
			val_list.append([img, formulas[ int(y[1])]])

for i in range(0,len(test)):
	z=test[i].split(' ')
	if(os.path.exists('/home/ec2-user/images_(160_500)/'+str(z[0]))):
		img = np.array(Image.open('/home/ec2-user/images_(160_500)/'+str(z[0])).convert('L'))
		#train(img, formulas[ int(train[1]) ])
		if(len(formulas[int(z[1])])<=60):
			test_list.append([img, formulas[ int(z[1])]])

'''		
train = map(import_images, train)
val = map(import_images, val)
test = map(import_images, test)
#return train
'''

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

print "Loading Data"
#print(len(train_list))
#train, val, test = load_data()he
vocab_size=0
max_decode_length=0
emb_size=128        #change
batch_size=8
max_w=0
max_h=0

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
random.shuffle(input_train)

####################### MODEL ##########################

embed=tf.get_variable("embedding",initializer=tf.truncated_normal([vocab_size,emb_size],stddev=0.1),trainable=True)


def init_cnn(inp):
  def weight_variable(name,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name + "_weights", initializer= initial,trainable=True)

  def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name + "_bias", initializer= initial,trainable=True)

  def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')
  
  W_conv1 = weight_variable("conv1", [3,3,1,256]) #change
  b_conv1 = bias_variable("conv1", [256]) #change
  h_conv1 = tf.nn.relu(conv2d(inp,W_conv1) + b_conv1)
  h_bn1   = tf.contrib.layers.batch_norm(h_conv1)

  W_conv2 = weight_variable("conv2", [3,3,256,256]) #change
  b_conv2 = bias_variable("conv2", [256]) #change
  h_pad2  = tf.pad(h_bn1, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv2 = tf.nn.relu(conv2d(h_pad2, W_conv2) + b_conv2)
  h_bn2   = tf.contrib.layers.batch_norm(h_conv2)
  h_pool2 = tf.nn.max_pool(h_bn2, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

  W_conv3 = weight_variable("conv3", [3,3,256,128]) #change
  b_conv3 = bias_variable("conv3", [128]) #change
  h_pad3  = tf.pad(h_pool2, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv3 = tf.nn.relu(conv2d(h_pad3, W_conv3) + b_conv3)

  h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

  W_conv4 = weight_variable("conv4", [3,3,128,128]) #change
  b_conv4 = bias_variable("conv4", [128]) #change
  h_pad4  = tf.pad(h_pool3, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv4 = tf.nn.relu(conv2d(h_pad4, W_conv4) + b_conv4)
  h_bn4   = tf.contrib.layers.batch_norm(h_conv4)

  W_conv5 = weight_variable("conv5", [3,3,128,64]) #change
  b_conv5 = bias_variable("conv5", [64]) #change
  h_pad5  = tf.pad(h_bn4, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv5 = tf.nn.relu(conv2d(h_pad5, W_conv5) + b_conv5)
  h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  W_conv6 = weight_variable("conv6", [3,3,64,128]) #change
  b_conv6 = bias_variable("conv6", [128]) #change
  h_pad6  = tf.pad(h_pool5, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv6 = tf.nn.relu(conv2d(h_pad6, W_conv6) + b_conv6)
  h_pad6  = tf.pad(h_conv6, [[0,0],[2,2],[2,2],[0,0]], "CONSTANT")
  h_pool6 = tf.nn.max_pool(h_pad6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  return h_pool6
next_layer=init_cnn(inp)


 


################# ENCODER ######################

def weight_variable(name,shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable(name + "_weights", initializer= initial,trainable=True)
def bias_variable(name, shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name + "_bias", initializer= initial,trainable=True)
def conv2d(x,W):
  return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')
def conv1d(x,W):
  return tf.nn.conv1d(x,W, stride=1, padding='SAME')
initial_res=next_layer 

for i in range(0,4):
  res=next_layer
  W = weight_variable("enc_weight"+str(i), [3,3,128,256]) #change5
  b = bias_variable("enc_bias"+str(i), [256]) #change6
  #next_layer=tf.pad(next_layer, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  next_layer=tf.nn.relu(conv2d(next_layer, W) + b)

  A=next_layer[:,:,:,0:128] #change
  B=next_layer[:,:,:,128:256] #change
  B=tf.sigmoid(B)
  next_layer=tf.multiply(A,B)
  next_layer=(next_layer+res)*tf.sqrt(0.5) 

cnn_c=(next_layer+initial_res)*tf.sqrt(0.5)  # doubt

sh=tf.shape(next_layer)
cnn_c=tf.reshape(next_layer,[batch_size,850,emb_size])
next_layer=tf.reshape(next_layer,[batch_size,850,emb_size])

 
####################  DECODER  ########################

start_tokens_batch = tf.fill([batch_size], 0) # fill(dim,val)
inputs = tf.nn.embedding_lookup(embed, start_tokens_batch)  #(embedding,ids)
inputs = tf.expand_dims(inputs, 1)
zeros_padding = tf.zeros([batch_size,max_decode_length-1,emb_size])
k1=tf.shape(inputs)
k2=tf.shape(zeros_padding)
inputs = tf.concat([inputs, zeros_padding], axis=1)
#initial input is matrix of size BW*MAX_DECODE_LENGTH*emb_dim 
# initialized with start tokens and zeros 
#first axis is time axis i.e. maximum decode length


time=0
final_out=tf.zeros([batch_size,1],dtype=tf.int64) #change
logits_for_softmax=tf.zeros([batch_size,1,vocab_size],dtype=tf.float32) #change B_W*1*vocab_size

loss=0

while(time < max_decode_length-1):
	#print('heeeeeeeeeeeeeeeeeee')
	initial_inputs = inputs[:,0:time+1,:] 
	zeros_padding = inputs[:,time+2:,:] 

	cur_inputs=initial_inputs[:,-3:,:]
	cur_inputs = tf.contrib.layers.dropout(
	      inputs=cur_inputs,
	      keep_prob=0.9,is_training=True)

	inp_emb=cur_inputs
	  
	next_layer=cur_inputs

	for i in range(0,3):
	  res=next_layer
	  next_layer = tf.pad(next_layer, [[0, 0], [0,2], [0, 0]], "CONSTANT")
	  Wd=tf.get_variable("wt"+str(i)+str(time),shape=[3,128,256],initializer=None,trainable=True) #change1
	  bd=tf.get_variable("biases"+str(i)+str(time),shape=[256],initializer=None,trainable=True) #change2
	  con=tf.nn.relu(tf.nn.conv1d(next_layer,filters=Wd,stride=1,padding="SAME")+bd)
	  next_layer=con[:,0:-2,:]

	  A=next_layer[:,:,0:128] #change3
	  B=next_layer[:,:,128:256] #change4
	  B=tf.sigmoid(B)
	  next_layer=tf.multiply(A,B)
	  dec_hid=next_layer
	  dec_hid=(dec_hid+cur_inputs)*tf.sqrt(0.5) 
	  att_score = tf.matmul(dec_hid,cnn_c , transpose_b=True) 
	  att_score = tf.nn.softmax(att_score)

	  length = batch_size #change
	  att_out = tf.matmul(att_score, cnn_c) * length * tf.sqrt(1.0/length)  
	  next_layer = (next_layer + att_out) * tf.sqrt(0.5) 
	  next_layer  += (next_layer + res) * tf.sqrt(0.5) 
	next_layer=tf.reduce_sum(next_layer,axis=1)
	dec_out=tf.layers.dense(next_layer,vocab_size,trainable=True,name="ds4"+str(i)+str(time))  
	shape = dec_out.get_shape().as_list()  
	logits = tf.reshape(dec_out,[-1,shape[-1]]) 
	logits=tf.expand_dims(logits,1)				#change added dimension 1 in middle (3*1*502 from 3*502)
	logits_for_softmax=tf.concat([logits_for_softmax,logits],1) #changed axis
	sample_ids = tf.argmax(logits, axis=-1)  #argmax on vocabulary
	next_inputs = tf.nn.embedding_lookup(embed, sample_ids)
	
	final_out=tf.concat([final_out,sample_ids],1) #change axis=1

	next_inputs = tf.reshape(next_inputs, [batch_size, 1,128]) #change
	next_inputs = tf.concat([initial_inputs, next_inputs], axis=1) #concatinate previously predicted labels and next_input
	next_inputs = tf.concat([next_inputs, zeros_padding], axis=1) #pad remaining i.e. MAX_DECODE_LENGTH-(time+1) with zeroes
	next_inputs.set_shape([batch_size, max_decode_length, 128]) #change
	inputs=next_inputs
	time=time+1


one_hot=tf.one_hot(indices=label,depth=vocab_size,on_value=1.0,off_value=0.0,axis=2,dtype=tf.float32)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot,logits=logits_for_softmax))
tf.summary.histogram("loss",loss)
train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)




sess=tf.Session()
list_of_variables = tf.all_variables()
sess.run(tf.global_variables_initializer())
#writer = tf.summary.FileWriter("output", sess.graph)
variable_names= [v.name for v in tf.trainable_variables()]
#print(trainable_variables)
uninitialized_variables = list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables(list_of_variables)))
sess.run(tf.variables_initializer(uninitialized_variables))
print(sess.run(tf.report_uninitialized_variables(list_of_variables)))






with sess.as_default():
	for epoch in range(0,15):
		l=0
		print("######################################")
		merged=tf.summary.merge_all()
		train_writer=tf.summary.FileWriter("/home/ec2-user/events/"+"train_loss",sess.graph)
		for i in range(0,len(input_train)):
			images,labels=input_train[i]
			
			print(train_step.run({inp:images,label:labels}))
			
			print("Predictions")

			print(np.split(final_out.eval({inp:images,label:labels}),batch_size))
			print("labels")
			print(labels)
			
			train_labels=open("train_labels.txt","a")
			for j in range(0,len(labels)):
				train_labels.write(labels[j])
				train_labels.write(final_out[j])
			train_labels.close()
			loss_i,summary=sess.run([loss,merged],feed_dict={inp:images,label:labels})
			#loss_i=loss.eval({inp:images,label:labels})
			train_writer.add_summary(summary,i)
			l=l+loss_i
			val_writer=tf.summary.FileWriter("/home/ec2-user/events/"+"val_loss",sess.graph)
			if(i%400000==0 and i >0 ):
				saver = tf.train.Saver()
				id = 'saved_models/model-epoch'+str(epoch)+time.strftime("%d-%m-%Y--%H-%M")
				os.mkdir(id)
				save_path = saver.save(sess, id+'/model' )
				for  z in range(0,len(input_val)):
					images,labels=input_val[z]
					loss_val,summary=sess.run([loss,merged],feed_dict={inp:images,label:labels})
					val_writer.add_summary(summary,z)
				
			batch_loss=open("batch_loss.txt","a")
			batch_loss.write(str(loss_i)+"\n")
			batch_loss.close()	
				
		print(l/len(input_train))
		epoch_loss=open("epoch_loss.txt","a")
		epoch_loss.write("epoch "+str(epoch)+str(l/len(input_train)+"\n"))
		epoch_loss.close()
		print()
		print()
		print()
		

 
#writer.close()

