import sys
sys.path.append('/home/ec2-user/sapang/lib64/python2.7/dist-packages/')
import random, time, os, decoder
from PIL import Image
import numpy as np
import tensorflow as tf

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
import abc
from collections import namedtuple
from pydoc import locate

import six
import random, time, os
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

def load_data():
	vocab = open('/home/ec2-user/files/latex_vocab.txt').read().split('\n')
	vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
	formulas = open('/home/ec2-user/files/formulas.norm.lst').read().split('\n')

  # four meta keywords
  # 0: START
  # 1: END
  # 2: UNKNOWN
  # 3: PADDING

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
		if(len(formulas[int(z[1])])<=60 and len(formulas[int(z[1])])>=3):
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

	input_train = batchify(train_list, batch_size)

	input_val=batchify(val_list,batch_size)
	input_test=batchify(test_list,batch_size)
	
	vocab_size=0
	max_decode_length=0
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

	return input_train,input_val,input_test
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')
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
def weight_variable(name,shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.get_variable(name + "_weights", initializer= initial)

def bias_variable(name, shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.get_variable(name + "_bias", initializer= initial)						
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

	return h_bn6




def build_model(inp, batch_size, num_rows, num_columns, dec_seq_len):
  #constants
	enc_lstm_dim = 256
	feat_size = 512
	dec_lstm_dim = 512
	vocab_size = 503
	embedding_size = 80

	cnn = init_cnn(inp)

  #f unction for map to apply the rnn to each row
	def fn(inp):
		enc_init_shape = [batch_size, enc_lstm_dim]
		with tf.variable_scope('encoder_rnn'):
			with tf.variable_scope('forward'):
				lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
				init_fw = tf.nn.rnn_cell.LSTMStateTuple(\
					tf.get_variable("enc_fw_c", enc_init_shape),\
					tf.get_variable("enc_fw_h", enc_init_shape)
					)
			with tf.variable_scope('backward'):
				lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
				init_bw = tf.nn.rnn_cell.LSTMStateTuple(\
					tf.get_variable("enc_bw_c", enc_init_shape),\
					tf.get_variable("enc_bw_h", enc_init_shape)
					)
			output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, \
                                                  lstm_cell_bw, \
                                                  inp, \
                                                  sequence_length = tf.fill([batch_size],\
                                                                            tf.shape(inp)[1]), \
                                                  initial_state_fw = init_fw, \
                                                  initial_state_bw = init_bw \
                                                  )
		return tf.concat(2,output)

	fun = tf.make_template('fun', fn)

	rows_first = tf.transpose(cnn,[1,0,2,3])
	res = tf.map_fn(fun, rows_first, dtype=tf.float32)
	encoder_output = tf.transpose(res,[1,0,2,3])

	dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(dec_lstm_dim)
	dec_init_shape = [batch_size, dec_lstm_dim]
	dec_init_state = tf.nn.rnn_cell.LSTMStateTuple( tf.truncated_normal(dec_init_shape),\
                                                  tf.truncated_normal(dec_init_shape) )

	init_words = np.zeros([batch_size,1,vocab_size])	

	decoder_output = decoder.embedding_attention_decoder(dec_init_state,\
                                                       tf.reshape(encoder_output,\
                                                                  [batch_size, -1,\
                                                                  2*enc_lstm_dim]),\
                                                       dec_lstm_cell,\
                                                       vocab_size,\
                                                       dec_seq_len,\
                                                       batch_size,\
                                                       embedding_size,\
                                                       feed_previous=True)

	return (encoder_output, decoder_output)

batch_size=16
epochs = 100
lr = 0.1
min_lr = 0.001
learning_rate = tf.placeholder(tf.float32)
inp = tf.placeholder(tf.float32)
num_rows = tf.placeholder(tf.int32)
num_columns = tf.placeholder(tf.int32)
num_words = tf.placeholder(tf.int32)
true_labels = tf.placeholder(tf.int32)
start_time = time.time()

print "Building Model"
_, (output,state) = build_model(inp, batch_size, num_rows, num_columns, num_words)
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output,true_labels))
tf.summary.histogram('cross entropy', cross_entropy)
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)
final_out=tf.to_int32(tf.argmax( output, 2))
correct_prediction = tf.equal(final_out, true_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
print "Loading Data"
train, val, test = load_data()
  #train = batchify(train, batch_size)
  #train = sorted(train,key= lambda x: x[1].shape[1])
random.shuffle(train)
  #val = batchify(val, batch_size)
  #test = batchify(test, batch_size)

last_val_acc = 0
reduce_lr = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session() as sess:
	q=0
	sess.run(tf.global_variables_initializer())
	merged = tf.summary.merge_all()
	saver.restore(sess,"/home/ec2-user/data_check/saved_models/model-0/model")
	train_writer =tf.summary.FileWriter("train_loss",sess.graph)
	print "Training"
	for i in range(epochs):
		if reduce_lr == 5:
			lr = max(min_lr, lr-0.005)
			reduce_lr = 0
		print "Epoch %d learning rate %.4f"%(i,lr)
		epoch_start_time = time.time()
		batch_50_start = epoch_start_time
		random.shuffle(train)
		for j in range(len(train)):
			images, labels = train[j]
			if j<5 or j%50==0:
				train_accuracy = accuracy.eval(feed_dict={inp: images,\
												  true_labels:labels,\
												  num_rows: images.shape[1],\
												  num_columns: images.shape[2],\
													  num_words:labels.shape[1]})
				new_time = time.time()
				print("step %d/%d, training accuracy %g, took %f mins"%\
				  (j, len(train), train_accuracy, (new_time - batch_50_start)/60))
				batch_50_start = new_time
			pred,loss_j,summary=sess.run([final_out,cross_entropy,merged],feed_dict={inp: images,\
													  true_labels:labels,\
													  num_rows: images.shape[1],\
													  num_columns: images.shape[2],\
													  num_words:labels.shape[1]})
			train_labels=open("epoch-"+str(i)+"train_labels.txt","a")
			#print(labels)
			#print(pred)
			for k in range(0,batch_size):
				train_labels.write("label  ")
				train_labels.write(" ".join(str(x) for x in labels[k])+"\n")
				train_labels.write("pred   ")
				train_labels.write(" ".join(str(x) for x in pred[k])+'\n')
			train_labels.close()
			print("loss " +str(j)+" "+str(loss_j))
			train_writer.add_summary(summary,q)
			q=q+1
			train_step.run(feed_dict={learning_rate: lr,\
									inp: images,\
									true_labels: labels,\
									num_rows: images.shape[1],\
									num_columns: images.shape[2],\
									num_words: labels.shape[1]})
		print "Time for epoch:%f mins"%((time.time()-epoch_start_time)/60)
		print "Running on Validation Set"
		accs = []
		for j in range(len(val)):
			images, labels = val[j]
			val_accuracy = accuracy.eval(feed_dict={inp: images,\
											  true_labels: labels,\
											  num_rows: images.shape[1],\
											  num_columns: images.shape[2],\
											  num_words: labels.shape[1]})
			accs.append( val_accuracy )
		val_acc = sess.run(tf.reduce_mean(accs))
		acc_val=open("val_acc.txt","a")
		acc_val.write("epoch"+str(i)+"-"+str(val_acc)+"\n")
		acc_val.close()
		if (val_acc - last_val_acc) >= .01:
			reduce_lr = 0
		else:
			reduce_lr = reduce_lr + 1
		last_val_acc = val_acc
		print("val accuracy %g"%val_acc)

		print 'Saving model'
		saver = tf.train.Saver()
		id = 'saved_models/model-'+str(i)
		os.mkdir(id)
		save_path = saver.save(sess, id+'/model' )
		print 'Running on Test Set'
		accs = []
		for j in range(len(test)):
			images, labels = test[j]
			pred,test_accuracy = sess.run([final_out,accuracy],feed_dict={inp: images,\
												  true_labels: labels,\
												  num_rows: images.shape[1],\
												  num_columns: images.shape[2],\
												  num_words: labels.shape[1]})
			accs.append( test_accuracy )
			test_labels=open("epoch-"+str(i)+"test_labels.txt","a")
			for k in range(0,batch_size):
				test_labels.write("label  ")
				test_labels.write(" ".join(str(x) for x in labels[k])+"\n")
				test_labels.write("pred   ")
				test_labels.write(" ".join(str(x) for x in pred[k])+'\n')
			test_labels.close()
		test_acc = sess.run(tf.reduce_mean(accs))
		acc_test=open("test_acc.txt","a")
		acc_test.write("epoch"+str(i)+"-"+str(test_acc)+"\n")
		acc_test.close()
