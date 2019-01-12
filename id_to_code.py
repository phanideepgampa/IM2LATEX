import numpy as np
import math
import string
vocab = open('/home/maneeshita/Downloads/im2latex-master/data/latex_vocab.txt').read().split('\n')
vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
idx_to_vocab = dict([ (i,vocab[i]) for i in range(len(vocab))])
#for i in range(len(vocab)):
	#print(i,vocab[i])
fin=open('/home/maneeshita/Downloads/train_labels.txt').read().split('\n')
fout=open('/home/maneeshita/Downloads/output_latex.txt', "a")
for line in fin:
	print("HIIIIIIIIIIIIIIIIIIIIII")
	tokens = line.strip().split()
	token_out=[]
	for token in tokens:
		if token.isdigit():
			token=int(token)
			if token != 0 and token!= 2 and token !=1 and token != 3:
				token_out.append(vocab[int(token)-4])
			else:
				token_out.append(str(token))
		else:
			token_out.append(token)
	fout.writelines(''.join(str(token_out))+'\n')

	                    	         	
	                    	
	                    
	              	
	    
	              
	        


