from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from argparse import ArgumentParser
import pandas as pd
import pickle
import numpy as np
def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--m', default=None, type=str, help = "path to the model")
	parser.add_argument('--p', default=None, type=str, help = "path to the prediction set")
	parser.add_argument('--o', default=None, type=str, help = "path to the output file")
	args = parser.parse_args()
	return args
def main():
	#load the model
	try:
		model = load_model(args.m)
	except: 
		raise Exception("Invalid path to model")
	print("model loaded")
	#read the prediction set 
	try:
		pred_set = pd.read_csv(args.p, sep = ' ', header =None).values
		y_set = model.predict(pred_set[:, 1:])
	except:
		raise Exception("Invalid prediction set")	
	#write predictions to the output file
	words = pred_set[:, 0]
	word_dict = dict(zip(words, y_set))
	for word in word_dict:
		with open(args.o, 'a') as f:
			try:
				f.write(word+" ")
				np.savetxt(f, word_dict[word].reshape(1, word_dict[word].shape[0]))	
			except:
				raise Exception("there was a problem writing your embeddings")
	




