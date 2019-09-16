from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from argparse import ArgumentParser
import pandas as pd
import pickle
import numpy as np
print('loading model')
model = load_model('/data1/sarah/avg/no_OOV/linear.h5')
print('model loaded')
pred_set = pd.read_csv('/home/paynesa/compositionality/multiplication_words.txt', sep=' ', header=None).values
y_set = model.predict(pred_set[:, 1:])
words = pred_set[:, 0]
print(words.shape[0])
word_dict = dict(zip(words, y_set))
print(len(word_dict))
for word in word_dict:
	with open('multiplication_predictions.txt', 'a') as f:
		try:
			f.write(word+" ")
			np.savetxt(f, word_dict[word].reshape(1, word_dict[word].shape[0]))	
		except:
			print(word)

