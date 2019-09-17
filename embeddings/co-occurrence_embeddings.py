import json, spacy, string, os, operator
import pandas as pd
import numpy as np
p = string.punctuation
nlp = spacy.load("en_core_web_lg")
from parseargs import ArgumentParser

#parse the arguments from the command line
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--i", default=None, type=str, help = "the directory containing the unprocessed BNC files")
	parser.add_argument("--o", default=None, type=str, help = "the file where you would like your embeddings to be written ")
	return args

def vocab():
	args = parse_args()
	d = {}
	for i in os.listdir(args.i):
		for line in open(args.i+i, 'r'):
			doc = line.split()
			for word in doc:
				if word in p or '\n' in word or word.isdigit():
					continue
				else:
					if word in d:
						d[word] += 1
					else:
						d[word] = 1
		print("Done reading {}".format(i))
	sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
	print(len(d)," vocabulary words found. Calculating 2000 most frequent...")
	y = 0 
	with open("vocab.txt", 'a') as f:
		while (y < 4096):
			f.write(sorted_d[y][0]+'\n')
			y += 1
	print(y, "most frequent words added")
	f.close()
	return()

def tcm():
	args = parse_args()
	finaldict = {}
	d = {}
	for line in open('vocab.txt', 'r'):
		d[line.strip()] = 0
	print(d)
	for file in os.listdir(args.i):
		for line in open(args.i+file, 'r'):
			line = line.split()
			for i in range(len(line)-1):
				if line[i] not in finaldict:
					c = d.copy()
					finaldict[line[i]] = c
				x = max(0, i-5)
				y = min(i+5, len(line)-1)
				for k in range (x, y+1):
					if line[k] in d:
						finaldict[line[i]][line[k]] +=1
		print("Done with {}".format(file))
	print("Writing embeddings...")
	for k, v in finaldict.items():
		with open(args.o , 'a') as f:
			f.write(k+" ")
			x = np.asarray(list(v.values()))
			np.savetxt(f, x.reshape(1, x.shape[0]))
if __name__ == '__main__':
	vocab()
	tcm()
