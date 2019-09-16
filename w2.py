import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pymagnitude import *
from scipy import stats
import spacy
nlp = spacy.load('en_core_web_lg')

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--e", type=str, default="total", help="The type of evaluation set to be used")
	parser.add_argument("--v", type=str, default=None, help="Path to the original vectors")
	parser.add_argument("--d", type=str, help="Compositionality type")
	parser.add_argument("--p", type=str, help= "type of tag")
	parser.add_argument("--t", type=str, help= "tag list")
	args = parser.parse_args()
	return args

def distance(word1, word2):
	dot_product = np.dot(word1, word2)
	mag_word1 = np.linalg.norm(word1)
	mag_word2 = np.linalg.norm(word2)
	return (dot_product/(mag_word1 * mag_word2))

def main():
	w1 = {
	"NUM": -0.39999999999999997,
	"NOUN": 4.400000000000001,
	"SPACE": 1, 
	"DET": 1.5999999999999999,
	"ADV": 1.7999999999999998,
	"ADJ": 2.8000000000000003,
	"ADP": 2.4,
	"VERB": 3.0000000000000004,
	"PROPN": 9.599999999999998,
	"PRON": 2.6,
	"INTJ": 1,
	"CCONJ": 0.4000000000000001,
	"PART": 1.5999999999999999,
	"X": -8.200000000000003,
	"PUNCT": 1
	}


	w2 = {
        "NUM": 0.000000000001,
        "NOUN": 3.8,
        "SPACE": 1,
        "DET": .6,
        "ADV": 1.7999999999999998,
        "ADJ": 1.4,
        "ADP": 1.4,
        "VERB": 3.0000000000000004,
        "PROPN": 11.2,
        "PRON": 1,
        "INTJ": 1,
        "CCONJ": 0.4000000000000001,
        "PART": .8,
        "X": -9.4,
        "PUNCT": 1
        }
		

	w = {
	"NUM":0,
	"NOUN": 6.200000000000001,
	"SPACE": 1,
	"DET": 1,
	"ADV": 5.0000000000000007,
	"ADJ": 1.8000000000000001,
	"ADP":3.2,
	"VERB": 5,
	"PROPN": -4.4,
	"PRON": .6,
	"INTJ": 1,
	"CCONJ": 0.8,
	"PART": 3,
	"X": 2.2,
	"PUNCT": 1
	}
	#d = {}
	print("Loading embeddings")
	args = parse_args()
	vecs = Magnitude(args.v)
	print("Embeddings loaded!")
	tags = [] 
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	for line in open(args.t, 'r'):
		tag = line.strip()
		
		one = correlation(eval_set, args.d, vecs, tag, args.p, w[tag], w)
		up = correlation(eval_set, args.d, vecs, tag, args.p, w[tag]+.2, w)
		#print(one, up)
		if up > one:
			weight = w[tag]+.2
			last = one
			new = up
			while new > last:
				weight += .2
				last = new
				new = correlation(eval_set, args.d, vecs, tag, args.p, weight, w)
			w[tag] = weight
			print(tag, weight)
		
		else:
			down = correlation(eval_set, args.d, vecs, tag, args.p, w[tag]-.2, w)
			if down > one:
				weight = w[tag]-.2
				last = one
				new = down
				while new > last:
					weight = weight-.2
					last = new
					new = correlation(eval_set, args.d, vecs, tag, args.p, weight, w)
				w[tag] = weight
				print(tag, weight)
			else:
				up = correlation(eval_set, args.d, vecs, tag, args.p, w[tag]+1, w)
				if up > one:
					weight = w[tag]+1
					last = one
					new = up
					while new > last:
						weight += .2
						last = new
						new = correlation(eval_set, args.d, vecs, tag, args.p, weight, w)
					w[tag] = weight
					print(tag, weight)
				else:
					down = correlation(eval_set, args.d, vecs, tag, args.p, w[tag]-1, w)
					if down > one:
						weight = w[tag]-1
						last = one
						new = down
						while new > last:
							weight = weight-.2
							last = new
							new = correlation(eval_set, args.d, vecs, tag, args.p, weight, w)
						w[tag] = weight
						print(tag, weight)
					else:
						w[tag] = w[tag]
						print(tag, w[tag])


def correlation(eval_set, comp, vecs, pos, typ, weight, dict):
	sim = []
	act0 = []
	act1 = []
	act2 = []
	for i in range(eval_set.shape[0]):
		w1 = get_embedding(eval_set[i][0], comp, vecs, pos, typ, weight, dict)
		w2 = get_embedding(eval_set[i][1], comp, vecs, pos, typ, weight, dict)
		if (len(w1) == len(w2)) and (len(w1) != 0):
			this_sim = distance(w1, w2)
			sim.append(this_sim)
			actuals = eval_set[i][3].split(',')
			act0.append(float(actuals[0]))
			act1.append(float(actuals[1]))
			act2.append(float(actuals[2]))
	sim = np.asarray(sim)
	act0 = np.asarray(act0)
	act1 = np.asarray(act1)
	act2 = np.asarray(act2)
	cor0, pval0 = stats.spearmanr(sim, act0)
	cor1, pval1 = stats.spearmanr(sim, act1)
	cor2, pval2 = stats.spearmanr(sim, act2)
	coravg = (cor0+cor1+cor2)/3
	return coravg	



	

def get_embedding(string, comp, vecs, pos, typ, weight, dictionary):
	if pd.isnull(string):
		return []
	else:
		if comp == "dilate":
			return dilate(string, vecs, pos, typ, weight)
		if comp == "normal":
			return normal(string, vecs, pos, typ, weight, dictionary)

def normal(string, vecs, pos, typ, weight, dictionary):
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	if typ == "google":
		for token in word:
			if token.pos_ == pos:
				new.append(weight*vecs.query(token.text))
			#elif token.pos_ in dictionary:
			else:
				new.append((dictionary[token.pos_])*vecs.query(token.text))
			#else:
			#	new.append(vecs.query(token.text))
	elif typ == "tag":
		for token in word:
			if token.tag_ == pos:
				new.append(weight*vecs.query(token.text))
			
			else:
			#elif token.pos_ in dictionary:
				new.append((dictionary[token.pos_])*vecs.query(token.text))
			#else:
			#	new.append(vecs.query(token.text))
	elif typ == "dep":
		for token in word:
			if token.dep_ == pos:
				new.append(weight*vecs.query(token.text))
			else:
				new.append((dictionary[token.pos_])*vecs.query(token.text))
			#else:
			#	new.append(vecs.query(token.text))
	try:
		return add(new)
	except:
		return []		
		
	
def dilate(string, vecs, pos, typ, weight):
	return 

def add(vecs):
        c = vecs[0]
        for i in range(len(vecs)-1):
                c = c+(vecs[i+1])
        return c

if __name__ == '__main__':
        main()


