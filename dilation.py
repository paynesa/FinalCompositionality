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
	d2  = {
        "NOUN": 3.900000000000002,
        "DET": 0.0000000000000000001,
        "ADV": 0.8999999999999999,
        "ADJ": 0.8999999999999999,
        "ADP": 0.7999999999999999,
        "VERB": 1.4000000000000001,
        "NUM": 0.0000000000000001,
        "PRON": 0.9999999999999999,
        "INTJ": 3.900000000000002,
        "CCONJ": 0.4,
        "PART": 0.7,
        "PROPN": 0.000000000000000001,
        "X": 3.900000000000002,
        "PUNCT": 3.900000000
        }

	d1 = {}
	d2 = {
        "NN": 3.3000000000000016,
        "DT": 0.0000000000000001,
        "RB": 0.8999999999999999,
        "IN": 0.7999999999999999,
        "JJ": 0.8999999999999999,
        "WRB": 3.900000000000002,
        "CD": 0.0000000000000001,
        "WDT": 3.900000000000002,
        "VB": 1.0999999999999999,
        "PRP": 0.9999999999999999,
        "UH": 3.900000000000002,
        "CC": 0.4,
        "WP": 3.900000000000002,
        "VBP": 1.0999999999999999,
        "VBN": 0.8999999999999999,
        "TO": 0.5,
        "PRP$": 1.0999999999999999,
        "NNP": 0.0000000000000001,
        "VBD": 1.5000000000000002,
        "RP": 0.7,
        "RBR": 0.9999999999999999,
        "VBG": 0.6,
        "NNS": 0.7,
        "JJR": 3.900000000000002,
        "MD": 3.900000000000002,
        "JJS": 0.7999999999999999,
        "RBS": 0.7,
        "WP$": 3.900000000000002,
        "FW": 3.900000000000002,
        "XX": 3.900000000000002,
        "LS": 3.900000000000002,
        "PDT": 0.00000000000000001,
        "EX": 3.900000000000002,
        "VBZ": 3.900000000000002,
        ".": 3.900000000000002,
        "AFX": 3.900000000000002
        }

	d2 = {}
	print("Loading embeddings")
	args = parse_args()
	vecs = Magnitude(args.v)
	print("Embeddings loaded!")
	tags = [] 
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	for line in open(args.t, 'r'):
		tag = line.strip()
		one = correlation(eval_set, 1, vecs, tag, args.p, 1, d1, d2)
		up = correlation(eval_set, 1, vecs, tag, args.p, 1.2, d1, d2)
		print(one, up)
		if up > one:
			weight = 1.2
			last = one
			new = up
			while new > last:
				weight += .2
				last = new
				new = correlation(eval_set, 1, vecs, tag, args.p, weight, d1, d2)
			d1[tag] = weight
			print(tag, weight)
		
		else:
			down = correlation(eval_set, 1, vecs, tag, args.p, .8, d1, d2)
			if down > one:
				weight = .8
				last = one
				new = down
				while new > last:
					weight = weight-.2
					last = new
					new = correlation(eval_set, 1, vecs, tag, args.p, weight, d1, d2)
				d1[tag] = weight
				print(tag, weight)
			else:
				up = correlation(eval_set, 1, vecs, tag, args.p, 2, d1, d2)
				if up > one:
					weight = 2
					last = one
					new = up
					while new > last:
						weight += .2
						last = new
						new = correlation(eval_set, 1, vecs, tag, args.p, weight, d1, d2)
					d1[tag] = weight
					print(tag, weight)
				else:
					down = correlation(eval_set, 1, vecs, tag, args.p, .5, d1, d2)
					if down > one:
						weight = .5
						last = one
						new = down
						while new > last:
							weight = weight-.2
							last = new
							new = correlation(eval_set, 1, vecs, tag, args.p, weight, d1, d2)
						d1[tag] = weight
						print(tag, weight)
					else:
						d1[tag] = 1
						print(tag, 1)
		x = 1 
		x2 = x+.2
		one = correlation(eval_set, 2, vecs, tag, args.p, x, d1, d2)
		up = correlation(eval_set, 2, vecs, tag, args.p, x2, d1, d2)
		if up > one:
			weight = x2
			last = one
			new = up
			while new > last:
				weight += .2
				last = new
				new = correlation(eval_set, 2, vecs, tag, args.p, weight, d1, d2)
			d2[tag] = weight
			print(tag, weight)

		else:
			down = correlation(eval_set, 2, vecs, tag, args.p, x-.2, d1, d2)
			if down > one:
				weight = x-.2
				last = one
				new = down
				while new > last:
					weight = weight-.2
					last = new
					new = correlation(eval_set, 2, vecs, tag, args.p, weight, d1, d2)
				d2[tag] = weight
				print(tag, weight)
			else:
				up = correlation(eval_set, 2, vecs, tag, args.p, 2, d1, d2)
				if up > one:
					weight = 2
					last = one
					new = up
					while new > last:
						weight += .2
						last = new
						new = correlation(eval_set, 2, vecs, tag, args.p, weight, d1, d2)
					d2[tag] = weight
					print(tag, weight)
				else:
					down = correlation(eval_set, 2, vecs, tag, args.p, .5, d1, d2)
					if down > one:
						weight = .5
						last = one
						new = down
						while new > last:
							weight = weight-.2
							last = new
							new = correlation(eval_set, 2, vecs, tag, args.p, weight, d1, d2)
						d2[tag] = weight
						print(tag, weight)
					else:
						d2[tag] = 1
						print(tag, 1)



def correlation(eval_set, comp, vecs, pos, typ, weight, d1, d2):
	sim = []
	act0 = []
	act1 = []
	act2 = []
	for i in range(eval_set.shape[0]):
		w1 = get_embedding(eval_set[i][0], comp, vecs, pos, typ, weight, d1, d2)
		w2 = get_embedding(eval_set[i][1], comp, vecs, pos, typ, weight, d1, d2)
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



	

def get_embedding(string, comp, vecs, pos, typ, weight, d1, d2):
	if pd.isnull(string):
		return []
	if comp == 1:
		return dilate(string, vecs, pos, typ, weight, d1, d2)
	elif comp == 2:
		return dilate2(string, vecs, pos, typ, weight, d1, d2)
	
def dilate(string, vecs, pos, typ, weight, d1, d2):
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	if typ == "dep":
		for token in word:
			u = vecs.query(token.head.text)
			v = vecs.query(token.text)
			uu = np.dot(u, u)
			uv = np.dot(u, v)
			if token.dep_ == pos:
				new.append((uv*u+(weight*uu*v)))
			elif token.dep_ in d2:
				new.append(d2[token.dep_]*(uv*u+(d1[token.dep_]*uu*v)))
			else:
				new.append((uv*u+uu*v))
	#try:
	return add(new)
	#except:
	#	return []
def dilate2(string, vecs, pos, typ, weight, d1, d2):
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	if typ == "dep":
		for token in word:
			u = vecs.query(token.head.text)
			v = vecs.query(token.text)
			uu = np.dot(u, u)
			uv = np.dot(u, v)
			if token.dep_ == pos:
				new.append(weight*(uv*u +(d1[token.dep_]*uu*v)))
			elif token.dep_ in d2:
				new.append(d2[token.dep_]*(uv*u+(d1[token.dep_]*uu*v)))
			else:
				new.append(uv*u+uu*v)
	#try:
	return add(new)
	#except:
	#	return []
def add(vecs):
        c = vecs[0]
        for i in range(len(vecs)-1):
                c = c+(vecs[i+1])
        return c

if __name__ == '__main__':
        main()


