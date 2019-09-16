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
	w = {
        "ROOT": 9.0,
        "det": -0.6,
        "advmod": 0.6000000000000001,
        "pobj": 0.4000000000000001,
        "prep": 0.20000000000000007,
        "amod": 0.20000000000000007,
        "nsubj": 1.4,
        "cc": 0.20000000000000007,
        "dobj": 1.9999999999999998,
        "intj": 1,
        "aux": 0.4000000000000001,
        "conj": -0.19999999999999996,
        "compound": 1.5999999999999999,
        "poss": 1,
        "mark": 1,
        "neg": 1,
        "prt": 0.4000000000000001,
        "xcomp": 2.2,
        "nummod": 2.4,
        "auxpass": 1.4,
        "pcomp": 1,
        "ccomp":1,
        "preconj": 2.6,
        "attr": 1,
        "npadvmod": 1,
        "acomp": 1,
        "dative": 0.20000000000000007,
        "punct": 1,
        "quantmod": 1,
        "dep": 1,
        "appos": 0.6000000000000001,
        "acl": 1,
        "predet": -0.8,
        "advcl": 1,
        "expl": 1,
        "nsubjpass": 1,
        "relcl": 1,
        "nmod": 1,
        "oprd" :1
        }

	t = {
        "NN": 3.600000000000001,
        "DT": -0.6,
        "RB": 0.6000000000000001,
        "IN": 0.6000000000000001,
        "JJ": 0.6000000000000001,
        "WRB": 1,
        "CD": -0.39999999999999997,
        "WDT": 1,
        "VB": 1.4,
        "PRP": 1,
        "UH": 1,
        "CC": 0.20000000000000007,
        "WP": 1,
        "VBP": 1,
        "VBN": 1,
        "TO": 0.4000000000000001,
        "PRP$": 1,
        "NNP": 0.4000000000000001,
        "VBD": 1.5999999999999999,
        "RP": 0.4000000000000001,
        "RBR": 1,
        "VBG": 0.4000000000000001,
        "NNS": 0.4000000000000001,
        "JJR": 1,
        "MD": 5.200000000000002,
        "JJS": 0.6000000000000001,
        "RBS": 0.4000000000000001,
        "WP$": 1,
        "FW": 2.6000000000000005,
        "XX": 1,
        "LS": 1,
        "PDT": -0.8,
        "EX": 1,
        "VBZ": 1,
        ".": 1,
        "AFX": 1
        }

	p = {
	"NUM":-.4,
	"NOUN": 3.8,
	"SPACE": 1,
	"DET": 1.4,
	"ADV": 1.8,
	"ADJ": 1.8,
	"ADP": 2.8,
	"VERB": 2.2,
	"PROPN": -2.8,
	"PRON": 1.6000000000000001,
	"INTJ": 1,
 	"CCONJ": .6,
	"PART": 1.4,
	"X":0.4,
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
		
		one = correlation(eval_set, args.d, vecs, tag, args.p, 1, w)
		up = correlation(eval_set, args.d, vecs, tag, args.p, 0, w)
		if one > up:
			print(tag, True)
		else:
			print(tag, False)
		
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
	weight = bool(weight)
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	if typ == "google":
		for token in word:
			if token.pos_ == pos:
				if weight:
					new.append(dictionary[token.pos_]*vecs.query(token.lemma_))
				else:
		#			print("hi")
					new.append(dictionary[token.pos_]*vecs.query(token.text))
			#elif token.pos_ in dictionary:
			else:
				new.append((dictionary[token.pos_])*vecs.query(token.text))
			#else:
			#	new.append(vecs.query(token.text))
	elif typ == "tag":
		for token in word:
			if token.tag_ == pos:
				if weight:
					new.append(dictionary[token.tag_]*vecs.query(token.lemma_))
				else:
					new.append(dictionary[token.tag_]*vecs.query(token.text))
			else:
			#elif token.pos_ in dictionary:
				new.append((dictionary[token.tag_])*vecs.query(token.text))
			#else:
			#	new.append(vecs.query(token.text))
	elif typ == "dep":
		for token in word:
			if token.dep_ == pos:
				if weight:
					new.append(dictionary[token.dep_]*vecs.query(token.lemma_))
				else:
					new.append(dictionary[token.dep_]*vecs.query(token.text))
			else:
				new.append((dictionary[token.dep_])*vecs.query(token.text))
			#else:
			#	new.append(vecs.query(token.text))
#	print(new)
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


