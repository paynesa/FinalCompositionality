import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pymagnitude import *
from scipy import stats
import spacy
nlp = spacy.load('en_core_web_lg')

#parse the command-line arguments
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--e", type=str, default="total", help="The type of evaluation set to be used")
	parser.add_argument("--v", type=str, default=None, help="Path to the original vectors")
	parser.add_argument("--p", type=str, help= "type of tag")
	parser.add_argument("--t", type=str, help= "tag list")
	args = parser.parse_args()
	return args

#compute the cosine distance between two word embeddings
def distance(word1, word2):
	dot_product = np.dot(word1, word2)
	mag_word1 = np.linalg.norm(word1)
	mag_word2 = np.linalg.norm(word2)
	return (dot_product/(mag_word1 * mag_word2))

def main():
	d = {}
	print("Loading embeddings")
	args = parse_args()
	try:
		vecs = Magnitude(args.v)
	except:
		raise Exception("Invalid path to embeddings")
	print("Embeddings loaded!")
	#iterate through the tag file to make a list of tags
	tags = [] 
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	for line in open(args.t, 'r'):
		tag = line.strip()
		#compare the correlation of the original and going slightly up. If up is better, then keep going up until it's not
		one = correlation(eval_set, args.d, vecs, tag, args.p, 1, d)
		up = correlation(eval_set, args.d, vecs, tag, args.p, 1.2, d)
		if up > one:
			weight = 1.2
			last = one
			new = up
			while new >= last:
				weight += .2
				last = new
				new = correlation(eval_set, args.d, vecs, tag, args.p, weight, d)
			d[tag] = weight
			print(tag, weight)
		
		#otherwise, keep going down until it gets better 
		else:
			down = correlation(eval_set, args.d, vecs, tag, args.p, .8, d)
			if down > one:
				weight = .8
				last = one
				new = down
				while new >= last:
					weight = weight-.2
					last = new
					new = correlation(eval_set, args.d, vecs, tag, args.p, weight, d)
				d[tag] = weight
				print(tag, weight)
			#if this doesn't work, try more drastic differences 
			else:
				up = correlation(eval_set, args.d, vecs, tag, args.p, 2, d)
				if up > one:
					weight = 2
					last = one
					new = up
					while new >= last:
						weight += .2
						last = new
						new = correlation(eval_set, args.d, vecs, tag, args.p, weight, d)
					d[tag] = weight
					print(tag, weight)
				else:
					down = correlation(eval_set, args.d, vecs, tag, args.p, .5, d)
					if down > one:
						weight = .5
						last = one
						new = down
						while new >= last:
							weight = weight-.2
							last = new
							new = correlation(eval_set, args.d, vecs, tag, args.p, weight, d)
						d[tag] = weight
						print(tag, weight)
					#if that doesn't work, then leave the weight at 1					
					else:
						d[tag] = 1
						print(tag, 1)

#compute the correlation of the weighted embeddings with human judgement 
def correlation(eval_set, comp, vecs, pos, typ, weight, dict):
	sim = []
	act0 = []
	act1 = []
	act2 = []
	#iterate through the evaluation set and add the cosine distance to a list
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
	#compute the correlation between the cosine distance and human judgement 	
	sim = np.asarray(sim)
	act0 = np.asarray(act0)
	act1 = np.asarray(act1)
	act2 = np.asarray(act2)
	cor0, pval0 = stats.spearmanr(sim, act0)
	cor1, pval1 = stats.spearmanr(sim, act1)
	cor2, pval2 = stats.spearmanr(sim, act2)
	coravg = (cor0+cor1+cor2)/3
	return coravg	



	
#get the embedding for the phrase based on the weighting by pos
def get_embedding(string, comp, vecs, pos, typ, weight, dictionary):
	if pd.isnull(string):
		return []
	else:
		return normal(string, vecs, pos, typ, weight, dictionary)

#weight the words by their pos
def normal(string, vecs, pos, typ, weight, dictionary):
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	#check which type of tagging is being used
	if typ == "google":
		for token in word:
			if token.pos_ == pos:
				new.append(weight*vecs.query(token.text))
			elif token.tag_ in dictionary:
				new.append((dictionary[token.tag_])*vecs.query(token.text))
			else:
				new.append(vecs.query(token.text))
	elif typ == "tag":
		for token in word:
			if token.tag_ == pos:
				new.append(weight*vecs.query(token.text))
			elif token.tag_ in dictionary:
				new.append((dictionary[token.tag_])*vecs.query(token.text))
			else:
				new.append(vecs.query(token.text))
	elif typ == "dep":
		for token in word:
			if token.dep_ == pos:
				new.append(weight*vecs.query(token.text))
			elif token.tag_ in dictionary:
				new.append((dictionary[token.tag_])*vecs.query(token.text))
			else:
				new.append(vecs.query(token.text))
	try:
		return add(new)
	except:
		return []		
		
	
#add all vectors together, regardless of order
def add(vecs):
        c = vecs[0]
        for i in range(len(vecs)-1):
                c = c+(vecs[i+1])
        return c

if __name__ == '__main__':
        main()


