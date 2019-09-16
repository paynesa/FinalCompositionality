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
	parser.add_argument("--d", type=str, help="Compositionality type")
	parser.add_argument("--p", type=str, help= "type of tag")
	parser.add_argument("--t", type=str, help= "tag list")
	args = parser.parse_args()
	return args

#compute the distance between two embeddings using cosine similarity
def distance(word1, word2):
	dot_product = np.dot(word1, word2)
	mag_word1 = np.linalg.norm(word1)
	mag_word2 = np.linalg.norm(word2)
	return (dot_product/(mag_word1 * mag_word2))

def main():
	#load the embeddings
	print("Loading embeddings")
	args = parse_args()
	try:
		vecs = Magnitude(args.v)
	except:
		raise Exception("Invalid path to embeddings")
	print("Embeddings loaded!")
	#load the evaluation set as a csv
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	tags = [] 
	d1 = {}
	d2 = {}
	#iterate through the tags file and create a list of the tags	
	for line in open(args.t, 'r'):
		################### get initial (inside) weights ####################
		tag = line.strip()
		one = correlation(eval_set, 1, vecs, tag, args.p, 1, d1, d2)
		up = correlation(eval_set, 1, vecs, tag, args.p, 1.2, d1, d2)
		#if going up improves the weight, keep going up until it doesn't
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
			#otherwise, try to go down
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
			#if that doesn't work, try taking more drastic up and down measurements
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
					#finally, if all else fails, make the weight 1
					else:
						d1[tag] = 1
						print(tag, 1)

		####################### get secondary (outside) weights ##########################
		x = 1 
		x2 = x+.2
		one = correlation(eval_set, 2, vecs, tag, args.p, x, d1, d2)
		up = correlation(eval_set, 2, vecs, tag, args.p, x2, d1, d2)
		#if increasing the weight improves correlation, keep increasing until it no longer does
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

		#otherwise, try decreasing the weight
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

			#if neither of these work, try more drastic increments
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

					#finally, set the weight to 1 
					else:
						d2[tag] = 1
						print(tag, 1)


#compute the correlation between the embeddings and human evaluations
def correlation(eval_set, comp, vecs, pos, typ, weight, d1, d2):
	sim = []
	act0 = []
	act1 = []
	act2 = []
	#iterate through the evaluation set, get the embeddings and their distances, and append to list 
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
	#compute and return correlation between the cosine list and human judgement 
	sim = np.asarray(sim)
	act0 = np.asarray(act0)
	act1 = np.asarray(act1)
	act2 = np.asarray(act2)
	cor0, pval0 = stats.spearmanr(sim, act0)
	cor1, pval1 = stats.spearmanr(sim, act1)
	cor2, pval2 = stats.spearmanr(sim, act2)
	coravg = (cor0+cor1+cor2)/3
	return coravg	


#the function to get the weighted embeddings for each phrase. Either gets inner-weighting or outer-weighting based on comp argument 
def get_embedding(string, comp, vecs, pos, typ, weight, d1, d2):
	if pd.isnull(string):
		return []
	if comp == 1:
		return dilate(string, vecs, pos, typ, weight, d1, d2)
	elif comp == 2:
		return dilate2(string, vecs, pos, typ, weight, d1, d2)

#the inner weight computation 	
def dilate(string, vecs, pos, typ, weight, d1, d2):
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	#get pos type, and then weight accordingly after querying the necessary vectors
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
	elif typ == "pos":
		for token in word:
			u = vecs.query(token.head.text)
			v = vecs.query(token.text)
			uu = np.dot(u, u)
			uv = np.dot(u, v)
			if token.pos_ == pos:
				new.append((uv*u+(weight*uu*v)))
			elif token.pos_ in d2:
				new.append(d2[token.pos_]*(uv*u+(d1[token.pos_]*uu*v)))
			else:
				new.append((uv*u+uu*v))
	elif typ == "tag":
		for token in word:
			u = vecs.query(token.head.text)
			v = vecs.query(token.text)
			uu = np.dot(u, u)
			uv = np.dot(u, v)
			if token.tag_ == pos:
				new.append((uv*u+(weight*uu*v)))
			elif token.tag_ in d2:
				new.append(d2[token.tag_]*(uv*u+(d1[token.tag_]*uu*v)))
			else:
				new.append((uv*u+uu*v))		
	try:
		return add(new)
	except:
		return []

#the outer weight computation
def dilate2(string, vecs, pos, typ, weight, d1, d2):
	word = " ".join(string.split('_'))
	word = nlp(word)
	new = []
	#get pos type, and then weight accordingly after querying the necessary vectors
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
	if typ == "pos":
		for token in word:
			u = vecs.query(token.head.text)
			v = vecs.query(token.text)
			uu = np.dot(u, u)
			uv = np.dot(u, v)
			if token.pos_ == pos:
				new.append(weight*(uv*u +(d1[token.pos_]*uu*v)))
			elif token.pos_ in d2:
				new.append(d2[token.pos_]*(uv*u+(d1[token.pos_]*uu*v)))
			else:
				new.append(uv*u+uu*v)	
	if typ == "tag":
		for token in word:
			u = vecs.query(token.head.text)
			v = vecs.query(token.text)
			uu = np.dot(u, u)
			uv = np.dot(u, v)
			if token.tag_ == pos:
				new.append(weight*(uv*u +(d1[token.tag_]*uu*v)))
			elif token.tag_ in d2:
				new.append(d2[token.tag_]*(uv*u+(d1[token.tag_]*uu*v)))
			else:
				new.append(uv*u+uu*v)
	try:
		return add(new)
	except:
		return []

#add a set of vectors together and return the sum 
def add(vecs):
        c = vecs[0]
        for i in range(len(vecs)-1):
                c = c+(vecs[i+1])
        return c

if __name__ == '__main__':
        main()


