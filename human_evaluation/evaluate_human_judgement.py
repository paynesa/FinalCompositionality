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
	parser.add_argument("comp", type=str, help="Compositionality type")
	args = parser.parse_args()
	return args

def main():
	#load the embeddings from magnitude
	args = parse_args()
	print("Loading embeddings...")
	try:
		vecs = Magnitude(args.v) 
	except:
		raise Exception("Invalid path to embeddings")
	print("Embeddings loaded!")
	#load the evaluation set
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	sim = []
	avg = []
	act0 = []
	act1 = []
	act2 = []
	#iterate through the evaluation set and get the distance between each pair of composed embeddings 
	for i in range(eval_set.shape[0]):
		w1 = get_embedding(eval_set[i][0], args.comp, vecs)
		w2 = get_embedding(eval_set[i][1], args.comp, vecs)
		if (len(w1) == len(w2)) and (len(w1) != 0):
			this_sim = distance(w1, w2)
			sim.append(this_sim)
			avg.append(eval_set[i][2])
			actuals = eval_set[i][3].split(',')
			act0.append(float(actuals[0]))
			act1.append(float(actuals[1]))
			act2.append(float(actuals[2]))
	#print the correlations between the composed embeddings and human judgement, as well as the leave-one-out resampling for inter-human agreement
	sim = np.asarray(sim)
	avg = np.asarray(avg)
	act0 = np.asarray(act0)
	act1 = np.asarray(act1)
	act2 = np.asarray(act2)
	cor, pval = stats.spearmanr(sim, avg)
	print("{} pairs evaluated".format(len(sim)))
	print("Average correlation: {:.4f} \n\tpval: {:.4f}".format(cor, pval))
	cor0, pval0 = stats.spearmanr(sim, act0)
	print("User 0 correlation: {:.4f} \n\tpval: {:.4f}".format(cor0, pval0))
	cor1, pval1 = stats.spearmanr(sim, act1)
	print("User 1 correlation: {:.4f} \n\tpval: {:.4f}".format(cor1, pval1))
	cor2, pval2 = stats.spearmanr(sim, act2)
	print("User 2 correlation: {:.4f} \n\tpval: {:.4f}".format(cor2, pval2))
	coravg = (cor0+cor1+cor2)/3
	pvalavg = (pval0+pval1+pval2)/3
	print("Averaged across user correlation: {:.4f} \n\tpval {:.4f}".format(coravg, pvalavg))
	avg12 = (act1+act2)/2
	avg02 = (act0+act2)/2
	avg01 = (act0+act1)/2
	c0, p0 = stats.spearmanr(act0, avg12)
	c1, p1 = stats.spearmanr(act1, avg02)
	c2, p2 = stats.spearmanr(act2, avg01)
	print("Leave one out resampling: {:.3f}".format((c0+c1+c2)/3))

#calculate the cosine distance between two embeddings
def distance(word1, word2):
	dot_product = np.dot(word1, word2)
	mag_word1 = np.linalg.norm(word1)
	mag_word2 = np.linalg.norm(word2)
	return (dot_product/(mag_word1 * mag_word2))		

#given a phrase, method of compositionality, and embeddings, return the processed embedding
def get_embedding(string, comp, vecs):
	if pd.isnull(string):
		return []
	else:
		if comp == "head":
			return dilate2(string, vecs)
		x = []
		string = string.strip().split('_')
		if (len(string) == 1) and (string[0].strip() in vecs):
			return vecs.query(string[0])
		elif all(word.strip() in vecs for word in string):
			for word in string:
				x.append(vecs.query(word.strip()))
			if (comp == "multiply"):
				return multiply(x)
			elif (comp == "add"):
				return add(x)
			elif (comp == "lapata"):
				return lapata_combination(x)
			elif (comp == "decompose"):
				return decompose(x)
			elif (comp == "weight"):
				return weighted_add(x)
			elif (comp == "average"):
				return average(x)
			elif (comp == "lapata_combination"):
				return lapata_combination(x)
			else:
				raise Exception("Invalid composition type")
		else:
			return []

#multiply all vectors together, regardless of order
def multiply(vecs):
        c = vecs[0]
        for i in range(len(vecs)-1):
                c = c*(vecs[i+1])
        return c

#add all the vectors together, regardless of order
def add(vecs):
        c = vecs[0]
        for i in range(len(vecs)-1):
                c = c+(vecs[i+1])
        return c

#vector decomposition by Lapata et. al.
def decompose(vecs):
        u = vecs[0]
        uu = np.dot(u, u)
        x = []
        for v in vecs[1:]:
                uv = np.dot(v, u)
                p = uu*v + uv*u
                x.append(p)
        if (len(x) == 1):
                return x[0]
        elif (len(x) ==0):
                return []
        else:
                c = add(x) / len(x)
                return c

#average the vectors
def average(vecs):
        return add(vecs)/len(vecs)

#combine addition and multiplication as per Lapata et. al.
def lapata_combination(vecs):
        m = multiply(vecs)
        a = add(vecs)
        c = .6*(m)+.4*(a)
        return(c)

#add the vectors based on previously calculated weights 
def weighted_add(vecs):
	if len(vecs)== 2:
		return 0.3*vecs[0]+0.7*vecs[1]
	elif len(vecs) == 3:
		return .33*vecs[0]+.27*vecs[1]+0.38*vecs[2]
	elif len(vecs) == 4:
		return .03*vecs[0]+.21*vecs[1]+.38*vecs[2]+ .38*vecs[3]
	elif len(vecs) == 5:
		return .1*vecs[0]+.1*vecs[1]+.3*vecs[2]+.2*vecs[3]+.3*vecs[4] 
	else:
		return .32*vecs[3]+.33*vecs[4]+ .33*vecs[5] 

#weight words in the embedding based on syntax. Some starting weights are included, but you can add your own as you calculate them 
def dilate2(s, vecs):
	wiki  = {
		"ADJ": 1.4,
		"NOUN":3.8, 
		"NUM": 0.0000000001,
		"SPACE": 1,
		"ADV": 1.8, 
		"VERB": 3, 
		"PROPN": 11.2, 
		"ADP": 1.4, 
		"INTJ": 1, 
		"PUNCT": 1,
		"CCONJ": 0.4, 
		"DET": .6,
		"PRON": 1, 
		"SYM": 1, 
		"X": -9.4, 
		"PART": .8
	}
	ppdb = {
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
	ppdb1 = {
	"NOUN": 3.8000000000000025,
	"DET": 1.3000000000000003,
	"ADV": 2.200000000000001,
	"ADJ": 1.7000000000000006,
	"ADP": 2.6000000000000014,
	"VERB": 1.7000000000000006,
	"NUM": -0.19999999999999987,
	"PRON": 1.2000000000000002,
	"INTJ": 1,
	"CCONJ": 0.5000000000000001,
	"PART": 1,
	"PROPN": -2.600000000000001,
	"X": 1,
	"PUNCT": 1
	}

	ppdb12 = {
	"NOUN": 3.8000000000000025,
	"DET": 1.3000000000000003,
	"ADV": 2.200000000000001,
	"ADJ": 1.7000000000000006,
	"ADP": 2.6000000000000014,
	"VERB": 1.7000000000000006,
	"NUM": -0.19999999999999987,
	"PRON": 1.2000000000000002,
	"INTJ": 1,
	"CCONJ": 0.30000000000000016,
	"PART": 1,
	"PROPN": -2.600000000000001,
	"X": 1,
	"PUNCT": 1
	}
 
	ppdbs ={
	"NOUN": 3.900000000000002,
	"DET": 0.0000000000001,
	"ADV": 0.8999999999999999,
	"ADJ": 0.8999999999999999,
	"ADP": 0.7999999999999999,
	"VERB": 1.4000000000000001,
	"NUM": 0.0000000000001,
	"PRON": 0.00000000000001,
	"INTJ": 0.00000000000000001,
	"CCONJ": 0.4,
	"PART": 0.7,
	"PROPN": 0.00000000000001,
	"X": 3.3000000000000016,
	"PUNCT" : 0.00000000000001,
	}

	ppdbs2 = {
	"NOUN": 4.999999999999998,
	"DET": 0.0000000000001,
	"ADV": 0.8999999999999999,
	"ADJ": 0.8999999999999999,
	"ADP": 0.7999999999999999,
	"VERB": 1.4000000000000001,
	"NUM": 0.0000000000001,
	"PRON": 0.0000000000001,
	"INTJ": 0.0000000000001,
	"CCONJ": 0.4,
	"PART": 0.7,
	"PROPN": 0.000000000001,
	"X": 4.899999999999999,
	"PUNCT": 0.000000000001
	}

	tag = {
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

	dep = {
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

	word = " ".join(s.split('_'))
	word = nlp(word)
	new = []
	for token in word:
		new.append((ppdbs2[token.pos_])*vecs.query(token.text))
	try:
		return add(new)
	except:
		return []

if __name__ == '__main__':
        main()


