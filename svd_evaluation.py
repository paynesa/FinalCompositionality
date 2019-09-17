import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pymagnitude import *
from scipy import stats
import spacy
from sklearn.decomposition import TruncatedSVD  
nlp = spacy.load('en_core_web_lg')

#parse the command-line arguments 
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--e", type=str, default="total", help="The type of evaluation set to be used")
	parser.add_argument("--v", type=str, default=None, help="Path to the original vectors")
	parser.add_argument("comp", type=str, help="Compositionality type")
	args = parser.parse_args()
	return args

def main():
	#load the embeddings and evaluation set 
	args = parse_args()
	print("Loading embeddings...")
	vecs = Magnitude(args.v)
	print("Embeddings loaded!")
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	#get SIF weights 
	weights = {}
	for line in open('enwiki_vocab_min200.txt', 'r'):
		line = line.split()	
		try:
			weights[line[0].strip()] = (1e-3)/(1e-3+int(line[1].strip())) 
		except:
			print(line)
	avg = []
	act0 = []
	act1 = []
	act2 = []
	x = []
	y = []
	#iterate through through the evaluation set and add the cosine between two phrases to the list
	for i in range(eval_set.shape[0]):
		w1 = get_embedding(eval_set[i][0], args.comp, vecs, weights)
		w2 = get_embedding(eval_set[i][1], args.comp, vecs, weights)
		print(w1, w2)
		if (len(w1) == len(w2)) and (len(w1) != 0) and True not in np.isnan(w1) and True not in np.isnan(w2):
			x.append(w1)
			x.append(w2)
			avg.append(eval_set[i][2])
			actuals = eval_set[i][3].split(',')
			act0.append(float(actuals[0]))
			act1.append(float(actuals[1]))
			act2.append(float(actuals[2]))
	#SVD
	avg = np.asarray(avg)
	act0 = np.asarray(act0)
	act1 = np.asarray(act1)
	act2 = np.asarray(act2)
	nosvg = []
	svg = []
	x = np.asarray(x)
	x = np.nan_to_num(x)
	x = x/np.linalg.norm(x, axis=0, keepdims=True)
	x = x/np.linalg.norm(x, axis=1, keepdims=True)
	svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
	svd.fit(x)
	pc = svd.components_
	XX = x - x.dot(pc.transpose()).dot(pc)
	y = 0
	t = x.shape[0]
	while y< t:
		w1 = x[y, :]
		w2 = x[y+1, :]
		nosvg.append(distance(w1, w2))
		w1 = XX[y, :]
		w2 = XX[y+1, :]
		svg.append(distance(w1, w2))
		y+=2
	#find the correlation between the embedding similarities and the human judgement 
	cor0, pval0 = stats.spearmanr(nosvg, act0)
	cor1, pval1 = stats.spearmanr(nosvg, act1)
	cor2, pval2 = stats.spearmanr(nosvg, act2)
	coravg = (cor0+cor1+cor2)/3
	pvalavg = (pval0+pval1+pval2)/3
	print("Averaged across user correlation: {:.4f} \n\tpval {:.4f}".format(coravg, pvalavg))
	cor0, pval0 = stats.spearmanr(svg, act0)
	cor1, pval1 = stats.spearmanr(svg, act1)
	cor2, pval2 = stats.spearmanr(svg, act2)
	coravg = (cor0+cor1+cor2)/3
	pvalavg = (pval0+pval1+pval2)/3
	print("Averaged across user correlation: {:.4f} \n\tpval {:.4f}".format(coravg, pvalavg))


#calculate the cosine distance between two embeddings
def distance(word1, word2):
	dot_product = np.dot(word1, word2)
	mag_word1 = np.linalg.norm(word1)
	mag_word2 = np.linalg.norm(word2)
	return (dot_product/(mag_word1 * mag_word2))		

#given a phrase, method of compositionality, and embeddings, return the processed embedding
def get_embedding(string, comp, vecs, weights):
	if pd.isnull(string):
		return []
	else:
		if comp == "d":
			return d(string, vecs, weights)
		if comp == "new":
			return new(string, vecs, weights)
		if comp == "spacy":
			return spacy(string, vecs)
		if comp == "tags":
			return tags(string, vecs)
		if comp == "dilate":
			return decompose(string, vecs)
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
				average(x)
			elif (comp == "lapata_combination"):
				lapata_combination(x)
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
        #return 1/(np.linalg.norm(c))*c
        #return c/len(vecs)
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
                c = add(x) 
                return c

#weight based on pos. Add more dictionaries as needed 
def d(string, vecs, sif):
	deps = {
        "ROOT": 3.900000000000002,
        "det": 0.0000000000001,
        "advmod": 0.7999999999999999,
        "pobj": 0.7,
        "prep": 0.4,
        "amod": 0.5,
        "nsubj": 1.2,
        "cc": 0.4,
        "dobj": 1.9000000000000006,
        "intj": 3.900000000000002,
        "aux": 0.5,
        "conj": 0.1,
        "compound": 1.3,
        "poss": 1.0999999999999999,
        "mark": 3.900000000000002,
        "neg": 0.8999999999999999,
        "prt": 0.7,
        "xcomp": 2.3000000000000007,
        "nummod": 2.2000000000000006,
        "auxpass": 1.2,
        "pcomp": 3.900000000000002,
        "ccomp": 3.900000000000002,
        "preconj": 3.900000000000002,
        "attr": 3.900000000000002,
        "npadvmod": 3.900000000000002,
        "acomp": 3.900000000000002,
        "dative": 0.4,
        "punct": 3.900000000000002,
        "quantmod": 3.900000000000002,
        "dep": 3.900000000000002,
        "appos": 0.0000000000001,
        "acl": 3.900000000000002,
        "predet": 0.000000000000001,
        "advcl": 3.900000000000002,
        "expl": 3.900000000000002,
        "nsubjpass": 3.900000000000002,
        "relcl": 3.900000000000002,
        "nmod": 3.900000000000002,
        "oprd": 3.900000000000002
        }


	poss = {
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
	mpos = {
	"NOUN": 1,
	"DET": 1.7999999999999998,
	"ADV": 2.6,
	"ADJ": -0.19999999999999996,
	"ADP": 0.3,
	"VERB": 4.000000000000001,
	"NUM": 1,
	"PRON": 0.6000000000000001,
	"INTJ": 1,	
	"CCONJ": 1,
	"PART": 1.5999999999999999,
	"PROPN": 1,
	"X": 0.6000000000000001,
	"PUNCT": 1
	}
	tpos = {
	"NOUN": 7.300000000000005,
	"DET": -0.6000000000000001,
	"ADV": 2.9000000000000004,
	"ADJ": 1.2999999999999998,
	"ADP": 1,
	"VERB": 0.6000000000000003,
	"NUM": 1.5999999999999999,
	"PRON": 1.4,
	"INTJ": 1,
	"CCONJ": 1,
	"PART": 1.0999999999999999,
	"PROPN": -4.600000000000001,
	"X": 4.900000000000003,
	"PUNCT": 1,
	}
	mpos2 = {
	"NOUN": 4.800000000000002,
	"DET": -2.1999999999999997,
	"ADV": 7.0000000000000036,
	"ADJ": 1.4,
	"ADP": 3.800000000000001,
	"VERB": 3.800000000000001,
	"NUM": 2.4,
	"PRON": 0.6000000000000001,
	"INTJ": 1,
	"CCONJ": -0.10000000000000003,
	"PART": 5.551115123125783e-17,
	"PROPN": -9.599999999999998,
	"X": 1.5999999999999999,
	"PUNCT": 1
	}

	tpos2 = {
	"NOUN": 1.9999999999999998,
	"DET": 2.8000000000000003,
	"ADV": 1.5999999999999999,
	"ADJ": 5.551115123125783e-17,
	"ADP": 1,
	"VERB": 1.5999999999999999,
	"NUM": 1,
	"PRON": 1.4,
	"INTJ": 1,
	"CCONJ": 0.6000000000000001,
	"PART": 1,
	"PROPN": 1,
	"X": 0.4000000000000001,
	"PUNCT": 1
	}
	mtag = {
	"NN": 3.800000000000001,
	"DT": -3.800000000000001,
	"RB": 5.000000000000002,
	"IN": 2.4,
	"JJ": 1.4,
	"WRB": 1,
	"CD": 2.1999999999999997,
	"WDT": 1,
	"VB": 4.400000000000001,
	"PRP": 1.5999999999999999,
	"UH": 1,
	"CC": 1,
	"WP": 1,
	"VBP": 1,
	"VBN": -0.19999999999999996,
	"TO": 1.4,
	"PRP$": 3.400000000000001,
	"NNP": -7.0000000000000036,
	"VBD": 3.600000000000001,
	"RP": 0.6000000000000001,
	"RBR": 1,
	"VBG": 1.4,
	"NNS": 3.800000000000001,
	"JJR": 1,
	"MD": -1.8999999999999997,
	"JJS": 0.6000000000000001,
	"RBS": 0.20000000000000007,
	"WP$": 1,
	"FW": 1.4,
	"XX": 1,
	"LS": 1,
	"PDT": -3.0000000000000004,
	"EX": 1,
	"VBZ": 1,
	".": 1,
	"AFX": 1
	}
	ttag = {
	"NN": 1.7999999999999998,
	"DT": 1,
	"RB": 1.5999999999999999,
	"IN": 0.6000000000000001,
	"JJ": 5.551115123125783e-17,
	"WRB": 1,
	"CD": 1,
	"WDT": 1,
	"VB": 2.2,
	"PRP": 0.6000000000000001,
	"UH": 1,
	"CC": -0.10000000000000003,
	"WP": 1,
	"VBP": 2.2,
	"VBN": 1.4,
	"TO": 0.6000000000000001,
	"PRP$": 7.800000000000004,
	"NNP": 1,
	"VBD": 1.7999999999999998,
	"RP": 1.4,
	"RBR": 1,
	"VBG": 1,
	"NNS": 0.6000000000000001,
	"JJR": 1,
	"MD": 1,
	"JJS": 0.6000000000000001,
	"RBS": 1.4,
	"WP$": 1,
	"FW": 1.7999999999999998,
	"XX": 1,	
	"LS": 1,
	"PDT": 1.4,
	"EX": 1,
	"VBZ": 1,
	".": 1,
	"AFX": 1
	}
	mdep = {
	"ROOT": 10.799999999999994 ,
	"det": -14.999999999999979,
	"advmod": 1,
	"pobj": 2.4,
	"prep": 0.4000000000000001,
	"amod": 1.5999999999999999,
	"nsubj": 2.2,
	"cc": 1,
	"dobj": 1,
	"intj": 1,
	"aux": 1,
	"conj": 1.4,
	"compound": 7.600000000000004,
	"poss": 1.4,
	"mark": 1,
	"neg": 0.20000000000000007,
	"prt": 1.4,
	"xcomp": 1.4,
	"nummod": 1.9999999999999998,
	"auxpass": 2.6,
	"pcomp": 1,
	"ccomp": 1,
	"preconj": 1,
	"attr": 1,
	"npadvmod": 1,
	"acomp": 1,
	"dative": 1.4,
	"punct": 1,
	"quantmod": 1,
	"dep": 1,
	"appos": 0.6000000000000001,
	"acl": 1,
	"predet": -9.599999999999998,
	"advcl": 1,
	"expl": 1,
	"nsubjpass": 1,
	"relcl": 1,
	"nmod": 1,
	"oprd": 1
	}
	tdep = {
	"ROOT": 3.2000000000000006,
	"det": 1,
	"advmod": 1,
	"pobj": 3.800000000000001,
	"prep": 1.5999999999999999,
	"amod": 1.7999999999999998,
	"nsubj": 1.7999999999999998,
	"cc": 1,
	"dobj": 1,
	"intj": 1,
	"aux": 2.2,
	"conj": 1.4,
	"compound": 1.4,
	"poss": -0.39999999999999997,
	"mark": 1,
	"neg": 0.6000000000000001,
	"prt": 2.2,
	"xcomp": 2.2,
	"nummod": 1,
	"auxpass": 4.400000000000001,
	"pcomp": 1,
	"ccomp": 1,
	"preconj": 1,
	"attr": 1,
	"npadvmod": 1,
	"acomp": 1,
	"dative": 1,
	"punct": 1,
	"quantmod": 1,
	"dep": 1,
	"appos": 1.5999999999999999,
	"acl": 1,
	"predet": 3.400000000000001,
	"advcl": 1,
	"expl": 1,
	"nsubjpass": 1,
	"relcl": 1,
	"nmod": 1,
	"oprd": 1
	}
	
	string = string.replace("_", " ")
	string = nlp(string)
	new = []
	for w in string [0:]:
		v = vecs.query(w.text)
		u = vecs.query(w.head.text)
		uu = np.dot(u, u)
		uv = np.dot(v,u)
		p = uu*v +(mdep[w.dep_]+3*mpos[w.pos_])/4*uv*u
		try:
			new.append(((tdep[w.dep_]+3*tpos[w.pos_])/4+0*100000000*sif[w.text])*p)
		except:
			new.append(((tdep[w.dep_]+3*tpos[w.pos_])/4*p))
		print(new)
	try:
		return add(new)
	except:
		return []	
		


#average the vectors
def average(vecs):
        return add(vecs)/len(vecs)

#combine addition and multiplication
def lapata_combination(vecs):
        m = multiply(vecs)
        a = add(vecs)
        c = .6*(m)+.4*(a)
        return(c)

#weight the words by position in the sentence 
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

#add weights to use this method 
def new(s, vecs, weights):
	word = s.replace("_", " ").split()
	new = []
	for w in word:
		try:	
			new.append(weights[w.strip()]*vecs.query(w.strip()))
		except:
			new.append(1e-3/(1e-3 + 1)*vecs.query(w.strip()))
	try:
		return np.asarray(add(new), dtype=float)
	except:
		return []


if __name__ == '__main__':
        main()


