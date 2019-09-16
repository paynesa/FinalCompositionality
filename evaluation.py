import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pymagnitude import *
from scipy import stats
import spacy
from sklearn import preprocessing 
nlp = spacy.load('en_core_web_lg')
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--e", type=str, default="total", help="The type of evaluation set to be used")
	parser.add_argument("--v", type=str, default=None, help="Path to the original vectors")
	parser.add_argument("comp", type=str, help="Compositionality type")
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	print("Loading embeddings...")
	vecs = Magnitude(args.v)
	print("Embeddings loaded!")
	eval_set = pd.read_csv(args.e+'_evaluations.txt', sep=' ', header=None).as_matrix()
	sim = []
	avg = []
	act0 = []
	act1 = []
	act2 = []
	for i in range(eval_set.shape[0]):
		w1 = get_embedding(eval_set[i][0], args.comp, vecs)
		w2 = get_embedding(eval_set[i][1], args.comp, vecs)
		if (len(w1) == len(w2)) and (len(w1) != 0):
			#with open('lppdb_evaluations.txt', 'a') as f:
			#	f.write("{} {} {} {}\n".format(eval_set[i][0], eval_set[i][1], eval_set[i][2], eval_set[i][3]))
			this_sim = distance(w1, w2)
			sim.append(this_sim)
			avg.append(eval_set[i][2])
			actuals = eval_set[i][3].split(',')
			act0.append(float(actuals[0]))
			act1.append(float(actuals[1]))
			act2.append(float(actuals[2]))
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
		if comp == "spacy":
			return spacy(string, vecs)
		if comp == "tags":
			return tags(string, vecs)
		if comp == "dilate":
			return dilate(string, vecs)
		if comp == "head":
			return dilate2(string, vecs)
		x = []
		string = string.strip().split('_')
		#print(string)
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
			#elif (comp == "spacy"):
			#	return spacy(x, string)
			#else:
			#	raise Exception("Invalid composition type")
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

def decompose2(vecs):
	if len(vecs) ==2:
		u = vecs[0]
		uu = np.dot(u, u)
		v = vecs[1]
		uv = np.dot(v, u)
		return uu*v+uv*u
	else:
		x = vecs
		while len(x) >=2: 
			v = x[-2:]
			x = x[:len(x)-3]
			x.append(decompose2(v))
		if len(x) != 1:
			raise Exception("Oh crap")
		return x[0]



#average the vectors
def average(vecs):
        return add(vecs)/len(vecs)

#combine addition and multiplication
def lapata_combination(vecs):
        m = multiply(vecs)
        a = add(vecs)
        c = .6*(m)+.4*(a)
        return(c)
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
	#	return .0*vecs[0]+ .0*vecs[1]+ .0*vecs[2] +.29*vecs[3] + .3*vecs[4] + .3*vecs[5] 
		return .32*vecs[3]+.33*vecs[4]+ .33*vecs[5] 


def dilate2(s, vecs):
	lemma = {
	"NOUN":True,
	"DET":False,
	"ADV":False,
	"ADJ":False,
	"ADP":False,
	"VERB":True,
	"NUM":False,
	"PRON":True,
	"INTJ":False,
	"CCONJ":False,
	"PART":False,
	"PROPN":False,
	"X":False,
	"PUNCT":False,
	"NN":True,
	"DT":False,
	"RB":False,
	"IN":False,
	"JJ":False,
	"WRB":False,
	"CD":False,
	"WDT":False,
	"VB":False,
	"PRP":False,
	"UH":False,
	"CC":False,
	"WP":False,
	"VBP":False,
	"VBN":True,
	"TO":False,
	"PRP$":False,
	"NNP":False,
	"VBD":True,
	"RP":False,
	"RBR":False,
	"VBG":True,
	"NNS":True,
	"JJR":False,
	"MD":False,
	"JJS":True,
	"RBS":False,
	"WP$":False,
	"FW":False,
	"XX":False,
	"LS":False,
	"PDT":False,
	"EX":False,
	"VBZ":False,
	".":False,
	"AFX":False,
	"ROOT":True,
	"det":False,
	"advmod":False,
	"pobj":True,
	"prep":False,
	"amod":True,
	"nsubj":False,
	"cc":False,
	"dobj":False,
	"intj":False,
	"aux":False,
	"conj":False,
	"compound":False,
	"poss":False,
	"mark":False,
	"neg":False,
	"prt":False,
	"xcomp":False,
	"nummod":False,
	"auxpass":True,
	"pcomp":False,
	"ccomp":False,
	"preconj":False,
	"attr":False,
	"npadvmod":False,
	"acomp":False,
	"dative":False,
	"punct":False,
	"quantmod":False,
	"dep":False,
	"appos":True,
	"acl":False,
	"predet":False,
	"advcl":False,
	"expl":False,
	"nsubjpass":False,
	"relcl":False,
	"nmod":False,
	"oprd":False
	}
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
	"NOUN": 4.900000000000002,
	"DET": 0.0000000000000000001,
	"ADV": 0.8999999999999999,
	"ADJ": 0.8999999999999999,
	"ADP": 0.7999999999999999,
	"VERB": 1.4000000000000001,
	"NUM": 0.0000000000000001,
	"PRON": 0.9999999999999999,
	"INTJ": 4.900000000000002,
	"CCONJ": 0.4,
	"PART": 0.7,
	"PROPN": 0.000000000000000001,
	"X": 4.900000000000002,
	"PUNCT": 4.900000000000002
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


	m1 = {
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

	t1 = {
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

	t = {
	"NN": 1.7999999999999998,
	"DT": 5.551115123125783e-17,
	"RB": 0.6000000000000001,
	"IN": 0.6000000000000001,
	"JJ": 0.20000000000000007,
	"WRB": 1,
	"CD": -0.39999999999999997,
	"WDT": 1,
	"VB": 1.5999999999999999,
	"PRP": 0.6000000000000001,
	"UH": 1,
	"CC": -0.6,
	"WP": 1,
	"VBP": 2.4000000000000004,
	"VBN": 1,
	"TO": 0.20000000000000007,
	"PRP$": 2.4,
	"NNP": 0.6000000000000001,
	"VBD": 1,
	"RP": 2.6,
	"RBR": 4.200000000000001,
	"VBG": -0.19999999999999996,
	"NNS": 1,
	"JJR": 1,
	"MD": 5.400000000000002,
	"JJS": 0.20000000000000007,
	"RBS": 3.0000000000000004,
	"WP$": 1,
	"FW": 1.4,
	"XX": 1,
	"LS": 1,
	"PDT": -0.6,
	"EX": 1,
	"VBZ": 1,
	".": 1,
	"AFX": 1
	}

	m = {
	"NN": 3.800000000000001,
	"DT": -3.800000000000001,
	"RB": 4.200000000000001,
	"IN": -0.6,
	"JJ": 1.4,
	"WRB": 1,
	"CD": 2.1999999999999997,
	"WDT": 1,
	"VB": -0.39999999999999997,
	"PRP": 1.4,
	"UH": 1,
	"CC": 1.4,
	"WP": 1,
	"VBP": 0.4000000000000001,
	"VBN": 5.551115123125783e-17,
	"TO": 1.5999999999999999,
	"PRP$": 1,
	"NNP": 2.1999999999999997,
	"VBD": 1.5999999999999999,
	"RP": 1,
	"RBR": 0.6000000000000001,
	"VBG": 1.4,
	"NNS": 1.4,
	"JJR": 1,
	"MD": 1,
	"JJS": 2.2,
	"RBS": 0.6000000000000001,
	"WP$": 1,
	"FW": 3.400000000000001,
	"XX": 1,
	"LS": 1,
	"PDT": 3.400000000000001,
	"EX": 1,
	"VBZ": 1,
	".": 1,
	"AFX": 1
	}
	adjp = {
	"NOUN": 0.20000000000000007,
	"DET": -0.39999999999999997,
	"ADV": 0.20000000000000007,
	"ADJ": 0.6000000000000001,
	"ADP": 5.551115123125783e-17,
	"VERB": 9.999999999999996,
	"NUM": 1,
	"PRON": -0.19999999999999996,
	"INTJ": 4.200000000000001,
	"CCONJ": -0.8,
	"PART": 1.4,
	"PROPN": 1,
	"X": 1,
	"PUNCT": 1
	}
	word = " ".join(s.split('_'))
	word = nlp(word)
	new = []
	for token in word:
	#	u = vecs.query(token.head.text)
	#	if lemma[token.pos_]:
	#		v = vecs.query(token.lemma_)
	#	else:
	#		v = vecs.query(token.text)
	#	uu = np.dot(u, u)
	#	uv = np.dot(u, v)
	#	new.append(t[token.tag_]*(uv*u+m[token.tag_]*uu*v))
		
		if lemma[token.pos_]: 
			new.append((ppdbs2[token.pos_])*vecs.query(token.text))
		else:
			new.append((ppdbs2[token.pos_])*vecs.query(token.text))
	
	try:
		return add(new)
#	return preprocessing.normalize(add(new).reshape(-1, 1), norm = 'l2').reshape(1, -1)
	except:
		return []



def dilate1(s, vecs):
	word = " ".join(s.split('_'))
	word = nlp(word)
	new = []
	for token in word:
		u = vecs.query(token.head.text)
		v = vecs.query(token.lemma_)
		uu = np.dot(u, u)
		uv = np.dot(u, v)
		if token.dep_ == "det":
			new.append(1.8*(uv*u+0*uu*v))
		elif token.dep_ == "ROOT":
			new.append(.9*(uv*u+0*uu*v))
		elif token.dep_ == "pobj":
			new.append(.9*(uv*u+1.8*uu*v))
		elif token.dep_ == "prep":
			new.append(1*(uv*u+.7*uu*v))
		elif token.dep_ == "intj":
			new.append(1*(uv*u+1.4*uu*v))
		elif token.dep_ == "advmod":
			new.append(.9*(uv*u+.4*uu*v))
		elif token.dep_ == "amod":
			new.append(1.2*(uv*u + 1.7*uu*v))
		elif token.dep_ == "dobj":
			new.append(.9*(uv*u + 2.6*uu*v))	
		elif token.dep_ == "nsubj":
			new.append(1.4*(uv*u + .2*uu*v))
		elif token.dep_ == "aux":
			new.append(1.2*(uv*u + 1.8*uu*v))
		elif token.dep_ == "prt":
			new.append(.8*(uv*u +1.6*uu*v))
		elif token.dep_ == "cc":
			new.append(1*(uv*u+1.6*uu*v))
		elif token.dep_ == "conj":
			new.append(.3*(uv*u+1.2*uu*v))
		elif token.dep_ == "pos":
			new.append(0.0000001*(uv*u+2*uu*v))
		elif token.dep_ == "xcomp":
			new.append(.8*(uv*u + 1.4*uu*v))
		elif token.dep_ == "compound":
			new.append(1.1*(uv*u + 2.2*uu*v))
		elif token.dep_ == "mark":
			new.append(1*(uv*u + 2.6*uu*v))
		elif token.dep_ == "ccomp":
			new.append(1.1*(uv*u + 2.4*uu*v))
		elif token.dep_ == "auxpass":
			new.append(.9*(uv*u+1.2*uu*v))
		elif token.dep_ == "acl":
			new.append(.8*(uv*u+.4*uu*v))
		elif token.dep_ == "npadvmod":
			new.append(1.2*(uv*u+2.8*uu*v))
		elif token.dep_ == "neg":
			new.append(.9*(uv*u+14*uu*v))
		elif token.dep_ == "attr":
			new.append(.000001*(uv*u))
		elif token.dep_ == "acomp":
			new.append(1.1*(uv*u+.8*uu*v))
		elif token.dep_ == "dep":
			new.append(1.5*(uv*u+uu*v))
		elif token.dep_ == "nsubjpass":
			new.append(.05*(uv*u+.7*uu*v))
		elif token.dep_ == "pcomp":
			new.append(.9*(uv*u+1.1*uu*v))
		elif token.dep_ == "relcl":
			new.append(3*(uv*u+.1*uu*v))
		elif token.dep_ == "quantmod":
			new.append(20*(uv*u+.7*uu*v))
		elif token.dep_ == "nmod":
			new.append(4*(uv*u+.5*uu*v))
		elif token.dep_ == "appos":
			new.append(.2*(uv*u + 3*uu*v))
		elif token.dep_ == "preconj":
			new.append(.4*(uv*u + 1*uu*v))
		elif token.dep_ == "dative":
			new.append(10*(uv*u + 2*uu*v))
		elif token.dep_ == "advcl":
			new.append(2*(uv*u + 40*uu*v))
		else:
			new.append(uv*u+uu*v)
	try:	
		return add(new)
	except:
		[]



def dilate3(s, vecs):

        word = " ".join(s.split('_'))
        word = nlp(word)
        new = []
        
        for token in word:
                u = vecs.query(token.head.text)
                v = vecs.query(token.lemma_)                
                uu = np.dot(u, u)
                uv = np.dot(u,v)
                if token.pos_ == "DET":
                      new.append(.1*(uv*u+ .01*uu*v))
                elif token.pos_ == "PUNCT":
                      new.append(.1*(uv*u))
                elif token.pos_ == "ADP":
                      new.append(.1*(uu*v + 0*uv*u))
                elif token.pos_ == "PART":
                      new.append(3*(uu*v + .1*uv*u))
                elif token.pos_ == "NOUN":
                      new.append(1*(uu*v + .8*uv*u))
                elif token.pos_ == "ADJ":
                      new.append(.2*(uu*v + 3*uv*u))
                elif token.pos_ == "VERB":
                      new.append(.1*(uu*v + .9*uv*u))
                elif token.pos_ == "ADV":
                      new.append(.2*(uu*v + 2*uv*u))
                elif token.pos_ == "NUM":
                      new.append(.4*(uu*v + .1*uv*u))
                elif token.pos_ == "X":
                      new.append(.7*(uu*v + 10*uv*u))
                elif token.pos_ == "PROPN":
                      new.append(5*(uu*v))
                elif token.pos_ == "PRON":
                      new.append(.4*(uu*v + 5*uv*u))
                elif token.pos_ == "INTJ":
                      new.append(.05*(uu*v + .4*uv*u))
                elif token.pos_ == "CCONJ":
                      new.append(2*(uu*v + 1.5*uv*u))
                else:
                      print(token.pos_)
                      new.append(decompose([u, vecs.query(token.text)]))
        try:
                return add(new)
        except:
                return []



def dilate(s, vecs):
	word = " ".join(s.split('_'))
	word = nlp(word)
	new = []
	u = []
	for token in word:
		if token.dep_ == "ROOT":
			u = vecs.query(token.text)
	uu = np.dot(u, u)
	for token in word:
		if token.dep_ == "ROOT":
			continue
		else:
			v = vecs.query(token.lemma_)
			uv = np.dot(u,v)
			if token.pos_ == "DET":
				new.append(.00001*(uv*u))
			elif token.pos_ == "PUNCT":
				new.append(.00001*(uv*u))
			elif token.pos_ == "ADP":
				new.append(.5*(uu*v + 3*uv*u))
			elif token.pos_ == "PART":
				new.append(1.8*(uu*v + .7*uv*u))
			elif token.pos_ == "NOUN":
				new.append(2*(uu*v + uv*u))
			elif token.pos_ == "ADJ":
				new.append(.3*(uu*v + 6*uv*u))
			elif token.pos_ == "VERB":
				new.append(1*(uu*v + 1.8*uv*u))
			elif token.pos_ == "ADV":
				new.append(.15*(uu*v + 7*uv*u))
			elif token.pos_ == "NUM":
				new.append(.5*(uu*v + 6*uv*u))
			elif token.pos_ == "X":
				new.append(uu*v + 5*uv*u)
			elif token.pos_ == "PROPN":
				new.append(5*(uu*v ))
			elif token.pos_ == "PRON":
				new.append(.6*(uu*v + 4*uv*u))
			elif token.pos_ == "INTJ":
				new.append(.05*(uu*v + 2*uv*u))
			elif token.pos_ == "CCONJ":
				new.append(1.8*(uu*v + .7*uv*u))
			else:
				print(token.pos_)
				new.append(decompose([u, vecs.query(token.text)])) 
	try:
		return add(new)
	except:
		return []
def spacy(s, vecs):
	word = " ".join(s.split('_'))
	word = nlp(word)
	new = []
	for token in word:
		if str(token.pos_) == "DET":
			new.append(0.000001*vecs.query(token.text))
		elif str(token.pos_) == "PUNCT":
			new.append(0.0000001*vecs.query(token.text))
		elif str(token.pos_) == "ADP":
			new.append(0.76*vecs.query(token.text))
		elif str(token.pos_) == "PART":
			new.append(1.5*vecs.query(token.text))
		elif str(token.pos_) == "NOUN":
			new.append(2*vecs.query(token.lemma_))
		elif str(token.pos_) == "ADJ":
			new.append(1.3*vecs.query(token.text))
		elif str(token.pos_) == "VERB":
			new.append(1.4*vecs.query(token.text))
		elif str(token.pos_) == "ADV":
			new.append(.5*vecs.query(token.text))
		elif str(token.pos_) == "NUM":
			new.append(.6*vecs.query(token.text)) 
		elif str(token.pos_) == "X":
			new.append(.6*vecs.query(token.text))
		elif str(token.pos_) == "PROPN":
			new.append(50*vecs.query(token.text))
		elif str(token.pos_) == "PRON":
			new.append(.65*vecs.query(token.text))
		elif str(token.pos_) == "INTJ":
			new.append(1.75*vecs.query(token.text)) 
		elif str(token.pos_) == "CCONJ":
			new.append(.8*vecs.query(token.text))
		else:
			print(token.pos_)
			new.append(vecs.query(token.text))
	return(add(new))  

def tags(s, vecs):
	word = " ".join(s.split('_'))
	word = nlp(word)
	new = []
	for token in word:
		if str(token.tag_) == "NN":
			new.append(3.35*vecs.query(token.lemma_))
		elif str(token.tag_) == "NNS":
			new.append(3.7*vecs.query(token.lemma_))
		elif str(token.tag_) == "NNP":
			new.append(1000*vecs.query(token.text))
		elif str(token.tag_) == "PDT":
			#print(token.text)
			new.append(.000000000001*vecs.query(token.text))
		elif str(token.tag_) == "PRP":
			new.append(0.55*vecs.query(token.text))
		elif str(token.tag_) == "PRP$":
			new.append(.8*vecs.query(token.text))
		elif str(token.tag_) == "RB":
			new.append(1.2*vecs.query(token.lemma_))
		elif str(token.tag_) == "RBR":
			new.append(1.35*vecs.query(token.text))
		elif str(token.tag_) == "RBS":
			new.append(.9*vecs.query(token.text))
		elif str(token.tag_) == "RP":
			new.append(1.4*vecs.query(token.text))
		elif str(token.tag_) == "TO":
			new.append(2.4*vecs.query(token.text))
		elif str(token.tag_) == "UH":
			new.append(2.5*vecs.query(token.text))
		elif str(token.tag_) == "VB":
			new.append(1.3*vecs.query(token.lemma_))
		elif str(token.tag_) == "VBD":
			new.append(1.7*vecs.query(token.text))
		elif str(token.tag_) == "VBG":
			new.append(1.2*vecs.query(token.lemma_))
		elif str(token.tag_) == "VBN":
			new.append(.8*vecs.query(token.lemma_))
		elif str(token.tag_) == "VBP":
			new.append(3.2*vecs.query(token.lemma_))
		elif str(token.tag_) == "VBZ":
			new.append(1.2*vecs.query(token.lemma_))	
		elif str(token.tag_) == "WDT":
			new.append(.9*vecs.query(token.text))	
		elif str(token.tag_) == "WP":
			new.append(.7*vecs.query(token.text))
		elif str(token.tag_) == "WP$":
			new.append(4.2*vecs.query(token.text))
		elif str(token.tag_) == "WRB":
			new.append(.4*vecs.query(token.text)) 
		elif str(token.tag_) == "DT":
			new.append(.2*vecs.query(token.text))
		elif str(token.tag_) == "CC":
			new.append(.9*vecs.query(token.text))
		elif str(token.tag_) == "JJ":
			new.append(2.5*vecs.query(token.text))
		elif str(token.tag_) == "JJR": 
			new.append(.7*vecs.query(token.lemma_))
		elif str(token.tag_) == "JJS":
			new.append(5*vecs.query(token.text))
		elif str(token.tag_) == "IN":
			new.append(1.2*vecs.query(token.text))
		elif str(token.tag_) == "MD":
			new.append(11*vecs.query(token.text))
		elif str(token.tag_) == "CD":
			new.append(1.1*vecs.query(token.text))
		elif str(token.tag_) == "EX":
			new.append(2*vecs.query(token.text))
		elif str(token.tag_) == "LS":
			new.append(.2*vecs.query(token.text))
		elif str(token.tag_) == "FW":
			new.append(3.5*vecs.query(token.text))
		else:
			#	print(token.tag_)
			new.append(.7*vecs.query(token.text))
	return(add(new))
if __name__ == '__main__':
        main()


