import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pymagnitude import *
from scipy import stats
import spacy
from sklearn.decomposition import TruncatedSVD  
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
	weights = {}
	for line in open('enwiki_vocab_min200.txt', 'r'):
		line = line.split()	
		try:
			weights[line[0].strip()] = (1e-3)/(1e-3+int(line[1].strip())) 
		except:
			print(line)
	#sim = []
	avg = []
	act0 = []
	act1 = []
	act2 = []
	x = []
	y = []
	for i in range(eval_set.shape[0]):
		w1 = get_embedding(eval_set[i][0], args.comp, vecs, weights)
		w2 = get_embedding(eval_set[i][1], args.comp, vecs, weights)
		print(w1, w2)
		if (len(w1) == len(w2)) and (len(w1) != 0) and True not in np.isnan(w1) and True not in np.isnan(w2):
			x.append(w1)
			x.append(w2)
		#	y.append(distance(w1, w2))				
			#with open('lppdb_evaluations.txt', 'a') as f:
			#	f.write("{} {} {} {}\n".format(eval_set[i][0], eval_set[i][1], eval_set[i][2], eval_set[i][3]))
			#this_sim = distance(w1, w2)
			#sim.append(this_sim)
			avg.append(eval_set[i][2])
			actuals = eval_set[i][3].split(',')
			act0.append(float(actuals[0]))
			act1.append(float(actuals[1]))
			act2.append(float(actuals[2]))
#	x0, y0 = stats.spearmanr(y, act0)
#	x1, y1 = stats.spearmanr(y, act1)
#	x2, y2 = stats.spearmanr(y, act2)
#	print((x0+x1+x2)/3)
	avg = np.asarray(avg)
	act0 = np.asarray(act0)
	act1 = np.asarray(act1)
	act2 = np.asarray(act2)
	nosvg = []
	svg = []
	x = np.asarray(x)
	x = np.nan_to_num(x)
#	print(np.any(np.isnan(x)), np.all(np.isfinite(x)))
#	print(x)
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
#		print(w1, w2)
		nosvg.append(distance(w1, w2))
		w1 = XX[y, :]
		w2 = XX[y+1, :]
		svg.append(distance(w1, w2))
		y+=2
#	print(nosvg)
	print(len(svg), len(act0))
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
#	cor, pval = stats.spearmanr(sim, avg)
#	print("{} pairs evaluated".format(len(sim)))
#	print("Average correlation: {:.4f} \n\tpval: {:.4f}".format(cor, pval))
##	cor0, pval0 = stats.spearmanr(sim, act0)
##	print("User 0 correlation: {:.4f} \n\tpval: {:.4f}".format(cor0, pval0))
#	cor1, pval1 = stats.spearmanr(sim, act1)
#	print("User 1 correlation: {:.4f} \n\tpval: {:.4f}".format(cor1, pval1))
#	cor2, pval2 = stats.spearmanr(sim, act2)
#	print("User 2 correlation: {:.4f} \n\tpval: {:.4f}".format(cor2, pval2))
#	coravg = (cor0+cor1+cor2)/3
#	pvalavg = (pval0+pval1+pval2)/3
#	print("Averaged across user correlation: {:.4f} \n\tpval {:.4f}".format(coravg, pvalavg))
#	avg12 = (act1+act2)/2
#	avg02 = (act0+act2)/2
#	avg01 = (act0+act1)/2
#	c0, p0 = stats.spearmanr(act0, avg12)
#	c1, p1 = stats.spearmanr(act1, avg02)
#	c2, p2 = stats.spearmanr(act2, avg01)
#	svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
#	x = np.asarray(x)
#	svd.fit(x)
#	pc = svd.components_
#	XX = x - x.dot(pc.transpose()) * pc
#	y = 0 
#	t =  x.shape[0]
#	s = []
#	while y < t:
#		w1 = x[y, :]
#		w2 = x[y+1, :]
#		s.append(distance(w1, w2))
#		y += 2
#	print(stats.spearmanr(s, act0))	
#	print(XX, XX.shape, svd.components_.shape)
#	print("Leave one out resampling: {:.3f}".format((c0+c1+c2)/3))

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
		if comp == "head":
			return dilate2(string, vecs, weights)
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
#	u = None
#	for token in string:
#		if token.dep_ == "ROOT":
#			u = vecs.query(token.text)
#	u = vecs.query(string[0].text)	
#	uu = np.dot(u, u)
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

def new(s, vecs, weights):
	word = s.replace("_", " ").split()
	new = []
	for w in word:
		try:	
			#print(weights[w.strip()]*vecs.query(w.strip()))
			new.append(weights[w.strip()]*vecs.query(w.strip()))
		except:
			#print(word)
			new.append(1e-3/(1e-3 + 1)*vecs.query(w.strip()))
	try:
		return np.asarray(add(new), dtype=float)
	except:
		return []

def dilate2(s, vecs, weights):
	pos={
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

	
	tags = {
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

	tag = {
	"NN": 3.600000000000001,
	"DT": -1.7999999999999998,
	"RB": 3.800000000000001,
	"IN": 1.7999999999999998,
	"JJ": 0.4000000000000001,
	"WRB": 1,
	"CD": 1.5999999999999999,
	"WDT": 1,
	"VB": 2.1999999999999997,
	"PRP": 0.6000000000000001,
	"UH": 1,
	"CC": 0.6000000000000001,
	"WP": 1,
	"VBP": 1,
	"VBN": 1.4,
	"TO": 1.7999999999999998,
	"PRP$": 7.600000000000004,
	"NNP": -3.2000000000000006,
	"VBD": 2.8000000000000003,
	"RP": 1.4,
	"RBR": 1.4,
	"VBG": 1.4,
	"NNS": 2.4,
	"JJR": 1,
	"MD": -0.8999999999999999,
	"JJS": 1.5999999999999999,
	"RBS": -1.4,
	"WP$": 1,
	"FW": 0.6000000000000001,
	"XX": 1,
	"LS": 1,
	"PDT": -1.0,
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
	"xcomp": 2.4000000000000004,
	"nummod": 2.4,
	"auxpass": 1.4,
	"pcomp": 1,
	"ccomp": 1,
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
	"oprd": 1
	}

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
		try:
#			new.append(((poss[token.pos_])+100000000*weights[token.text])*vecs.query(token.text))
#			new.append((dep[token.dep_]+100000000*weights[token.text])*vecs.query(token.text))
#			new.append((poss[token.pos_])*vecs.query(token.text))
#			new.append(((tags[token.tag_]+tag[token.tag_])/2+100000000*weights[token.text])*vecs.query(token.text))
			new.append(((deps[token.dep_]+pos[token.pos_]+poss[token.pos_])/3+100000000*weights[token.text])*vecs.query(token.text))
		except:
			continue
	
	try:
		return add(new)
	except:
		return []


if __name__ == '__main__':
        main()


