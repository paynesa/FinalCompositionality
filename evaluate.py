from pymagnitude import *
from argparse import ArgumentParser
import subprocess
import numpy as np
np.set_printoptions(suppress=True)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--v", default=None, type=str, help="The path to the vectors")
	parser.add_argument("comp", default=None, type=str, help="The type of composition")
	parser.add_argument("--w", default=None, type=str, help="second embeddings")
	args = parser.parse_args()
	return args

def main():
	correct = 0
	total = 0
	args = parse_args()
	v = Magnitude(args.v)
	if (args.w != None):
		w = Magnitude(args.w)
		vecs = Magnitude(v, w)
	else:
		vecs = v
	for line in open('2_phrase_w2v.txt', 'r'):
		line = line.strip().split()
		for word in line:
			if "_" in word:
				word = word.split("_")
				v = []
				for w in word:
					x = vecs.query(w.strip())
					v.append(x)
				
				if (args.comp == "multiply"):
					c = multiply(v)
				elif (args.comp == "add"):
					c = add(v)
				elif(args.comp == "decompose"):
					c = decompose(v)
				elif(args.comp == "average"):
					c = average(v)	
				elif(args.comp == "lapata"):
					c = lapata_combination(v)	
				else:
					raise Exception("Invalid compositionality method")
				with open('eval.txt', 'a') as f:
					f.write ("_".join(word)+" ")
					np.savetxt(f, c.reshape(1, c.shape[0]))
			else:
				c = vecs.query(word.strip())
				with open('eval.txt', 'a') as f:
                                        f.write (word.strip()+" ")
                                        np.savetxt(f, c.reshape(1, c.shape[0]))
	

	command = "python3.6 -m  pymagnitude.converter -i eval.txt -o eval.magnitude -a".split()
	try:	
		subprocess.check_output(command)
	except:
		raise Exception("There was a problem converting your vectors to magnitude")
	
	newvecs = Magnitude('eval.magnitude')
	
	for line in open('2_phrase_w2v.txt', 'r'):
		line = line.strip().split()
		if any(word not in newvecs for word in line):
			print("AHHHHHHHHHHHHHHHHHHHHHHHHH")
	#	print(newvecs.most_similar(positive=[line[0], line[3]], negative=line[1])[0][0], line[2].lower())
		if (newvecs.most_similar(positive=[line[0], line[3]], negative=line[1])[0][0].strip().lower() == line[2].lower().strip()):
			correct += 1
			total += 1
		else:
			total +=1
		#if (newvecs.most_similar(positive=[line[1], line[2]], negative=line[0])[0][0].lower().strip() == line[3].lower().strip()):
		#	correct += 1
		#	total += 1
		#else:
		#	total +=1
	percent = correct/total*100
	print("{}% analogies correct".format(percent))
				 
		

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
        else:
                c = add(x) / len(x)
                return c
def average(vecs):
	return add(vecs)/len(vecs)

def lapata_combination(vecs):
	m = multiply(vecs)
	a = add(vecs)
	c = .6*(m)+.4*(a)
	return(c)

if __name__ == '__main__':
	main() 
		
