from pymagnitude import *
from argparse import ArgumentParser
import subprocess
import numpy as np
np.set_printoptions(suppress=True)

#parse the command-line arguments
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--v", default=None, type=str, help="The path to the vectors")
	parser.add_argument("comp", default=None, type=str, help="The type of composition (multiply, decompose lapata, lapata_combination, or add)")
	parser.add_argument("--a", default=None, type=str, help="path to analogies")
	args = parser.parse_args()
	return args

def main():
	correct = 0
	total = 0
	args = parse_args()
	try:
		v = Magnitude(args.v)
	except:
		raise Exception("Invalid embeddings path")	
	#iterate through the analogies
	for line in open(args.a, 'r'):
		line = line.strip().split()
		for word in line:
			#process phrases
			if "_" in word:
				word = word.split("_")
				v = []
				for w in word:
					x = vecs.query(w.strip())
					v.append(x)
				#compose the vectors for the phrase based on the composition function given in the arguments 
				if (args.comp == "multiply"):
					c = multiply(v)
				elif (args.comp == "add"):
					c = add(v)
				elif(args.comp == "decompose"):
					c = decompose(v)	
				elif(args.comp == "lapata"):
					c = lapata(v)	
				elif(args.comp == "lapata_combination"):
					c = lapata_combination(v)		
				else:
					raise Exception("Invalid compositionality method")
				#write out the embeddings to a text file 
				with open('eval.txt', 'a') as f:
					f.write ("_".join(word)+" ")
					np.savetxt(f, c.reshape(1, c.shape[0]))
			#process single words			
			else:				
				c = vecs.query(word.strip())
				with open('eval.txt', 'a') as f:
                                        f.write (word.strip()+" ")
                                        np.savetxt(f, c.reshape(1, c.shape[0]))
	#convert the composed embeddings to magnitude format
	command = "python3.6 -m  pymagnitude.converter -i eval.txt -o eval.magnitude -a".split()
	try:	
		subprocess.check_output(command)
	except:
		raise Exception("There was a problem converting your vectors to magnitude")

	#load the composed embeddings
	newvecs = Magnitude('eval.magnitude')
	#iterate through the analogies again to get the most similar phrases and evaluate their correctness for each one 
	for line in open(args.a, 'r'):
		line = line.strip().split()
		if any(word not in newvecs for word in line):
			continue
		if (newvecs.most_similar(positive=[line[0], line[3]], negative=line[1])[0][0].strip().lower() == line[2].lower().strip()):
			correct += 1
			total += 1
		else:
			total +=1
	#return the percent that were correct
	percent = correct/total*100
	print("{}% analogies correct".format(percent))
				 
		
#multiply all of the vectors together, regardless of order
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

#vector decomposition (by Lapata et. al.)
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

#the combination of multiplication and addition presented by Lapata et. al. 
def lapata_combination(vecs):
	m = multiply(vecs)
	a = add(vecs)
	c = .6*(m)+.4*(a)
	return(c)

if __name__ == '__main__':
	main() 
		
