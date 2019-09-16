#parse the vectors (multimodal or regular)
from pymagnitude import *
from argparse import ArgumentParser
import string
punc = string.punctuation
import numpy as np
np.set_printoptions(suppress=True)
l = ["the", "a", "an", "to"]

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--e", default=None, type=str, help = "the path to the embeddings")
	parser.add_argument("--w", default=None, type=str, help = "the path to the word list")
	parser.add_argument("--o", default=None, type=str, help = "the path where you would like the embeddings to be written")
	parser.add_argument("comp", default=None, type=str, help = "the type of composition: multiplication, addition, decomposition")
	parser.add_argument("--a", default=False, type=bool, help = "should articles be included? (Default = they will not be)")
	args = parser.parse_args()
	return args
def main():
	args = parse_args()
	if (args.e == None) or (args.w == None) or (args.o == None) or (args.comp == None):
		raise Exception("You must input the path to the embeddings (--e), the path to the word list (--w), the path where you would like the files to be written (--o), and the type of composition (multiplication, addition, or decomposition)")
	try:
		vectors = Magnitude(args.e)
	except:
		raise Exception("Invalid file to your word embeddings")
	try:
		for line in open(args.w, 'r'):
			line = line.strip()
			#handle single words first, check to make sure that they aren't just empty
			if (line in vectors) and (line != ""):
				x = vectors.query(line)
				if False in (np.isnan(x)):
					with open(args.o, 'a') as f:
						f.write(line + " ")
						print(line)
						np.savetxt(f, x.reshape(1, x.shape[0]))
			#handle phrases that don't include punctuation
			if (" " in line) and all(char not in punc for char in line):
				line = line.split()
				if all(word in vectors for word in line):
					#remove articles
					if (not args.a) and (line[0].strip in l):
						line = line[1:]
					#make sure that there is more than one word in the phrase and that its representation is valid
					if (len(line) != 1):
						x = []
						for word in line:
							vec = vectors.query(word.strip())
							if (word.strip() != "") and (False in np.isnan(vec)):
								x.append(vec)
						#compose the vectors
						if (args.comp == "multiplication"):
							c = multiply(x)
						elif (args.comp == "addition"):
							c = add(x)
						elif (args.comp == "decomposition"):
							c = decompose(x)
						else:
							raise Exception("You must input a valid compositionality type (multiplication, addition, decomposition)")
						#write the vectors
						try:						
							with open(args. o, 'a') as f:
								f.write("_".join(line)+" ")
								print(" ".join(line))
								np.savetxt(f, c.reshape(1, c.shape[0]))
						except: 
							raise Exception("There was a problem writing your embeddings")
	except:
		raise Exception("Couldn't open your word list. Please check the filepath")

#multiply all the vectors together, regardless of order							
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
if __name__ == '__main__':
	main()
	
