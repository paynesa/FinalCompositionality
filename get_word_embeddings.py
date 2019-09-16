from pymagnitude import *
import string
import numpy as np
np.set_printoptions(suppress=True)
punc = string.punctuation
from argparse import ArgumentParser

#parse the command-line arguments 
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--w", default=None, type=str, help = "The path to the word embeddings")
	parser.add_argument("--l", default=None, type=str, help = "The list of processed words")
	parser.add_argument("--o", default=None, type=str, help = "The output filepath") 
	args = parser.parse_args()
	return args

def main():
	#load the vectors from magnitude 
	args = parse_args()
	try:
		vectors = Magnitude(args.w)
	except:
		raise exception("Invalid file path")
	#iterate through the processed words
	for line in open(args.l):
		line = line.strip()
		#if the word is in vectors, write it to the output file
		if (line in vectors):
			c = vectors.query(line)
			with open(args.o, 'a') as f:
				f.write(line+ " ")
				np.savetxt(f, c.reshape(1, c.shape[0]))
		#otherwise, check if it is a phrase and all of the constituents are in vectors. If they are, write them to the output file
		elif (" " in line) and all(char not in punc for char in line):
			line = line.split()
			if all(word in vectors for word in line):
				for word in line:
					c = vectors.query(word.strip())
					with open(args.o, 'a') as f:
						f.write(word + " ")
						np.savetxt(f, c.reshape(1, c.shape[0]))
	
if __name__ == '__main__':
	main()
