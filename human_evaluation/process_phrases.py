from pymagnitude import *
import string
punc = string.punctuation 
from parseargs import ArgumentParser 
def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--w', type=str, help='path to word vectors')
	args = parser.parse_args()
	return args

def main():
	#load the word embeddings 
	try:
		vecs = Magnitude(args.w)
	except:
		raise Exception("Invalid path to word embeddings")
	print ("word embeddings loaded")
	paths = ['ppdb-sample.tsv', 'wiki-sample.tsv']
	#iterate through both evaluation files 
	for p in paths:
		out = "{}_evaluations.txt".format(p.split("-")[0])
		for line in open(p, 'r'):
			line = line.strip().split('\t')
			k = False
			flist = []
			for word in line[2:4]:
				if (" " in word.strip()):
					k = True
					final = ""
					word = word.strip()
					for l in word:
						if (l in punc) and (l != '\''):
							continue
						else:
							final +=l
					final = final.strip().replace("  ", " ").split()
					for f in final:
						if f.strip() not in vecs:
							k = False
					flist.append("_".join(final))
				else:
					flist.append(word.strip())		
			#write out the evaluations if they are valid phrases for which all words are in vocabulary 			
			if k:
				with open('total_evaluations.txt', 'a') as f:
					f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))
				with open(out, 'a') as f:
					f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))

	
if __name__ == '__main__':
	main()

