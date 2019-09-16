from pymagnitude import *
from argparse import ArgumentParser
import string
punc = string.punctuation

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--v', default = None, type=str, help = "the path to the word embeddings")
	parser.add_argument('--n', default = None, type=int, help = "the length of phrases you would like to have written to your evaluation sets")
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	#attempt to open the word vectors that the user input
	try:
		vecs = Magnitude(args.v)
	except:
		raise Exception("The path to your vectors isn't valid")
	#iterate through both of the evaluation files 
	paths = ['ppdb-sample.tsv', 'wiki-sample.tsv']
	total = "total_{}.txt".format(str(args.n))
	for p in paths:
		out = "{}_{}.txt".format(p.split("-")[0], str(args.n))
		#read the file line by line
		for line in open(p, 'r'):
			line = line.strip.split('\t')
			k = False
	                n = False
	                flist = []
			#check the phrases 
			for word in line[2:4]:
				#see if phrase or single word. If phrase, remove punctuation
				if (" " in word.strip()):
					k = True
					final = ""
					word = word.strip()
					for l in word:
						if (l in punc) and (l != '\''):
							continue
						else:
							final += l
					#check that the phrase is of the desired length
					final = final.strip().replace("  ", " ").split()
					if len(final) == args.n:
						n = True
						#make sure the phrase is in vocabulary
						for f in final:
							if f.strip() not in vecs:
								k = False
						flist.append("_".join(final))
				#if the phrase is only one word, keep it and don't modify the truth value of k
				elif len(word.strip().split()) == 1:
					flist.append(word.strip())
				else:
					k = False
			#write out the phrase pairs if at least one of them is of the desired length
			if k and n:
				with open(total, 'a') as f:
					f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))
				with open(out, 'a') as f:
					f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))

if __name__ == '__main__':
	main()

