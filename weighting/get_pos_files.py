import spacy, operator
from argeparse import ArgumentParser
nlp = spacy.load('en_core_web_lg')

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--i", default=None, type=str, help="The path to the evaluation set")
	args = parser.parse_args()
	return args

def main():
	pos = {}
	tag = {}
	dep = {}
	for line in open(args.i, 'r'):
		line = line.split()
		w1 = line[0].replace("_", " ")
		w2 = line[1].replace("_", " ")
		w1 = nlp(w1)
		w2 = nlp(w2)

		#add tags to the dictionary for each tag in the first word
		for token in w1:
			if token.pos_ not in pos:
				pos[token.pos_] = 1
			else:
				pos[token.pos_] +=1
			if token.tag_ not in tag:
				tag[token.tag_] = 1
			else:
				tag[token.tag_] += 1
			if token.dep_ not in dep:
				dep[token.dep_] = 1
			else:
				dep[token.dep_] += 1

		#add tags to the dictionary for each tag in the second word
		for token in w2:
			if token.pos_ not in pos:
				pos[token.pos_] = 1
			else:
				pos[token.pos_] += 1
			if token.tag_ not in tag:
				tag[token.tag_] = 1
			else:
				tag[token.tag_] += 1
			if token.dep_ not in dep:
				dep[token.dep_] = 1
			else:
				dep[token.dep_] += 1

	#write the dictionaries out to their respective files 
	for k, v in sorted(pos.items(), key=operator.itemgetter(1), reverse=True):
		with open('pos.txt', 'a') as f:
			f.write("{}\n".format(k))

	for k, v in  sorted(tag.items(), key=operator.itemgetter(1), reverse=True):
		with open('tag.txt', 'a') as f:
			f.write("{}\n".format(k))

	for k, v in sorted(dep.items(), key=operator.itemgetter(1), reverse=True):
		with open('dep.txt', 'a') as f:
			f.write("{}\n".format(k))

if __name__ = '__main__':
	main()


