import spacy, operator
nlp = spacy.load('en_core_web_lg')
pos = {}
tag = {}
dep = {}
for line in open('ltotal_evaluations.txt', 'r'):
	line = line.split()
	w1 = line[0].replace("_", " ")
	w2 = line[1].replace("_", " ")
	w1 = nlp(w1)
	w2 = nlp(w2)
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

for k, v in sorted(pos.items(), key=operator.itemgetter(1), reverse=True):
	with open('pos.txt', 'a') as f:
		f.write("{}\n".format(k))

for k, v in  sorted(tag.items(), key=operator.itemgetter(1), reverse=True):
	with open('tag.txt', 'a') as f:
		f.write("{}\n".format(k))

for k, v in sorted(dep.items(), key=operator.itemgetter(1), reverse=True):
	with open('dep.txt', 'a') as f:
		f.write("{}\n".format(k))


