import spacy
phrase = {}
first = {}
second = {}
third = {}
fourth = {}
fifth = {}
sixth = {}
nlp = spacy.load('en_core_web_lg')
for line in open('../ntotal_evaluations.txt', 'r'):
	line = line.split()
	print(line)
	if "_" in line[0] or "_" in line[1]:
		x = nlp(line[0].replace("_", " "))
		y = nlp(line[1].replace("_", " "))
		for token in x:
			print(token.text)
			if token.pos_ not in first:
				first[token.pos_] = 1
			else:
				first[token.pos_] +=1
		for token in y:
			if token.pos_ not in first:
				first[token.pos_] = 1
			else:
				first[token.pos_] +=1
		#if len(x) == 6:
		#	if x[0].pos_ in first:
		#		first[x[0].pos_] += 1
		#	else:
		#		first[x[0].pos_] = 1
		#	if x[1].pos_ in second:
		#		second[x[1].pos_] +=1
		#	else:
		#		second[x[1].pos_] = 1
		#	if x[2].pos_ in third:
		#		third[x[2].pos_] += 1
		#	else:
		#		third[x[2].pos_] = 1
		#	if x[3].pos_ in fourth:
		#		fourth[x[3].pos_] += 1
		#	else:
		#		fourth[x[3].pos_] = 1
		#	if x[4].pos_ in fifth:
		#		fifth[x[4].pos_] += 1
		#	else:
		#		fifth[x[4].pos_] = 1
		#	if x[5].pos_ in sixth:
		#		sixth[x[5].pos_] += 1
		#	else:
		#		sixth[x[5].pos_] = 1
		#	if line[1] not in phrase:
		#		phrase[line[1]] = 1
		#	else:
		#		phrase[line[1]] += 1
		#	print(first, second)	
	#	if len(y) == 6:
	#		if y[0].pos_ in first:
	#			first[y[0].pos_] += 1
	#		else:
	#			first[y[0].pos_] = 1
	#		if y[1].pos_ in second:
	#			second[y[1].pos_] +=1
	#		else:
	#			second[y[1].pos_] = 1
	#		if y[2].pos_ in third:
	#			third[y[2].pos_] +=1
	#		else:
	#			third[y[2].pos_] = 1
	#		if y[3].pos_ in fourth:
	#			fourth[y[3].pos_] +=1
	#		else:
	#			fourth[y[3].pos_] = 1
	#		if y[4].pos_ in fifth:
	#			fifth[y[4].pos_] +=1
	#		else:
	#			fifth[y[4].pos_] = 1
	#		if y[5].pos_ in sixth:
	#			sixth[y[5].pos_] +=1
	#		else:
	#			sixth[y[5].pos_] = 1
	#		if line[1] in phrase:
	#			if len(x) !=2:
	#				phrase[line[1]] += 1
	#		else:
	#			phrase[line[1]] = 1
	#		print(first, second, phrase)    
		
		#for token in x:
		#	print(token.text, token.pos_)
		#	print(x[0].pos_)
    			#print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)
		#print(line)
total_first = sum(first.values())
#total_second = sum(second.values())
#total_third = sum(third.values())
#total_fourth = sum(fourth.values())
#total_fifth = sum(fifth.values())
#total_sixth = sum(sixth.values())
#phrase_sum = sum(phrase.values())
print("FIRST:")
for k, v in first.items():
	print ("\t{}, {:.3f}".format(k, v/total_first*100))
print("SECOND:")
for k, v in second.items():
	print ("\t{}, {:.3f}".format(k, v/total_second*100))
print("THIRD:")
for k, v in third.items():
	print("\t{}, {:.3f}".format(k, v/total_third*100))
print("FOURTH:")
for k, v in fourth.items():
        print("\t{}, {:.3f}".format(k, v/total_fourth*100))
print("FIFTH:")
for k, v in fifth.items():
	print("\t{}, {:.3f}".format(k, v/total_fifth*100))
print("SIXTH:")
for k, v in sixth.items():
	print("\t{}, {:.3f}".format(k, v/total_sixth*100)) 
print("PHRASES:")
for k, v in phrase.items():
	print("\t{}, {:.3f}".format(k, v/phrase_sum*100)) 

	
