from pymagnitude import *
import string
punc = string.punctuation
vecs = Magnitude('/home/paynesa/compositionality/original_words.magnitude')
paths = ['ppdb-sample.tsv', 'wiki-sample.tsv']
for p in paths:
	for line in open(p, 'r'):
		line = line.strip().split('\t')
		out = line[1].replace('[', '').replace(']', '')
		out = "{}_evaluations.txt".format(out.lower())
		print(out)	
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
		if k:
			with open(out, 'a') as f:
				f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))
                #        with open('total_evaluations.txt', 'a') as f:
                 #               f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))
                  #      with open(out, 'a') as f:
                   #             f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))
