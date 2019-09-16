from pymagnitude import *
vecs = Magnitude('/data1/minh/magnitude/word.magnitude')
print("loaded")
for line in open('questions-phrases.txt', 'r'):
	if line.strip()[0] == ':':
		print(line)	
	else:
		line = line.strip().split()
		k = True
		for word in line:
			if "_" in word:
				word = word.strip().split("_")
				#for sub in word:
				#	if sub.strip() not in vecs:
				#		k = False
				if len(word) != 2:
					k = False
			else:
				if word.strip() not in vecs:
					k = False
		if k:
			with open('2_phrase_w2v.txt', 'a') as f:
				f.write("{}\n".format(" ".join(line)))
		#else:
		#	with open('zs_w2v.txt', 'a') as f:
		#		f.write("{}\n".format(" ".join(line)))

