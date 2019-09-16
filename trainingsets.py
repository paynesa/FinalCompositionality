from pymagnitude import *
import pandas as pd
oov_counter = 0
print("loading...")
img_dict = Magnitude('/data1/sarah/iter/image.magnitude')
word_dict = Magnitude('decomposition_words.magnitude')
print("done loading")
for line in open('/data1/sarah/iter/words.txt', 'r'):
	word = line.strip()
	if "row" in word:
		phrase = word.split('-')[1]
	else:
		phrase = word.split('-')[0]
	word_embedding = word_dict.query(phrase)
	img_embedding = img_dict.query(word)
	#print(phrase)
	if (False in pd.isnull(np.asarray(word_embedding))) and (False in pd.isnull(np.asarray(img_embedding))):
		if (phrase.strip() in word_dict) and (word.strip() in img_dict):
			with open('/data1/sarah/iter/compositionality/decomposition/words_processed.txt', 'a') as f:
				f.write("{}\n".format(phrase))
			with open('/data1/sarah/iter/compositionality/decomposition/x_train.txt', 'a') as f:
				np.savetxt(f, word_embedding.reshape(1, word_embedding.shape[0]))
			with open('/data1/sarah/iter/compositionality/decomposition/y_train.txt', 'a') as f:
				np.savetxt(f, img_embedding.reshape(1, img_embedding.shape[0]))
			print(phrase) 
		else:
			 oov_counter += 1
	
print("Done. {} OOV words found".format(oov_counter))
