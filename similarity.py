from pymagnitude import *
glove = Magnitude("/home/paynesa/compositionality/decomposition_words.magnitude")
multimodal  = Magnitude("/home/paynesa/compositionality/decomposition_multimodal.magnitude")
predictions = Magnitude("/home/paynesa/compositionality/decomposition_predictions.magnitude")
#concatenated = Magnitude(glove, multimodal)
for line in open('/data1/sarah/avg/OOV/words_processed.txt', 'r'):
	line = line.strip().split()
	if (len(line) == 1) and (line[0].strip() in multimodal):
		glove_sim = glove.most_similar_approx(line[0].strip(), topn = 2)
		mult_sim = multimodal.most_similar_approx(line[0].strip(), topn = 2)
		pred_sim = predictions.most_similar_approx(line[0].strip(), topn = 2)
		#conc_sim = concatenated.most_similar(line[0].strip(), topn = 2)
		print(line[0]+": {}({})  {} ({}), {} ({})".format(glove_sim[0][0], glove_sim[1][0], mult_sim[0][0], mult_sim[1][0], pred_sim[0][0], pred_sim[1][0] ))

	else:
		line = "_".join(line)
		if (line.strip() in multimodal):
			glove_sim = glove.most_similar_approx(line.strip(), topn = 2)
			mult_sim = multimodal.most_similar_approx(line.strip(), topn = 2)
			pred_sim = predictions.most_similar_approx(line.strip(), topn = 2)
			#conc_sim = concatenated.most_similar(line.strip(), topn = 2)
			print(line+": {} ({}),  {} ({}), {} ({})".format(glove_sim[0][0], glove_sim[1][0], mult_sim[0][0], mult_sim[1][0], pred_sim[0][0], pred_sim[1][0]))
		

                                                   
