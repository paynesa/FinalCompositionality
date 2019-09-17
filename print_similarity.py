from pymagnitude import *
from parseargs import ArgumentParser

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--w', type=str, help='path to word list')
	parser.add_argument('--g', type=str, help ='path to glove embeddings')
	parser.add_argument('--m', type=str, help = 'path to multimodal embeddings')
	parser.add_argument('--p', type=str, help = 'path to predicted embeddings')
	args = parser.parse_args()
	return args

def main():
	glove = Magnitude(args.g)
	multimodal = Magnitude(args.m)
	predictions = Magnitude(args.p)
	for line in open(args.w, 'r'):
		line = line.strip().split()
		if (len(line) == 1) and (line[0].strip() in multimodal):
			glove_sim = glove.most_similar_approx(line[0].strip(), topn = 2)
			mult_sim = multimodal.most_similar_approx(line[0].strip(), topn = 2)
			pred_sim = predictions.most_similar_approx(line[0].strip(), topn = 2)
			print(line[0]+": {}({})  {} ({}), {} ({})".format(glove_sim[0][0], glove_sim[1][0], mult_sim[0][0], mult_sim[1][0], pred_sim[0][0], pred_sim[1][0] ))

		else:
			line = "_".join(line)
			if (line.strip() in multimodal):
				glove_sim = glove.most_similar_approx(line.strip(), topn = 2)
				mult_sim = multimodal.most_similar_approx(line.strip(), topn = 2)
				pred_sim = predictions.most_similar_approx(line.strip(), topn = 2)
				print(line+": {} ({}),  {} ({}), {} ({})".format(glove_sim[0][0], glove_sim[1][0], mult_sim[0][0], mult_sim[1][0], pred_sim[0][0], pred_sim[1][0]))	
	
		
if __name__ == '__main__':
	main()
                                                   
