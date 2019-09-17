# Vector-Based Semantic Compositionality
This code offers a collection of methods for creating and evaluating composed phrase embeddings. These embeddings are composed in a number of ways, including methods of addition and multiplication originally proposed by [Mitchell and Lapata (2010](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01106.x). We additionally give methods for incorporating image information using [multimodal embeddings](https://github.com/paynesa/multimodal), and for weighting words in a phrase based on their syntactic value according to the [Spacy model](https://spacy.io/usage/linguistic-features). The code is divided into sections as follows: 

# Embeddings 
This folder contains the code (but not the data) necessary to obtain and manipulate the embeddings for the rest of the work. The code is divided up as follows:
### co-occurrence_embeddings.py
Mitchell and Lapata conduct all of their experiments on 2000D co-occurrence embeddings derived from a lemmatized version of the [British National Corpus](http://www.natcorp.ox.ac.uk/). To run this code, you will need to obtain a copy of the corpus and clean and lemmatize it. Then run the code with the parameters --i and --o: the directory containing the lemmatized BNC and the file where you would like your embeddings to be saved, respectively. The embeddings may then be converted to the [Magnitude](https://github.com/plasticityai/magnitude) format on the command line. 
### get_word_embeddings.py 
Iterates through a list of words and phrases (joined by "\_") and obtains the word embeddings for the words in the phrases, then writes them to an output file. This gives the embeddings needed to compose embeddings in the rest of the code. This file takes in three arguments: --l is the path to the list of words and phrases, --w is the path to the word embeddings, which should be in [Magnitude](https://github.com/plasticityai/magnitude) format, and --o is the file where you would like your embeddings to be written. This file can be converted to the Magnitude format in the command line after creation. 
### predict_mmembeddings.py 
Uses a [ultimodal embedding model]https://github.com/paynesa/multimodal) to predict the multimodal embeddings for the word embeddings created by _get_word_embeddings.py_. It takes in three command-line arguments: --m is the path to the model you'd like to use to generate your multimodal embeddings, --p is the path to the prediction set (in this case, the same as the file created by _get_word_embeddings.py_. This file should not be in Magnitude format) and --o is the path where you would like your embeddings to be written. This output can once again be converted to Magnitude in the command line. 
### print_similarity.py 
Iterates through unimodal, multimodal, and composed embeddings to find the most similar for a list of words and phrases, and prints the most similar words and phrases to the console.This allows you to subjectively evaluate the embeddings that you have been working with with regards to their performance on similarity tasks. This code takes in four command-line arguments: --w gives the path to the list of words, --g gives the path to the unimodal embeddings, --m gives the path to the multimodal embeddings, and --p gives the path to the predicted/composed embeddings. 
### write_composed.py 
Writes the composed embeddings to an output file so that you may work with them outside of this environment, or may input them to _print_similarity.py_. This code takes in the following command-line arguments: --e gives the path to the word embeddings to be composed, --w gives the word/phrase list of things to be composed, --o gives the path to the output file, --a gives whether or not you would like articles to be included, and comp is the method of compositionality you would like to utilize. Multiplication, addition, and decomposition are currently available for this model. The output may be converted to the Magnitude format in the command line. 

# Analogies
We evaluate the composed embeddings on analogies taken from [Mikolov et al. (2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). You will need to obtain these analogies yourself from the link provided. 
### evaluate_analogies.py 
Evaluates your composition method based on its performance on the analogies. This code takes in the following command-line arguments: --v gives the path to the word embeddings, --a gives the path to the Mikolov et al. analogies, and comp gives the composition type (addition, multiplication, decomposition, or the combination proposed by Mitchell and Lapata).

# Human_Evaluation
This directory uses the human paraphrase judgements collected by [Pavlick et al. (2015)] (https://cs.brown.edu/people/epavlick/papers/ppdb2.pdf) to give correlation with cosines between composed phrase embeddings. You will need to download this dataset from the link provided. 
### process_phrases.py
Processes the Pavlick et al. data to contain only pairs in which one constituent is a phrase and all words are in vocabulary. This file simply takes in the command-line argument --w, which gives the path to the word embeddings. This code should be run from the directory containing the Pavlick data, namely _ppdb-sample.tsv_ and _wiki-sample.tsv_. 
### process_by_phrase_length.py
Processes the Pavlick et al. data to contain only pairs in which one constituent is a phrase of a given length and all words are in the vocabulary. This is helpful for determining if a given composition method is length-dependent. This code takes in the command-line arguments --v, which gives the path to the word vectors, and --n, which gives the desired phrase length.
### evaluate_human_judgement.py
Evaluates a given composition method against human judgements and returns the correlation per user, the averaged correlation, and the inter-user agreement. This code takes in the following command-line arguments: --e gives the evaluation set to be used, --v gives the path to the word embeddings, and comp gives the method of compositionality to be used (our weighting scheme, multiply, add, Mitchell and Lapata's combination, decomposition, weighted addition, or average). Correlations are printed to the console. 
### svd_evaluation.py
Works the same as _evaluate_human_judgement.py_, except that it applies unit-length normalization and SVD to the vectors before returning the correlation. The command-line arguments are the same.

# Weighting 
This directory contains the code necessary to utilize our Spacy-based weighting schema for your composed vectors.
### get_pos_files.py 
Iterates through the evaluation set and collects all tags, pos, and dep tags that are present before writing them to their respective files. This is a necessary step to utilize any of our weighting schema. Ittakes only one command-line argument, --i, which gives the location of the evaluation set. It writes the pos files to the directory in which it is placed.  
### get_dilation_weights.py
An algorithm that finds the optimal weights for each part of speech using dilation/decomposition. It first finds and returns the "inner" weights, before proceeding to the outer weights. This code takes in the following command-line arguments: --e gives the evaluation set that should be used to obtain the weights, --v gives the path to the word vectors, --p gives the  type of tag (google, tag, or dep), and --t gives the path to the tag list (generated above). 
### line_search.py 
Finds the optimal weights for weighted addition of the vectors using line search and prints them to the console. This code takes in the following command-line arguments: --e gives the evaluation set to be used to set the weights, --v gives the path to the word embeddings, --p gives the type of tag (google, tag, or dep), and --t gives the path to the tag list (generated above). 
### search_all.py
Tries all weights between 0 and 4 for each part of speech and then prints the optimal weight for each part of speech. This code takes in the following command-line arguments: --e gives the evaluation set to be used to set the weights, --v gives the path to the word embeddings, --p gives the type of tag (google, tag, or dep), and --t gives the path to the tag list (generated above).

## Authors
This code was developed by Sarah Payne (University of Pennsylvania).
