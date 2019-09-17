# Vector-Based Semantic Compositionality
This code offers a collection of methods for creating and evaluating composed phrase embeddings. These embeddings are composed in a number of ways, including methods of addition and multiplication originally proposed by [Mitchell and Lapata (2010](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01106.x). We additionally give methods for incorporating image information using [multimodal embeddings](https://github.com/paynesa/multimodal), and for weighting words in a phrase based on their syntactic value according to the [Spacy model](https://spacy.io/usage/linguistic-features). The code is divided into sections as follows: 

# Embeddings 
This folder contains the code (but not the data) necessary to obtain the embeddings for the rest of the work. The code is divided up as follows:
### co-occurrence_embeddings.py
Mitchell and Lapata conduct all of their experiments on 2000D co-occurrence embeddings derived from a lemmatized version of the [British National Corpus](http://www.natcorp.ox.ac.uk/). To run this code, you will need to obtain a copy of the corpus and clean and lemmatize it. Then run the code with the parameters --i and --o: the directory containing the lemmatized BNC and the file where you would like your embeddings to be saved, respectively. The embeddings may then be converted to the [Magnitude](https://github.com/plasticityai/magnitude) format on the command line. 
### get_word_embeddings.py
Iterates through a list of words and phrases (joined by "\_") and obtains the word embeddings for the words in the phrases, then writes them to an output file. This gives the embeddings needed to compose embeddings in the rest of the code. This file takes in three arguments: --l is the path to the list of words and phrases, --w is the path to the word embeddings, which should be in [Magnitude](https://github.com/plasticityai/magnitude) format, and --o is the file where you would like your embeddings to be written. This file can be converted to the Magnitude format in the command line after creation. 
### predict_mmembeddings.py 
Uses a [ultimodal embedding model]https://github.com/paynesa/multimodal) to predict the multimodal embeddings for the word embeddings created by _get_word_embeddings.py_. It takes in three command-line arguments: --m is the path to the model you'd like to use to generate your multimodal embeddings, --p is the path to the prediction set (in this case, the same as the file created by _get_word_embeddings.py_. This file should not be in Magnitude format) and --o is the path where you would like your embeddings to be written. This output can once again be converted to Magnitude in the command line. 
























# Multimodal Embeddings
This code offers a collection of models which can be trained to create multimodal embeddings for a variety of applications. We also offer multiple options for data processing, including handling of repeated and OOV words.

Before beginning, please ensure that your data is in a format that is compatable with this code. In order to be compatable, your word embeddings should be in the [magnitude](https://github.com/plasticityai/magnitude) format, which was developed by [Patel, Callison-Burch, et al. (2018)](https://www.cis.upenn.edu/~ccb/publications/magnitude-fast-efficient-vector-embeddings-in-python.pdf). You can convert most common file-formats to magnitude from the command line using the directions in the link above. Please name your output file [word.magnitude].

Additionally, if your unprocessed image embeddings contain images that you do not want the model to be trained on, ensure that 'column-' comes at the beginning of the words that should not be processed. You may also choose to place 'row-' at the beginning of the ones that should be processed. 

#### Example:
```
'row-car' 0  1 .5 ...
'column-cara' 0  1...
'row-cat' 1  0.5 1...
```
would successfully exclude the Spanish word 'cara' so that only the English words 'car' and 'cat' are processed. 

Additionally, you should place all unprocessed image embeddings in .txt file(s) in a single directory with nothing else in it. These text files must be readable by [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html). We further recommend that you place your magnitude word embeddings in a directory with nothing else in it (separate from the location of your unprocessed image embeddings). In this directory, you should also create two subdirectories: OOV and no_OOV. It will then be possible to save your processed image embeddings and training sets to this directory, placing the corresponding training sets in OOV and no_OOV subdirectories depending on whether or not they include OOV word embeddings. 

## load_data.py
This file loads and cleans your data before creating the training sets for the model creation. 

load_data.py takes in 3 mandatory command-line arguments, and one optional argument. They are as follows: 

--i The path to the directory in which you placed your unprocessed image embeddings

--o The path to the directory where you would like the processed magnitude files and training sets to be saved

How you would like the image embeddings to be processed:
* 'avg' if you would like to average all vectors corresponding to the same word
* 'iter' if you would like vectors corresponding to the same words to appear separately in the training set

--w (optional) the location of the word embeddings, if it is different from the location given in --o

#### Example 1:
```
python3 load_data.py --i /home/data --o /home/results avg
```
Will load the image embeddings located in the 'data' folder and save the processed embeddings and training sets to 'results.' Repeated words' embeddings will be averaged. No 4th argument was given, so the word embeddings are located in 'results.'

#### Example 2: 

```
python3 load_data.py --i /home/data --o /home/results iter --w /home/embeddings
```
Will load the image embeddings located in the 'data' folder and save the processed embeddings and training sets to 'results.' Repeated words will not be averaged. A 4th argument was given, so the word embeddings are located in 'embeddings.'

#### Output
You will be notified as each file containing image embeddings is processed. Once all of the files have been processed, they will be converted to the magnitude format, and you will be notified of this as well. Finally, upon successful conversion, the training sets X_TRAIN and Y_TRAIN will be created, and you will be ready to work with the model of your choice.

#### Note
If magnitude conversion is not successful, you may need to convert the text file to magnitude on the command-line. Then, to create the training sets, open load_data.py, comment out the indicated lines, and re-run it. 

## model.py
This file creates and trains models and saves them and predictions. Additionally, it can load existing models and make predictions with them. Model.py takes in the following arguments:

--lr The learning rate (default = 0.1 for both models)

--u Number of hidden units for the neural model (default = 300)

--e Number of epochs for training (default = 25 for neural net, 175 for linear)

--s Path for saving model if you are creating a new model

--l path to existing model if loading existing model

--i path to the directory containing x_train and y_train if training a new model

--p path to the directory containing prediction sets if different from that given in (i)

The type of model:
* linear
* neural

#### Example 1:
```
python3 model.py linear --i /home/trainingdata --s /home/model
```
Creates a linear model with default parameters (0.1 learning rate; 175 epochs) and saves it to /home/model. The directory /home/trainingdata contaings x_train and y_train as will as the prediction sets.

#### Example 2:
```
python3 model.py neural --i /home/trainingdata --s /home/model --u 200 --lr 0.01 --e 100 --p /home/predictions
```
Creates a neural model with 200 hidden units, a learning rate of 0.01, and 100 epochs. It then saves predictions based off of files in /home/predictions and saves the model and these predictions to /home/model.

#### Example 3: 
```
python3 model.py linear --l /home/model/linear_model --p /home/predictions
```
Opens the existing linear model located at /home/model/linear_model and makes predictions based off of the files stored in /home/predictions. These predictions will be saved to /home/model/.

#### Output
You will be notified when your model is created/loaded, when it is trained if you are training a new model, and when the predictions are being run and saved. 


## evaluation.py
This file evaluates the predictions of the model against the following similarity tests: 
* Wordsim_sim
* Wordsim_rel
* SimLex
* MEN
* SemSim
* VisSim
Additionally, it evaluates on the entire sets, on VIS words in these sets, and on ZS words in these sets. The paths to these evaluation files are currently hard-coded but will be updated soon. 

evaluation.py takes in the following arguments:

type of evalution:
 * normal for simply evaluating the predictions of the model
 * c_linear or c_neural to evaluate the predictions concatenated with the GloVe vectors
 
path to the model and the predictions

--w the path to the word vectors if evaluating a concatenated model

#### Example 1:
```
python3 evaluation.py normal /home/model/linear
```
Will evaluate the predictions of the model linear.h5, located in /home/model/. It will not concatenate the predictions with GloVe vectors.

#### Example 2: 
```
python3 evaluation.py c_neural /home/neural_model --w /home/wordembeddings/word.magnitude
```
Will evaluate the predictions of the model neural_model.h5, located in /home/. It will concatenate these predictions with the word embeddings located at /home/wordembeddings/word.magnitude and evaluate the concatenated predictions against the human evaluations. 

## Authors
This code was developed by Minh Nguyen (Swarthmore) and Sarah Payne (University of Pennsylvania).
