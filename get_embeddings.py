from pymagnitude import *
import string
import numpy as np
np.set_printoptions(suppress=True)
punc = string.punctuation
vectors = Magnitude("/data1/minh/magnitude/word.magnitude")
#l = ["the", "a", "an", "to"]
#total = []
for line in open('/data1/sarah/avg/OOV/words_processed.txt', 'r'):
        line = line.strip()
        if (line in vectors):
                c = vectors.query(line)
                with open('original_words.txt', 'a') as f:
                        f.write(line+ " ")
                        np.savetxt(f, c.reshape(1, c.shape[0]))
        elif (" " in line) and all(char not in punc for char in line):
                line = line.split()
                if all(word in vectors for word in line):
                        for word in line:
                                c = vectors.query(word.strip())
                                with open('original_words.txt', 'a') as f:
                                        f.write(word + " ")
                                        np.savetxt(f, c.reshape(1, c.shape[0]))

