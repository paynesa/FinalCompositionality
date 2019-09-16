from pymagnitude import *
import string
punc = string.punctuation
vecs = Magnitude('/home/paynesa/compositionality/original_words.magnitude')
paths = ['ppdb-sample.tsv', 'wiki-sample.tsv']
for p in paths:
        out = "{}_4.txt".format(p.split("-")[0])
        for line in open(p, 'r'):
                line = line.strip().split('\t')
                k = False
                n = False
                l = True
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
                                if len(final) == 3:
                                        n = True
                                if len(final) > 4:
                                        print(final)
                                        l = False
                                for f in final:
                                        if f.strip() not in vecs:
                                                k = False
                               	flist.append("_".join(final))
                        elif len(word.strip().split()) ==1: 
                                flist.append(word.strip())
                        else:
                                k = False
                if k and n and l:
                        with open('total_4.txt', 'a') as f:
                                f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))
                        with open(out, 'a') as f:
                                f.write("{} {} {} {}\n".format(flist[0], flist[1], line[0], line[-1]))

