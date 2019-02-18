import nltk
import io
from collections import defaultdict
from nltk.corpus import wordnet as wn
from collections import Counter
import itertools
import re

#Generation of pos_tags from input file
input_file = 'input.txt'
input_data = " ".join(io.open(input_file, encoding='utf8').readlines())
tokens = nltk.word_tokenize(input_data)
pos_tag = nltk.pos_tag(tokens)
#Genration of noun list from tokenized text
nouns = []
nouns_list = []
for i in pos_tag:
    if i[1] == 'NN':
        nouns_list.append(i[0])
nouns = list(set(nouns_list))
#Word counts stored
count_words = {}
count = Counter()
count.update(tokens)
count_words = dict(count)
#Synset retrieval
synset = {}
for n in nouns:
    synset_list = wn.synsets(n)
    synset[n] = synset_list    
hypernym = {}
for i in synset:
    hyper = []
    if synset[i]:
        for j in synset[i]:
            hyper_var=j.hypernyms()
            if hyper_var:
                hyper.append(hyper_var[0])
                hypernym[i] = hyper
hyponyms = {}
for i in synset:
    hypo = []
    if synset[i]:
        for j in synset[i]:
            hypo_var=j.hyponyms()
            if hypo_var:
                hypo.append(hypo_var[0])
            hyponyms[i] = hypo
#Combining synsets             
comb_list = {}
for i in synset:
    z = []
    x = synset[i]
    if x:
        z.append(synset[i])
    if i in hyponyms:
        z.append(hyponyms[i])
    if i in hypernym:
        z.append(hypernym[i])
    combined = list(itertools.chain(*z))
    comb_list[i] = combined   
#Generation of lemma list
lemmas = {}
for i in comb_list:
    words = []
    for j in range(len(comb_list[i])):
        y = comb_list[i][j].lemma_names()
        words.append(y)
    combined = list(itertools.chain(*words))
    lemmas[i] = list(set(combined))   
#Formation of lexical chain
t1 = {}
for i in lemmas:
    t2 = []
    for j in lemmas:
        if i != j:
            for x in lemmas[i]:
                if x in lemmas[j]:
                    t2.append(j)
    t1[i] = list(set(t2))   
c1 = []
for i in t1:
    x = t1[i]
    x.insert(0, i)
    c1.append(x)    
order = []
for i in nouns:
    if i not in order:
        order.append(i)        
lexical_chain = {}
for i in order:
    t2 =[]
    for j in order:
        if i != j:
            flag = False
            count = 0
            while(flag == False):
                if count >= len(lemmas[i]):
                    flag = True
                elif (lemmas[i][count] in set(lemmas[j])) and count < len(lemmas[i]):
                    t2.append(j)
                    flag = True
                else:
                    count = count + 1
    lexical_chain[i] = list(set(t2))    
t3 = []
for i in lexical_chain:
    x = lexical_chain[i]
    x.insert(0,i)
    t3.append(x)
chain1 = t3
t4 =[]
for i in order:
    flag = False
    for j in t4:
        if i in set(j):
            flag = True
    if flag == False:
        xx = []
        for j in chain1:
            if i in set(j):
                xx.append(j)
                chain1.remove(j)
        combined = list(itertools.chain(*xx))
        t4.append(list(set(combined)))        
chain2 = t4
for i in chain2:
    for j in i:
        for x in chain2:
            if i != x:
                for m in x:
                    if j == m:
                        x.remove(j)                        
final = []
for x in chain2:
    arr = []
    for y in x:
        arr.append((y,(count_words[y])))
    final.append(arr)
#Printing the lexical chain
for i in range(len(final)):
    arr1 = []
    for x in final[i]:
        str1 = str(x[0]) + "(" + str(x[1]) + ")"
        arr1.append(str1)
    print("Chain %d : " %(i+1),arr1)   
