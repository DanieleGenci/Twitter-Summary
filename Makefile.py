import numpy as np
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict
import pickle
import operator
import re, math
from collections import Counter
from Tkinter import *

WORD = re.compile(r'\w+')
txt_open = "clean_tweets.txt"

#define functions that are used to build the algorithm

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


def cosine_sim(u,v):
    return np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)))

def ngrams(sentence, n):
  return zip(*[sentence.split()[i:] for i in range(n)])

def tfidf(corpus, vocab):
    """
    INPUT:

    corpus = [('this is a foo bar', [1, 1, 0, 1, 1, 0, 0, 1]), 
    ('foo bar bar black sheep', [0, 2, 1, 1, 0, 0, 1, 0]), 
    ('this is a sentence', [1, 0, 0, 0, 1, 1, 0, 1])]

    vocab = ['a', 'bar', 'black', 'foo', 'is', 'sentence', 
    'sheep', 'this']

    OUTPUT:

    [[0.300, 0.300, 0.0, 0.300, 0.300, 0.0, 0.0, 0.300], 
    [0.0, 0.600, 0.600, 0.300, 0.0, 0.0, 0.600, 0.0], 
    [0.375, 0.0, 0.0, 0.0, 0.375, 0.75, 0.0, 0.375]]

    """
    def termfreq(matrix, doc, term):
        try: return matrix[doc][term] / float(sum(matrix[doc].values()))
        except ZeroDivisionError: return 0
    def inversedocfreq(matrix, term):
        try: 
            return float(len(matrix)) /sum([1 for i,_ in enumerate(matrix) if matrix[i][term] > 0])
        except ZeroDivisionError: return 0

    matrix = [{k:v for k,v in zip(vocab, i[1])} for i in corpus]
    tfidf = defaultdict(dict)
    for doc,_ in enumerate(matrix):
        for term in matrix[doc]:
            tf = termfreq(matrix,doc,term)
            idf = inversedocfreq(matrix, term)
            tfidf[doc][term] = tf*idf

    return [[tfidf[doc][term] for term in vocab] for doc,_ in enumerate(tfidf)]


def corpus2vectors(corpus):
    def vectorize(sentence, vocab):
        return [sentence.split().count(i) for i in vocab]
    vectorized_corpus = []
    vocab = sorted(set(chain(*[i.lower().split() for i in corpus])))
    for i in corpus:
        vectorized_corpus.append((i, vectorize(i, vocab)))
    return vectorized_corpus, vocab


data = open(txt_open,'r')

tweet_list = list()

for line in data:
  tweet_list.append(line)

data.close()
dict_words= dict()
num_word = 0
 
  

num_word = 0
 # thanks to the python dictionary, calculates the frequency of each single word.
 
for line in tweet_list:
  if len(line) > 1:
    words = line.split()
    for word in words:
      num_word = num_word + 1 
      dict_words[word] = dict_words.get(word,0) + 1


file = open("dictionary1.txt",'w')

file.write(str(dict_words))
file.close


file.write(str(dict_words))
file.close
dict2_words = dict()

#thanks to the python dictionary and using "ngram function", calculates the frequency of each couple of words called "bigrams" 


for line in tweet_list:
    words = ngrams(line,2)
    for word in words:
      dict2_words[word] = dict2_words.get(word,0) + 1

file = open("dictionary_grams.txt",'w')

file.write(str(dict2_words))
file.close

dict_prob = dict()

#calculate posterior probability of each bigram that is found in tweets' text 

for word in dict2_words:
  value = dict2_words[word]
  a = 0
  for words in word:
    if a == 0:
      p = value/float(dict_words[words])
      dict_prob[word] = p
      a = a + 1
      
#calculate posterior probability of each tweet. 

file = open("dictionary_prob.txt",'w')
file.write(str(dict_prob))
file.close

dict_tweet = dict()
for line in tweet_list:
    words = ngrams(line,2)
    prob_line = 1
    line.replace("\n","")
    for word in words:
      prob_line = prob_line*float(dict_prob[word])
      dict_tweet[line] = prob_line



file = open("dictionary_probTweet.txt",'w')
file.write(str(dict_tweet))
file.close()

#we identify the 6 tweets with the higher  posterior probability and take them as main blocks to represent the event.

max_1 = max(dict_tweet.iteritems(), key=operator.itemgetter(1))[0]
del dict_tweet[max_1]
max_2 = max(dict_tweet.iteritems(), key=operator.itemgetter(1))[0]
del dict_tweet[max_2]
max_3 = max(dict_tweet.iteritems(), key=operator.itemgetter(1))[0]
del dict_tweet[max_3]
max_4 = max(dict_tweet.iteritems(), key=operator.itemgetter(1))[0]
del dict_tweet[max_4]
max_5 = max(dict_tweet.iteritems(), key=operator.itemgetter(1))[0]
del dict_tweet[max_5]
max_6 = max(dict_tweet.iteritems(), key=operator.itemgetter(1))[0]
del dict_tweet[max_6]


block_list = list()
block_list_copy = list()

block_list.append(max_1)
block_list.append(max_2)
block_list.append(max_3)
block_list.append(max_4)
block_list.append(max_5)
block_list.append(max_6)
block_list_copy.append(max_1)
block_list_copy.append(max_2)
block_list_copy.append(max_3)
block_list_copy.append(max_4)
block_list_copy.append(max_5)
block_list_copy.append(max_6)




#each of the remaining tweets is associated to one of the main blocks using the highest value of cosine similarity calculated between each remaining
#tweet and each main blocks.


data = open(txt_open,'r')

tweet_list = list()

for line in data:
      if len(line) > 1:
            tweet_list.append(line)
data.close()
a = 0

list_cs = list()
block1 = list()
block2 = list()
block3 = list()
block4 = list()
block5 = list()
block6 = list()

for tweet,tweets in product(tweet_list,block_list):
    vector1 = text_to_vector(tweet)
    vector2 = text_to_vector(tweets)
    a = a + 1
    cosine = get_cosine(vector1, vector2)
    list_cs.append(cosine)
    #print 'Cosine:', cosine
    if a == len(block_list):
         a = 0
         max_list = max(list_cs)
         index = list_cs.index(max_list)
         del list_cs[:]
         if index == 0:
              block1.append(tweet)
         if index == 1:
              block2.append(tweet)
         if index == 2:
              block3.append(tweet)
         if index == 3:
              block4.append(tweet)
         if index == 4:
              block5.append(tweet)
         if index == 5:
              block6.append(tweet)
              

#build a Tree to represent the information using the main blocks. 

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
     
#by default take as root the main block with the highest posterior probability      
##filling the fist the 2 node using cosine similarity
        
root = Tree()
root.data = block_list[0]
root.left = Tree()
root.right = Tree()

cs_block = list()

del block_list[0]

for tweet in block_list:
          vector1 = text_to_vector(root.data)
          vector2 = text_to_vector(tweet)
          cosine = get_cosine(vector1, vector2)
          cs_block.append(cosine)
max_list = max(cs_block)
index = cs_block.index(max_list)
root.left.data = block_list[index]
root.left.left = Tree()
root.left.right = Tree()
del block_list[index]

cs_block = list()

for tweet in block_list:
          vector1 = text_to_vector(root.data)
          vector2 = text_to_vector(tweet)
          cosine = get_cosine(vector1, vector2)
          cs_block.append(cosine)
max_list = max(cs_block)
index = cs_block.index(max_list)
root.right.data = block_list[index]
root.right.left = Tree()
root.right.right = Tree()
del block_list[index]

##filling the second level of node, using cosine similarity 

cs1_block = list()
cs2_block = list()
for tweet in block_list:
     
     vector1 = text_to_vector(root.left.data)
     vector2 = text_to_vector(tweet)
     cosine = get_cosine(vector1, vector2)
     cs1_block.append(cosine)

     vector1 = text_to_vector(root.right.data)
     vector2 = text_to_vector(tweet)
     cosine = get_cosine(vector1, vector2)
     cs2_block.append(cosine)
max_list1 = max(cs1_block)
max_list2 = max(cs2_block)
if max_list1 > max_list2:
     index = cs1_block.index(max_list1)
     root.left.left.data = block_list[index]
     del block_list[index]
else:
     index = cs2_block.index(max_list2)
     root.right.left.data = block_list[index]
     del block_list[index]
     
cs1_block = list()
cs2_block = list()

for tweet in block_list:
     
     vector1 = text_to_vector(root.left.data)
     vector2 = text_to_vector(tweet)
     cosine = get_cosine(vector1, vector2)
     cs1_block.append(cosine)

     vector1 = text_to_vector(root.right.data)
     vector2 = text_to_vector(tweet)
     cosine = get_cosine(vector1, vector2)
     cs2_block.append(cosine)

max_list1 = max(cs1_block)
max_list2 = max(cs2_block)

if max_list1 > max_list2:
     index = cs1_block.index(max_list1)
     if root.left.left.data == None:
          root.left.left.data = block_list[index]
          del block_list[index]
     else:
          root.left.right.data = block_list[index]
          del block_list[index]
else:
     index = cs2_block.index(max_list2)
     if root.right.left.data == None:
          root.right.left.data = block_list[index]
          del block_list[index]
     else:
          root.right.right.data = block_list[index]
          del block_list[index]



cs1_block = list()
cs2_block = list()

for tweet in block_list:
          vector1 = text_to_vector(root.left.data)
          vector2 = text_to_vector(tweet)
          cosine = get_cosine(vector1, vector2)
          cs1_block.append(cosine)
          
          vector1 = text_to_vector(root.right.data)
          vector2 = text_to_vector(tweet)
          cosine = get_cosine(vector1, vector2)
          cs2_block.append(cosine)

max_list1 = max(cs1_block)
max_list2 = max(cs2_block)

          
if max_list1 > max_list2:
     if (root.left.left.data != None) and (root.left.right.data != None):
          cs1_block = list()
          cs2_block = list()
          for tweet in block_list:
               vector1 = text_to_vector(root.left.left.data)
               vector2 = text_to_vector(tweet)
               cosine = get_cosine(vector1, vector2)
               cs1_block.append(cosine)
     
               vector1 = text_to_vector(root.left.right.data)
               vector2 = text_to_vector(tweet)
               cosine = get_cosine(vector1, vector2)
               cs2_block.append(cosine)
               
          max_list1 = max(cs1_block)
          max_list2 = max(cs2_block)
          if max_list1 > max_list2:
               index = cs1_block.index(max_list1)
               root.left.left.left = Tree()
               root.left.left.left.data = block_list[index]
               del block_list[index]
          else:
               index = cs2_block.index(max_list2)
               root.left.right.left = Tree()
               root.left.right.left.data = block_list[index]
               del block_list[index]
     else:
          if root.left.left.data == None:
               index = cs1_block.index(max_list1)
               root.left.left.data = block_list[index]
               del block_list[index]
          else:
               index = cs1_block.index(max_list1)
               root.left.right.data = block_list[index]
               del block_list[index]
               
          
else:
     if (root.right.left.data != None) and (root.right.right.data != None):
          cs1_block = list()
          cs2_block = list()
          for tweet in block_list:
               vector1 = text_to_vector(root.right.left.data)
               vector2 = text_to_vector(tweet)
               cosine = get_cosine(vector1, vector2)
               cs1_block.append(cosine)
     
               vector1 = text_to_vector(root.right.right.data)
               vector2 = text_to_vector(tweet)
               cosine = get_cosine(vector1, vector2)
               cs2_block.append(cosine)
               
          max_list1 = max(cs1_block)
          max_list2 = max(cs2_block)
          if max_list1 > max_list2:
               index = cs1_block.index(max_list1)
               root.right.left.left = Tree()
               root.left.left.left.data = block_list[index]
               del block_list[index]
          else:
               index = cs2_block.index(max_list2)
               root.right.right.left = Tree()
               root.left.right.left.data = block_list[index]
               del block_list[index]
     else:
          if root.right.left.data == None:
               index = cs2_block.index(max_list2)
               root.left.left.data = block_list[index]
               del block_list[index]
          else:
               index = cs2_block.index(max_list2)
               root.left.right.data = block_list[index]
               del block_list[index]



#print the the Tree

print root.data

print "\t",root.left.data

if root.left.left.data != None:
     print "\t\t",root.left.left.data
if root.left.left.left != None:
     print "\t\t\t",root.root.left.left.left.data
if root.left.left.right != None:
     print "\t\t\t",root.root.left.left.right.data

    
if root.left.right.data != None:
     print "\t\t",root.left.right.data
if root.left.right.left != None:
     print "\t\t\t",root.left.right.left.data
if root.left.right.right != None:
     print "\t\t\t",root.left.right.right.data


print "\t",root.right.data

if root.right.left.data != None:
     print "\t\t",root.left.left.data
if root.right.left.left != None:
     print "\t\t\t",root.right.left.left.data
if root.right.left.right != None:
     print "\t\t\t",root.right.left.right.data
if root.right.right.data != None:
     print "\t\t",root.left.right.data
if root.right.right.left != None:
     print "\t\t\t",root.right.right.left.data
if root.right.right.right != None:
     print "\t\t\t",root.right.right.right.data

print "\n\n\n\n\n\n"


def sel1():
     print "#1st block:"
     for tweet in block1:
          print tweet
     print"----------------------------------------------------------------"

def sel2():
     print "#2nd block:"
     for tweet in block2:
          print tweet
     print "---------------------------------------------------------------"
def sel3():
     print "#3th block:"
     for tweet in block3:
          print tweet
     print "---------------------------------------------------------------"
def sel4():
     print "#4th block:"
     for tweet in block4:
          print tweet
     print "---------------------------------------------------------------"
def sel5():
     print "#5th block:"
     for tweet in block5:
          print tweet
     print "---------------------------------------------------------------"
def sel6():
     print "#6th block:"
     for tweet in block6:
          print tweet
     print "---------------------------------------------------------------"

     
          



root = Tk()
R1 = Radiobutton(root, text="1st block",command=sel1)
R1.pack()

R2 = Radiobutton(root, text="2nd block",command=sel2)
R2.pack()

R3 = Radiobutton(root, text="3th block",command=sel3)
R3.pack()

R4 = Radiobutton(root, text="4th block",command=sel4)
R4.pack()

R5 = Radiobutton(root, text="5th block",command=sel5)
R5.pack()

R6 = Radiobutton(root, text="6th block",command=sel6)
R6.pack()

label = Label(root)
label.pack()
root.mainloop()

"""print "#1st block:",block_list_copy[0]
for tweet in block1:
     print tweet
print "#2st block:",block_list_copy[1]
for tweet in block2:
     print tweet
print "#3st block:",block_list_copy[2]
for tweet in block3:
     print tweet
print "#4st block:",block_list_copy[3]
for tweet in block4:
     print tweet
print "#5st block:",block_list_copy[4]
for tweet in block5:
     print tweet
print "#6st block:",block_list_copy[5]
for tweet in block6:
     print tweet"""
