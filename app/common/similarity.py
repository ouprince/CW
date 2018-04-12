#-*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import sys
import numpy as np
curdir = os.path.dirname(os.path.abspath(__file__))
rootdir=os.path.join(curdir, os.path.pardir, os.path.pardir)
sys.path.append(os.path.join(curdir, os.path.pardir))


PLT = 2
if sys.version_info[0] < 3:
    default_stdout = sys.stdout
    default_stderr = sys.stderr
    reload(sys)
    sys.stdout = default_stdout
    sys.stderr = default_stderr
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    PLT = 3

import json
import gzip
import shutil
from common.word2vec import KeyedVectors
from common.utils import any2utf8
from common.utils import any2unicode
import common.tokenizer as tokenizer

_vectors = None
_stopwords = set()

_fin_stopwords_path = os.path.join(rootdir, 'data', 'word2vec', 'news.stopwords.txt')
def _load_stopwords(file_path):
    global _stopwords
    words = open(file_path, 'r')
    stopwords = words.readlines()
    for w in stopwords:
        _stopwords.add(any2unicode(w).strip())

print(">> similarity: loading stopwords ...")
_load_stopwords(_fin_stopwords_path)

words = open(os.path.join(rootdir,'app','resources','stopwords.utf8'),'r')
stopwords = words.readlines()
for w in stopwords:
    _stopwords.add(any2unicode(w).strip())

_f_model = os.path.join(rootdir, 'data', 'word2vec', 'news.w2v.bin.gz')
def _load_w2v(model_file=_f_model, binary=True):
    if not os.path.exists(model_file):
        raise Exception("Model file %s does not exist" % model_file)
    return KeyedVectors.load_word2vec_format(
            model_file, binary=binary, unicode_errors='ignore')

print(">> similarity: loading vectors ...")

_vectors = _load_w2v(model_file=_f_model)

_sim_molecule = lambda x: np.sum(x, axis=0)  # 将x 按列相加，得到长度 100 的数组

def vector(sentence):
    global _vectors
    vectors = []
    for y in sentence.split():
        y_ = any2unicode(y).strip()
        if not y_ in _stopwords:
            try:
                vectors.append(_vectors.word_vec(y_))
            except KeyError as error:
                pass
    return vectors,len(vectors)

def _unigram_overlap(sentence1, sentence2):
    x = set()
    y = set()
    for x_ in sentence1.split():
        if x_ in _stopwords:
            continue
        x.add(x_)
            
    for y_ in sentence2.split():
        if y_ in _stopwords:
            continue
        y.add(y_)

    intersection = x & y #交集
    union = x | y        #并集
    return ((float)(len(intersection)) / (float)(len(union)))

def _levenshtein_distance(sentence1, sentence2):
    first = sentence1.split()
    second = sentence2.split()
    if len(first) > len(second):
        first, second = second, first
    distances = range(len(first) + 1)
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                        distances[index1 + 1],
                                        new_distances[-1])))
        distances = new_distances
    levenshtein = distances[-1]
    return 2 ** (-1 * levenshtein)

def _similarity_distance(s1, s2):
    vector_s1,len_s1 = vector(s1)
    vector_s2,len_s2 = vector(s2)
    a = _sim_molecule(vector_s1)
    b = _sim_molecule(vector_s2)
    g = 1 / (np.linalg.norm(a - b) + 1) #np.lianlg.norm 默认就是平方开根号(二阶范数)
    #u = _levenshtein_distance(s1, s2)
    u = _unigram_overlap(s1,s2)
    r = g * (12 + abs(len_s1 - len_s2)) + u * 0.8
    r = min(r , 1.0)
    return float("%.3f" %r)

def compare(s1, s2, seg=True):
    assert len(s1) > 0 and len(s2) > 0, "The length of s1 and s2 should > 0."
    if seg:
        s1, _ = tokenizer.word_segment(s1)
        s2, _ = tokenizer.word_segment(s2)
        s1 = ' '.join(s1)
        s2 = ' '.join(s2)

    return _similarity_distance(s1, s2)

import unittest
class Test(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_sim_sens(self):
        print("test_sim_sens")
        sen1 = ["计算出每篇文章的关键词", "计算出每篇文章的关键词" ,"每篇文章的关键词", "当然你会好奇这里的IDF是什么","我喜欢你"]
        sen2 = ["计算出每篇文章的关键词", "计算出每篇关键词" ,"计算出每篇文章", "IDF是什么","手机怎么解锁？"]
        for (x,y) in zip(sen1, sen2):
            print("相似度：%f, %s v.s. %s" % (compare(x, y), x, y))
            print("*"*20 + "\n")
def test():
    unittest.main()

if __name__ == "__main__":
    test()
