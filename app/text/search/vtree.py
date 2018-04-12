#-*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
rootdir=os.path.join(curdir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(os.path.join(rootdir, 'app'))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from sklearn.neighbors import KDTree
import numpy
from common import tokenizer
from common import similarity
from common import utils as common_utils
from tqdm import tqdm

_sim_molecule = lambda x: numpy.sum(x, axis=0)

class VectorKDTree(object):
    def __init__(self):
        self.data = []
        self.indices = []
        self.vectors = []
        self.tokens = []
        self.text = []
        self.graph = None
        self.kdt = None

    def v(self, sentence):
        tokens = []
        words, tags = tokenizer.word_segment(sentence)
        for (w,t) in zip(words,tags):
            if t.startswith("n") or t.startswith("v"):
                tokens.append(w)

        if len(tokens) > 0:
            v, _ = similarity.vector(" ".join(tokens))
            if len(v) > 0:
                return _sim_molecule(v), tokens
            else:
                return None, tokens
        else:
            return None, None

    def vectorized(self, data):
        for x in tqdm(data):
            id, post = x
            bow, tokens = self.v(post)
            if bow is not None:
                assert len(bow) == 100, "wrong size of bag of words model."
                self.indices.append(id)
                self.vectors.append(bow)
                self.text.append(post)
                self.tokens.append(tokens)

        print("vectorized: size %d" % len(self.vectors))

    def kdtree(self, metric='minkowski'):
        if len(self.vectors) > 0:
            print ("building training points is ",len(self.vectors))
            self.kdt = KDTree(numpy.array(self.vectors, dtype=float), leaf_size=30, metric = metric)
            print("kdtree: done built.")
            return True
        else:
            print("kdtree: vectors is None for KDTree Built.")
            return False

    def neighbours(self, v, size = 10):
        if not self.kdt:
            print("neighbours: KDTree is not built yet.")
            yield None, None, None
        [distances], [points] = self.kdt.query(numpy.array([v]), k = size, return_distance = True)
        assert len(distances) == len(points), "distances and points should be in same shape."
        for (x,y) in zip(points, distances):
            yield self.indices[x], self.text[x], y

import unittest
class Test(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_get_neighbors(self):
        print("test_get_neighbors")
        to_ = os.path.join(rootdir, 'tmp', 'test_doc2vec')
        from_ = os.path.join(rootdir, 'corpus', 'gfzq', 'gfzq.2017-08-25.visitor.less')
        text = "这个股票你看好吗"
        T = VectorKDTree()
        data = []
        with common_utils.smart_open(from_) as fin:
            for x in fin.readlines():
                o = x.split()
                id = o[0].strip()
                post = o[1].strip()
                data.append([id, post])

        T.vectorized(data)
        T.kdtree()
        v, _ = T.v(text)
        print("sen [%s] neighbours: \n" % text)
        for x,y,z in T.neighbours(v, size = 10):
            print("id: %s, post: %s, distance: %f" % (x,y.decode(encoding="utf-8"),z))

def test():
    unittest.main()

if __name__ == "__main__":
    test()
