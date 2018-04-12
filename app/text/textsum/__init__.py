#-*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import jieba.analyse as analyzer
JIEBA_ANALYZER_IDF = os.path.join(curdir, os.path.pardir, os.path.pardir, "resources", "similarity.vocab.idf")
JIEBA_ANALYZER_STOPWORDS = os.path.join(curdir, os.path.pardir, os.path.pardir, "resources", "jieba_ext", "stop_words.txt")
analyzer.set_idf_path(JIEBA_ANALYZER_IDF)
analyzer.set_stop_words(JIEBA_ANALYZER_STOPWORDS)


def keywords(content, topK=10, vendor = "tfidf", title = None):
    words = []
    scores = []
    if vendor == 'tfidf':
        for x,y in analyzer.extract_tags(content, topK=topK, withWeight=True):
            words.append(x)
            scores.append(y)
    else:
        raise BaseException("Invalid vendor")
    return words, scores


import unittest
class Test(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_idf_keywords(self):
        print("test_idf_keywords")
        content = ''' “习近平主席那次讲话具有历史意义。”时隔一年，世界经济论坛创始人施瓦布回忆起习主席的演讲，依然感叹不已。他强调，今年论坛确定这一主题，就是希望可以继续顺承习主席去年演讲时所提到的“共建人类命运共同体”的主张。'''
        words, scores = keywords(content)
        for x,y in zip(words, scores):
            print("word: %s, score: %s" % (x,y))


def test():
    unittest.main()

if __name__ == "__main__":
    test()
