#-*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import sys
import re

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))

from common import log
logger = log.getLogger(__name__)

PLT = 2
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    PLT = 3

import jieba
import jieba.posseg as tokenizer

COMPANY_DICT_PATH = os.path.join(curdir, os.path.pardir, "resources", "vocab.company.utf8")
SF_DICT_PATH = os.path.join(curdir, os.path.pardir, "resources", "vocab.security.utf8")
SOUGOU_DICT_PATH = os.path.join(curdir, os.path.pardir, "resources", "vocab.sougou.utf8")
STOPWORD_DICT_PATH = os.path.join(curdir, os.path.pardir, "resources", "stopwords.utf8")
jieba.load_userdict(COMPANY_DICT_PATH)
jieba.load_userdict(SOUGOU_DICT_PATH)
jieba.load_userdict(SF_DICT_PATH)
jieba_stopwords = set()

def resolve_utf8(word):
    if PLT == 2:
        return word.encode("utf8")
    else:
        return word

def load_stop_words():
    if len(jieba_stopwords) > 0:
        return True
    if not os.path.exists(STOPWORD_DICT_PATH):
        return None
    with open(STOPWORD_DICT_PATH, "r") as fin:
        for x in fin:
            x = x.strip()
            if not x.startswith("#"): jieba_stopwords.add(x)
    print("jieba stopwords loaded.")
    return True

load_stop_words()

def seg_jieba(body):
    y = tokenizer.cut(body["content"], HMM=True)
    words, tags = [], []
    for o in y:
        if "type" in body and body["type"] == "nostopword":
            if o.word in jieba_stopwords: continue
        if "punct" in body and body["punct"] == False:
            if o.flag.startswith("x"): continue
        words.append(o.word)
        tags.append(o.flag)
    assert len(words) == len(tags), "words and tags should be the same length with jieba tokenizer."
    return words, tags

def cut(sent, cut_all=False, HMM=True):
    words = []
    for x in jieba.cut(sent.strip(), cut_all=cut_all, HMM=HMM):
        words.append(x)
    return words

def word_segment(utterance, punct = False, ty = "nostopword", vendor = "jieba"):
    words, tags = [], []
    try:
        if vendor == "jieba":
            words, tags = seg_jieba({
                                    "type": ty,
                                    "content": utterance,
                                    "punct": punct
                                    })
        else:
            raise Exception("None tokenizer.")
    except Exception as e:
        print("seg error\n", utterance, e)

    logger.debug("word seg result: %s, %s" % (" ".join(words), " ".join(tags)))
    return words, tags

person_names = []
with open(os.path.join(curdir, os.path.pardir, "resources", "names.utf8"), "r") as fin:
    [ person_names.append(x.strip()) for x in fin.readlines()]
assert len(person_names) > 0, "person names set should not be empty."

emoji = []
with open(os.path.join(curdir, os.path.pardir, "resources", "emoji.utf8"), "r") as fin:
    [ emoji.append(x.strip()) for x in fin.readlines()]
assert len(emoji) > 0, "emoji set should not be empty."

punct = []
with open(os.path.join(curdir, os.path.pardir, "resources", "punctuation.utf8"), "r") as fin:
    [ punct.append(x.strip()) for x in fin.readlines()]
assert len(punct) > 0, "punct set should not be empty."

def filter_name(utterance):
    result = []
    for o in utterance:
        if not o in person_names:
            result.append(o)
        else:
            result.append("TPERSON")
    return result

def filter_emoji(utterance):
    for o in emoji:
        utterance = utterance.replace(o, "")
    return utterance

def filter_number(utterance):
    utterance = re.sub("[0-9.]+", " TNUMBER", utterance)
    return utterance

def filter_url(utterance):
    utterance = re.sub("http[s]?://[^\s]*", "TURL", utterance)
    return utterance

def filter_date(utterance):
    utterance = re.sub(u"\d{1,}\s*年\d{1,}\s*月\d{1,}\s*日", "TDATE", utterance)
    utterance = re.sub(u"\d{1,}\s*月\d{1,}\s*日", "TDATE", utterance)
    utterance = re.sub(u"\d{1,}\s*月\d{1,}\s*日", "TDATE", utterance)
    utterance = re.sub(u"\d{2,}\s*年\d{1,}月", "TDATE", utterance)
    utterance = re.sub("\d{4}-\d{1,2}-\d{1,2}", "TDATE", utterance)
    return utterance

def filter_full_to_half(utterance):
    n = []
    utterance = utterance.decode('utf-8')
    for char in utterance:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = unichr(num)
        n.append(num)
    return ''.join(n)

def filter_eng_to_tag(utterance):
    utterance = re.sub("[A-Za-z]+", "TENGLISH", utterance)
    return utterance

def filter_special_punct(utterance):
    for o in punct:
        utterance = utterance.replace(o, " ")
    return utterance

COMP_DICT = set()
COMPANY_DICT_PATH = os.path.join(curdir, os.path.pardir, "resources", "vocab.company.utf8")
def _load_company_names():
    with open(COMPANY_DICT_PATH, "r") as fin:
        for x in fin.readlines():
            COMP_DICT.add(x.split()[0])
    print("company names dict loaded, len %d" % len(COMP_DICT))
_load_company_names()

SF_STOPWORDS = set()
SF_STOPWORDS_PATH = os.path.join(curdir, os.path.pardir, "resources", "stopwords.security.utf8")
def _load_sf_stopwords():
    with open(SF_STOPWORDS_PATH, "r") as fin:
        for x in fin.readlines():
            SF_STOPWORDS.add(x.strip())
    print("sf stopwords dict loaded, len %d" % len(SF_STOPWORDS))
_load_company_names()

def is_equal_query_as_array(x, t):
    '''
    判断 t是否全部在 x 里面
    '''
    r = []
    o = False
    for z in t:
        if z in x: r.append(z)
    if len(r) == len(t):
        o = True
    return o

def replacement(utterance):
    q = [] # result
    if is_equal_query_as_array(utterance, ["接单", "继续", "服务", "客户"]):
        return None
    if is_equal_query_as_array(utterance, ["用户", "登录"]):
        return None
    if is_equal_query_as_array(utterance, ["语音", "通话", "留意", "来电"]):
        return None
    if is_equal_query_as_array(utterance, ["客户", "成功", "签署", "协议"]):
        return None
    if is_equal_query_as_array(utterance, ["订单", "转让", "留意"]):
        return None
    for y in utterance:
        if not y in SF_STOPWORDS: q.append(y)
    p = []
    for z in q:
        if z in COMP_DICT:
            p.append("TCOMPANY")
        else:
            p.append(z)
    if len(p) < 3:
        return None
    return q

import unittest
class Test(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_word_segment(self):
        w, t = word_segment("添加入了Attention注意力分配机制后，使得Decoder在生成新的Target Sequence时，能得到之前Encoder编码阶段每个字符的隐藏层的>信息向量Hidden State，使得生成新序列的准确度提高。")
        for (x, y) in zip(w, t):
            print("word: %s, tag: %s" % (x, y))
def test():
    unittest.main()

if __name__ == "__main__":
    test()
