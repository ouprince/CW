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

ENVIRON = os.environ.copy()
import json
import text.textsum as textsum
import common.tokenizer as tokenizer
from common import utils as common_utils
from text.clustering.cw import ChineseWhispers

def wordlen(s):
    s = common_utils.any2unicode(s).decode("utf-8")
    return len(s)

def isfloat(s):
    try:
        float(s)
        return True
    except:
        return False

def points(s):
    return s.count(".") > 0

def processing_data(from_, to_):
    '''
    Process data
    '''
    print("test_cize_cw_job: input", from_)
    print("test_cize_cw_job: output", to_)

    if from_ and os.path.exists(from_):
        data = []
        with common_utils.smart_open(from_) as fin:
            for x in fin.readlines():
                o = x.split('\t')
                id = o[0].strip()
                post = o[-1].strip()
                data.append([id, post])
    
        cw = ChineseWhispers()
        neighbours = 20
        threshold = 0.6
        iterations = 100
        count = len(data)
        if count < 1000:
            neighbours = 10
            threshold = 0.8
            iterations = 100
        elif count >= 1000 and count < 10000:
            neighbours = 15
            threshold = 0.7
            iterations = 200
        elif count >= 10000:
            neighbours = 20
            threshold = 0.6
            iterations = 300

        cw.g(data, neighbours = neighbours, threshold = threshold)
        result = cw.clust(iterations = iterations)

        indices = result.keys()
        output = dict()
        output['errorNo'] = 0
        output['records'] = []
        output['errorInfo'] = 'success'
        output['rc'] = 1

        for o in indices:
            txts = []
            dedup = set()
            vol = dict()
            vol['vol'] = "cluster_id_%s" % o
            vol['members'] = [dict({"id": o, "text": cw.t(o)})]
            txts.append(cw.t(o))
            dedup.add(o)
            for x in result[o]:
                if x in dedup: continue
                dedup.add(x)
                txts.append(cw.t(x))
                vol['members'].append(dict({"id": x, "text": cw.t(x)}))
            keywords_k, keywords_s = textsum.keywords("".join(txts), 5)
            vol["keywords"] = keywords_k
            output['records'].append(vol)

    with open(to_, "w") as fout:
        json.dump(output, fout, ensure_ascii=False, allow_nan=True, encoding="utf-8")


import unittest
class Test(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_cw_clustring(self):
        print("test_cw_clustring")
        to_ = os.path.join(rootdir, 'tmp', 'test_cw')
        from_ = os.path.join(rootdir, 'corpus', 'gfzq', 'gfzq.2017-08-25.visitor')
        data = []
        with common_utils.smart_open(from_) as fin:
            for x in fin.readlines():
                o = x.split()
                id = o[0].strip()
                post = o[1].strip()
                data.append([id, post])
        cw = ChineseWhispers()
        cw.g(data)
        result = cw.clust(iterations = 300)
        indices = result.keys()
        for o in indices:
            if len(result[o]) == 0: continue
            print("%s %s %s" % ('*'*20, 'volume center: %s , size: %d' % (o, len(result[o])), '*'*20))
            print("center %s" % (cw.t(o)))
            for x in result[o]:
                print("member: %s %s" % (x, cw.t(x)))

def test():
    unittest.main()

if __name__ == "__main__":
    test()
