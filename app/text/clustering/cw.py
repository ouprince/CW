from __future__ import print_function
from __future__ import division
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
rootdir=os.path.join(curdir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(os.path.join(rootdir, 'app'))

ENVIRON = os.environ.copy()

import uuid
import numpy
import networkx
import copy
import json
from tqdm import tqdm
from random import shuffle
from common import tokenizer
from common import similarity
from common import utils as common_utils
from text.search.vtree import VectorKDTree
from text.search.posting import LuceneSearch
from text.search.posting import build_query_condition

class ChineseWhispers():
    '''
    Chinese Whispers Graph
    '''
    def __init__(self):
        self.G = networkx.Graph()
        self.dict = dict()
        self.data = []
        self.vtree = VectorKDTree()
        self.lucene = None
        self.jib = None
        self.workdir = None
        self.indexdir = None
        self.edges = []
        self.graph = None
        self.graphed = False

    def v(self, text):
        return self.vtree.v(text)

    def t(self, id):
        if id in self.dict:
            return self.dict[id]
        else:
            return None

    def g(self, data, neighbours = 10, threshold = 0.6):
        '''
        Build Graph
        '''
        try:
            self.add_nodes(data)
            self.add_edges(size = neighbours, threshold = threshold)
        except Exception as e:
            print(e)

    def _index(self):
        '''
        index data with lucene and vtree
        '''
        if self.data and len(self.data) > 0: pass
        else: raise BaseException("_index: invalid data.")
        print("index data")
        self.jid = str(uuid.uuid1())
        if 'CIZE_JOB_CW_WORKDIR' in ENVIRON:
            self.workdir = ENVIRON['CIZE_JOB_CW_WORKDIR']
        else:
            self.workdir = os.path.join(rootdir, 'app', 'static', 'jobs', self.jid)
        self.indexdir = os.path.join(self.workdir, 'lucene')
        common_utils.create_dir(self.workdir, remove = False)
        common_utils.create_dir(self.indexdir, remove = True)
        self.lucene = LuceneSearch(index_path = self.indexdir)
        self.lucene.index(self.data, remove = True)
        self.vtree.vectorized(self.data)
        self.vtree.kdtree()

    def add_nodes(self, data):
        '''
        add data as nodes.
        '''
        print("add nodes")
        if len(data) > 0: pass
        else: raise BaseException("add_nodes: invalid data.")
        nodes = []
        dup = dict()
        self.dict = dict()
        for o in tqdm(data):
            id, post = o
            if not post in dup:
                nodes.append(id)
                self.data.append(o)
                self.dict[id] = post
                dup[post] = id
        print("add nodes: unique = %d" % len(self.data))
        self.G.add_nodes_from(nodes)
        self._index()
        
    def neighbours(self, text, size = 10):
        '''
        Get neighbours
        '''
        words, tags = tokenizer.word_segment(text)
        if len(words) == 0:
            raise BaseException("invalid tokenizer result")
        matched = self.lucene.query(build_query_condition(words, tags), size = size)

        adjacents = []
        vec,_ = self.vtree.v(text)
        if vec and len(vec) > 0:
            for x,y,z in self.vtree.neighbours(vec, size = size):
                adjacents.append(dict({
                                    "id": x,
                                    "post": y,
                                    "distance": z
                                    }))
        neighbours = dict()
        for o in matched:
            neighbours[o['id']] = o
        for o in adjacents:
            if not o['id'] in neighbours:
                neighbours[o['id']] = o
            else:
                neighbours[o['id']]['distance'] = o['distance']
        return neighbours

    def add_edges(self, size = 10 , threshold = 0.6):
        '''
        Add edges for data
        '''
        print("add edges")
        if self.data and len(self.data) > 0: pass
        else: raise BaseException("add_edges: invalid nodes.")

        dup = set()
        for o in tqdm(self.data):
            id, post = o
            try:
                neighbours = self.neighbours(post, size = 10)
                for x in neighbours:
                    nid = neighbours[x]['id']
                    if "%s|%s" % (id, nid) in dup or "%s|%s" % (nid, id) in dup:
                        continue
                    dup.add("%s|%s" % (id, nid))
                    npost = neighbours[x]['post']
                    post_w, post_t = tokenizer.word_segment(post)
                    npost_w, npost_t = tokenizer.word_segment(npost)
                    post_repl = tokenizer.replacement(post_w)
                    npost_repl = tokenizer.replacement(npost_w)

                    if len(post_repl) > 0 and len(npost_repl) > 0:
                        weight = similarity.compare(" ".join(post_repl) , " ".join(npost_repl), seg = False)
                        if weight > threshold: # filter out low weights
                            self.edges.append((id, nid, {'weight': weight}))
            except: pass # ignore any mistake

        if len(self.edges) > 0:
            self.G.add_edges_from(self.edges)
            self.graphed = True
            print("add_edges: done, size " % len(self.edges))

    def clust(self, iterations = 100):
        '''
        Cluster data
        '''
        print("clust")
        if self.graphed: pass
        else: raise BaseException("clust: invalid graph.")
        for o in self.data:
            try:
                id, _ = o
                self.G.node[id]['class'] = id
            except: pass # ignore any mistake
    
        for iteration in tqdm(range(iterations)):
            gn = self.G.nodes()
            rg = range(len(gn))
            reorder = range(len(gn))
            shuffle(rg)
            for (k,v) in enumerate(gn):
                reorder[rg[k]] = v
            gn = reorder

            for node in gn:
                neighs = self.G[node]
                classes = {}
                for ne in neighs:
                    if self.G.node[ne]['class'] in classes:
                        classes[self.G.node[ne]['class']] += self.G[node][ne]['weight']
                    else:
                        classes[self.G.node[ne]['class']] = self.G[node][ne]['weight']

                maxsum = 0
                maxclass = None
                for c in classes.keys():
                    if classes[c] > maxsum:
                        maxsum = classes[c]
                        maxclass = c
                if maxclass != None:
                    self.G.node[node]['class'] = maxclass

        result = dict()
        gn = self.G.nodes()
        for nid in gn:
            n = self.G.node[nid]
            if not 'class' in n: continue
            c = n['class']
            if c in result:
                result[c].append(nid)
            elif c == nid:
                result[c] = []
            else:
                result[c] = [nid]

        return result
                    
