from __future__ import print_function
from __future__ import division
import os
import sys
from tqdm import tqdm
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    xrange = range
from common import utils
from numpy import dot, zeros, dtype, float32 as REAL,\
    double, array, vstack, fromstring, sqrt, newaxis,\
        ndarray, sum as np_sum, prod, ascontiguousarray,\
            argmax


class Vocab(object):
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):
        return self.count < other.count

    def __str__(self):
        vals = [
                 '%s:%r' %
                          (key,
                               self.__dict__[key]) 
                               for key in sorted(
                   self.__dict__) if not key.startswith('_')]

        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class KeyedVectors():
    def __init__(self):
        self.syn0 = []
        self.syn0norm = None
        self.vocab = {}
        self.index2word = []
        self.vector_size = None

    @property
    def wv(self):
        return self

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])
        super(KeyedVectors, self).save(*args, **kwargs)

    @classmethod
    def load_word2vec_format(
                    cls,
                    fname,
                    fvocab=None,
                    binary=False,
                    encoding='utf-8',
                    unicode_errors='strict',
                    limit=None,
                    datatype=REAL):
        counts = None
        if fvocab is not None:
            print("loading word counts from %s" % fvocab)
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        print("loading projection weights from %s" % fname)
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())
            # vocab_size = 242215  vector_size = 100
            if limit:
                vocab_size = min(vocab_size, limit)
            result = cls()
            result.vector_size = vector_size
            result.syn0 = zeros((vocab_size, vector_size), dtype=datatype)

            def add_word(word, weights):
                word_id = len(result.vocab)
                # print("word id: %d, word: %s, weights: %s" % (word_id, word, weights))
                if word in result.vocab:
                    print("duplicate word '%s' in %s, ignoring all but first" %(word, fname))
                    return
                if counts is None:
                    result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
                elif word in counts:
                    result.vocab[word] = Vocab(index=word_id, count=counts[word])
                else:
                    print("vocabulary file is incomplete: '%s' is missing" %word)
                    result.vocab[word] = Vocab(index=word_id, count=None)

                result.syn0[word_id] = weights
                result.index2word.append(word)

            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                #binary_len = 400
                #tq = tqdm(xrange(vocab_size))
                for _ in xrange(vocab_size):
                #for _ in tq:
                    #tq.set_description("Loading Binary file ,size %d" %vocab_size)
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError(
                                "unexpected end of input; is count incorrect or file otherwise damaged?")
                        if ch != b'\n':
                            word.append(ch)

                    word = utils.to_unicode(
                                        b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    add_word(word, weights)
            else:
                for line_no in xrange(vocab_size):
                    line = fin.readline()
                    if line == b'':
                        raise EOFError(
                            "unexpected end of input; is count incorrect or file otherwise damaged?")
                    parts = utils.to_unicode(line.rstrip(),
                                        encoding=encoding,
                                        errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError(
                            "invalid vector on line %s (is this really the text format?)" %line_no)
                    word, weights = parts[0], [REAL(x) for x in parts[1:]]
                    add_word(word, weights)

        if result.syn0.shape[0] != len(result.vocab):
            print("duplicate words detected, shrinking matrix size from %i to %i" %(result.syn0.shape[0], len(result.vocab)))
            result.syn0 = ascontiguousarray(result.syn0[: len(result.vocab)])
        assert (len(result.vocab), vector_size) == result.syn0.shape
        print("loaded %s matrix from %s" % (result.syn0.shape, fname))
        return result

    def word_vec(self, word, use_norm=False):
        if word in self.vocab:
            if use_norm:
                result = self.syn0norm[self.vocab[word].index]
            else:
                result = self.syn0[self.vocab[word].index]

            result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)


import unittest
class Test(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_load_w2v_data(self):
        _fin_wv_path = os.path.join(curdir, os.path.pardir,os.pardir,'data', 'word2vec','news.w2v.bin.gz')
        #_fin_stopwords_path = os.path.join(curdir, 'data', 'stopwords.txt')
        kv = KeyedVectors()
        binary = True
        kv.load_word2vec_format(
                        _fin_wv_path,
                        binary=binary,
                        unicode_errors='ignore')
def test():
    unittest.main()

if __name__ == '__main__':
    test()
