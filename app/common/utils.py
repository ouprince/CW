#-*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))

ENVIRON = os.environ.copy()

import re
import unicodedata
import os
import random
import sys
import subprocess
from contextlib import contextmanager
import numpy as np
import numbers

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    unicode = str

import collections
import warnings

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle
try:
    from smart_open import smart_open
except ImportError:
    print("smart_open library not found; falling back to local-filesystem-only")
    def make_closing(base, **attrs):
        if not hasattr(base, '__enter__'):
            attrs['__enter__'] = lambda self: self
        if not hasattr(base, '__exit__'):
            attrs['__exit__'] = lambda self, type, value, traceback: self.close()
        return type('Closing' + base.__name__, (base, object), attrs)

    def smart_open(fname, mode='rb'):
        _, ext = os.path.splitext(fname)
        if ext == '.bz2':
            from bz2 import BZ2File
            return make_closing(BZ2File)(fname, mode)
        if ext == '.gz':
            from gzip import GzipFile
            return make_closing(GzipFile)(fname, mode)
        return open(fname, mode)

PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)

def get_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
            '%r cannot be used to seed a np.random.RandomState instance' %
                    seed)

class NoCM(object):
    def acquire(self):
        pass
    def release(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass

nocm = NoCM()
@contextmanager
def file_or_filename(input):
    if isinstance(input, string_types):
        yield smart_open(input)
    else:
        input.seek(0)
        yield input

def deaccent(text):
    if not isinstance(text, unicode):
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = u('').join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)

def copytree_hardlink(source, dest):
    copy2 = shutil.copy2
    try:
        shutil.copy2 = os.link
        shutil.copytree(source, dest)
    finally:
        shutil.copy2 = copy2

def tokenize(
        text,
        lowercase=False,
        deacc=False,
        encoding='utf8',
        errors="strict",
        to_lower=False,
        lower=False):
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)

def simple_tokenize(text):
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()

def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    tokens = [
            token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
                    if min_len <= len(token) <= max_len and not token.startswith('_')
             ]
    return tokens


'''转换成utf8编码'''
def any2utf8(text, errors='strict', encoding='utf8'):
    if isinstance(text, unicode):
        return text.encode('utf8')
    return unicode(text, encoding, errors=errors).encode('utf8')
to_utf8 = any2utf8

def any2unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)
to_unicode = any2unicode

def call_on_class_only(*args, **kwargs):
    raise AttributeError('This method should be called on a class object.')

def is_zhs(chars):
    '''是否全中文'''
    chars = any2unicode(chars)
    for i in chars:
        if not is_zh(i):
            return False
    return True

def is_zh(ch):
    '''是否中文'''
    x = ord(ch)
    if 0x2e80 <= x <= 0x2fef:
        return True
    elif 0x3400 <= x <= 0x4dbf:
        return True
    elif 0x4e00 <= x <= 0x9fbb:
        return True
    elif 0xf900 <= x <= 0xfad9:
        return True
    elif 0x20000 <= x <= 0x2a6df:
        return True
    else:
        return False

def is_punct(ch):
    x = ord(ch)
    if x < 127 and ascii.ispunct(x):
        return True
    elif 0x2000 <= x <= 0x206f:
        return True
    elif 0x3000 <= x <= 0x303f:
        return True
    elif 0xff00 <= x <= 0xffef:
        return True
    elif 0xfe30 <= x <= 0xfe4f:
        return True
    else:
        return False

def exec_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=ENVIRON)
    out, err = p.communicate()
    return out, err

def create_dir(target, remove = False):
    rmstr = ""
    if os.path.exists(target):
        if remove: rmstr = "rm -rf %s &&" % target
        else:return None
    return exec_cmd('%smkdir -p %s' % (rmstr, target))


#print (any2unicode("我们不一样"))
