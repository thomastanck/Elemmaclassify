from utils import persist_to_file, normalize
from AST import Var, Const, Dist, Func, Eq, Disj, Conj
import numpy as np
import functools
import zlib

def associate(x, y):
    """ Create association between two objects """
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    z = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
    return normalize(z)

def merge2(x, y, weight=0.5):
    """ Merge two associations together """
    z = normalize(weight * x + (1-weight) * y)
    return normalize(weight * x + (1-weight) * y)

def merge(it):
    """ Merge a bag of associations together """
    s = it[0].copy()
    for i in it[1:]:
        s += i
    s = normalize(s)
    return normalize(s)

def involution(x):
    """ Flips array from second element onwards """
    return np.concatenate((x[0:1], x[-1:0:-1]))

def decode(x, xy):
    """ (Attempts to) Decode association of two objects """
    return circular_convolution(involution(x), xy)

@functools.lru_cache()
def get_ground_vector(hrr_size, label):
    """ Deterministically generate a random vector from its label """
    rs = np.random.RandomState(
            zlib.adler32(
                (str(hrr_size)+label).encode('utf-8')
                ) & 0xffffffff)
    rs.randint(2)

    if ':' in label:
        parent, _, specifier = label.rpartition(':')
        parentvec = get_ground_vector(hrr_size, parent)
        newvec = merge2(parentvec, normalize(rs.standard_normal(hrr_size)))
        return newvec
    else:
        return normalize(rs.standard_normal(hrr_size))

class HRR:
    """ Interface for HRR manipulations """

    def __init__(self, hrr_size):
        self.hrr_size = hrr_size

    @functools.lru_cache(maxsize=8192)
    def fold_term(self, term):
        """ Outputs a HRR vector given a FOF term """
        if isinstance(term, Var):
            return self.var(term.name)
        elif isinstance(term, Const):
            return self.const(term.name)
        elif isinstance(term, Dist):
            return self.dist(term.name)
        elif isinstance(term, Func):
            return self.func(
                    term.funcname,
                    [ self.fold_term(x)
                      for x in term.args ])

    def encode_eq(self, eq):
        """ Outputs a HRR vector given a literal """
        return self.eq(
                eq.pos,
                self.fold_term(eq.t1),
                self.fold_term(eq.t2))

    def encode_disj(self, disj):
        """ Outputs a HRR vector given a disjunction of literals """
        return self.disj(
                disj.role,
                [ self.encode_eq(eq)
                  for eq in disj.eqs ])

    def encode_conj(self, conj):
        """ Outputs a HRR vector given a CNF formula """
        return self.conj(
                [ self.encode_disj(disj)
                  for disj in conj.disjs ])

    def var(self, name):
        """ Output a HRR vector given a variable name """
        raise NotImplementedError

    def const(self, name):
        """ Output a HRR vector given a constant name """
        raise NotImplementedError

    def dist(self, name):
        """ Output a HRR vector given a distinct object name """
        raise NotImplementedError

    def func(self, name, vecs):
        """ Output a HRR vector given function name and HRR vectors of its inputs """
        raise NotImplementedError

    def eq(self, pos, vec1, vec2):
        """ Output a HRR vector given positivity of this literal (literals are equations), and HRR vectors of the left and right sides """
        raise NotImplementedError

    def disj(self, role, vecs):
        """ Output a HRR vector given role of this clause, and HRR vectors of all the literals """
        raise NotImplementedError

    def conj(self, vecs):
        """ Output a HRR vector given HRR vectors of all the clauses """
        raise NotImplementedError

class FlatTreeHRR(HRR):
    def var(self, name):
        """ Output a HRR vector given a variable name """
        return get_ground_vector(self.hrr_size, 'Var:{}'.format(name))

    def const(self, name):
        """ Output a HRR vector given a constant name """
        return get_ground_vector(self.hrr_size, 'Const:{}'.format(name))

    def dist(self, name):
        """ Output a HRR vector given a distinct object name """
        return get_ground_vector(self.hrr_size, 'Dist:{}'.format(name))

    def func(self, name, vecs):
        """ Output a HRR vector given function name and HRR vectors of its inputs """
        arity = len(vecs)
        funcobj = get_ground_vector(self.hrr_size, 'Func:{}:{}'.format(arity, name))

        argobjs = [
                associate(
                    get_ground_vector(self.hrr_size, 'Arg:{}:{}:{}'.format(arity, name, i)),
                    vec)
                for i, vec in enumerate(vecs) ]
        result = merge([funcobj, merge(argobjs), merge(vecs)])
        return result

    def eq(self, pos, vec1, vec2):
        """ Output a HRR vector given positivity of this literal (literals are equations), and HRR vectors of the left and right sides """
        return merge([
            merge2(
                associate(get_ground_vector(self.hrr_size, 'Eq:{}:Left'.format(pos)), vec1),
                associate(get_ground_vector(self.hrr_size, 'Eq:{}:Right'.format(pos)), vec2)),
            vec1,
            vec2])

    def disj(self, role, vecs):
        """ Output a HRR vector given role of this clause, and HRR vectors of all the literals """
        roleobj = get_ground_vector(self.hrr_size, 'DisjRole:{}'.format(role))
        objs = [ associate(roleobj, vec) for vec in vecs ]
        merged = merge(objs)
        disjobj = associate(merged, merged) # Essentially associates each literal with each other
        return merge2(disjobj, merge(vecs))

    def conj(self, vecs):
        """ Output a HRR vector given HRR vectors of all the clauses """
        return merge(vecs)

# class TreeHRR(HRR):

# class FlatTreeHRR(HRR):

