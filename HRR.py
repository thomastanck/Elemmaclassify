from utils import persist_to_file, normalize
from AST import Var, Const, Func, Eq, Disj, Conj
import numpy as np

VEC_DIM = 512

def associate(x, y):
    """ Create association between two objects """
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    z = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
    if np.ndim(z) == 1:
        z = z[None, :]
    return z

def merge2(x, y, weight=0.5):
    """ Merge two associations together """
    return normalize(weight * x + (1-weight) * y)

def merge(it):
    """ Merge a bag of associations together """
    s = it[0]
    for i in it[1:]:
        s += i
    return normalize(s)

def involution(x):
    """ Flips array """
    if np.ndim(x) == 1:
        x = x[None, :]
    return np.concatenate([x[:, None, 0], x[:, -1:0:-1]], 1)


def decode(x, xy):
    """ (Attempts to) Decode association of two objects """
    return circular_convolution(involution(x), xy)

@persist_to_file('HRR_ground_vecs_{}.pickle'.format(VEC_DIM))
def get_ground_vector(label):
    if ':' in label:
        parent, _, specifier = label.rpartition(':')
        return merge2(get_ground_vector(parent), normalize(np.random.standard_normal(VEC_DIM)))
    else:
        return normalize(np.random.standard_normal(VEC_DIM))

class HRR:
    """ Interface for HRR manipulations """

    @classmethod
    def fold_term(cls, term):
        """ Outputs a HRR vector given a FOF term """
        if isinstance(term, Var):
            return cls.var(term.name)
        elif isinstance(term, Const):
            return cls.const(term.name)
        elif isinstance(term, Func):
            return cls.func(
                    term.funcname,
                    [ fold_term(v, c, f, x)
                      for x in term.args ])

    @classmethod
    def encode(cls, conj):
        """ Outputs a HRR vector given a CNF formula """
        return cls.conj(
                [ cls.disj(
                    disj.role,
                    [ cls.eq(
                        eq.pos,
                        cls.fold_term(eq.t1),
                        cls.fold_term(eq.t2))
                      for eq in disj.eqs ])
                  for disj in conj.disjs ])

    @staticmethod
    def var(name):
        """ Output a HRR vector given a variable name """
        raise NotImplementedError

    @staticmethod
    def const(name):
        """ Output a HRR vector given a constant name """
        raise NotImplementedError

    @staticmethod
    def func(name, vecs):
        """ Output a HRR vector given function name and HRR vectors of its inputs """
        raise NotImplementedError

    @staticmethod
    def eq(pos, vec1, vec2):
        """ Output a HRR vector given positivity of this literal (literals are equations), and HRR vectors of the left and right sides """
        raise NotImplementedError

    @staticmethod
    def disj(role, vecs):
        """ Output a HRR vector given role of this clause, and HRR vectors of all the literals """
        raise NotImplementedError

    @staticmethod
    def conj(vecs):
        """ Output a HRR vector given HRR vectors of all the clauses """
        raise NotImplementedError

class FlatTreeHRR(HRR):
    @staticmethod
    def var(name):
        """ Output a HRR vector given a variable name """
        return get_ground_vector('Var:{}'.format(name))

    @staticmethod
    def const(name):
        """ Output a HRR vector given a constant name """
        return get_ground_vector('Const:{}'.format(name))

    @staticmethod
    def func(name, vecs):
        """ Output a HRR vector given function name and HRR vectors of its inputs """
        arity = len(vecs)
        funcobj = get_ground_vector('Func:{}:{}'.format(arity, name))

        argobjs = [
                associate(
                    get_ground_vector('Arg:{}:{}:{}'.format(arity, name, i)),
                    vec)
                for i, vec in enumerate(vecs) ]
        return merge([funcobj, merge(argobjs), merge(vecs)])

    @staticmethod
    def eq(pos, vec1, vec2):
        """ Output a HRR vector given positivity of this literal (literals are equations), and HRR vectors of the left and right sides """
        return merge([
            merge2(
                associate(get_ground_vector('Eq:{}:Left'.format(pos)), vec1),
                associate(get_ground_vector('Eq:{}:Right'.format(pos)), vec2)),
            vec1,
            vec2])

    @staticmethod
    def disj(role, vecs):
        """ Output a HRR vector given role of this clause, and HRR vectors of all the literals """
        roleobj = get_ground_vector('DisjRole:{}'.format(role))
        objs = [ associate(roleobj, vec) for vec in vecs ]
        merged = merge(objs)
        disjobj = associate(merged, merged) # Essentially associates each literal with each other
        return merge2(disjobj, merge(vecs))

    @staticmethod
    def conj(vecs):
        """ Output a HRR vector given HRR vectors of all the clauses """
        return merge(vecs)


# class TreeHRR(HRR):

# class FlatTreeHRR(HRR):

