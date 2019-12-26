from utils import persist_to_file
from AST import Var, Const, Dist, Func, Eq, Disj, Conj

import torch
import torch.nn as nn
import numpy as np
import functools
import zlib

def normalize(vec):
    """ Normalize a vector """
    return vec / vec.norm()

def associate(x, y):
    """ Create association between two objects """
    x = torch.cat([x[1:], x])
    xx, yy = x.reshape(1,1,-1), y.flip(0).reshape(1,1,-1)
    zz = torch.nn.functional.conv1d(xx, yy)
    z = zz.reshape(-1)
    return normalize(z)

def decode(x, xy):
    """ Create association between two objects """
    return associate(torch.cat([x[0:1], x[-1:1:-1]]), xy)

def merge2(x, y, weight=0.5):
    """ Merge two associations together """
    z = normalize(weight * x + (1-weight) * y)
    return z

def merge(it):
    """ Merge a bag of associations together """
    s = it[0]
    for i in it[1:]:
        s = s + i
    s = normalize(s)
    return normalize(s)

class HRRTorch(nn.Module):
    """ Interface for HRR manipulations """

    __constants__ = ['hrr_size']

    def __init__(self, hrr_size):
        super(HRRTorch, self).__init__()

        self.hrr_size = hrr_size

        self._cache_clear_hook = self.register_backward_hook(self._cache_clear_hook_func)

    @staticmethod
    def _cache_clear_hook_func(model, _, __):
        model.cache_clear()

    def cache_clear(self):
        """
        Clear cached tensors

        We need to do this whenever we compute gradients,
        else these tensors which already have gradients
        will get reused (incorrectly)
        if the module is subsequently used

        This is done with a backward hook (see __init__)
        """
        self.fold_term.cache_clear()

    def forward(self, input):
        return self.fold_term(input)

    @functools.lru_cache()
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
                    tuple( self.fold_term(x)
                      for x in term.args ))
        elif isinstance(term, Eq):
            return self.encode_eq(term)
        elif isinstance(term, Disj):
            return self.encode_disj(term)
        elif isinstance(term, Conj):
            return self.encode_conj(term)
        raise RuntimeError('Invalid term {}'.format(term))

    def encode_eq(self, eq):
        """ Outputs a HRR vector given a literal """
        out = self.eq(
                eq.pos,
                self.fold_term(eq.t1),
                self.fold_term(eq.t2))
        return out

    def encode_disj(self, disj):
        """ Outputs a HRR vector given a disjunction of literals """
        return self.disj(
                disj.role,
                tuple( self.fold_term(eq)
                  for eq in disj.eqs ))

    def encode_conj(self, conj):
        """ Outputs a HRR vector given a CNF formula """
        return self.conj(
                tuple( self.fold_term(disj)
                  for disj in conj.disjs ))

    def get_ground_vector(self, label):
        """ Deterministically generate a random vector from its label """
        raise NotImplementedError

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

class FlatTreeHRRTorch(HRRTorch):

    def __init__(self, hrr_size):
        super(FlatTreeHRRTorch, self).__init__(hrr_size)

        fixed_encodings = dict()
        specifier_covariances = dict()
        ground_vec_merge_ratios = dict()

        specifier_dict = {
                '!Var': 1,
                '!Const': 1,
                '!Dist': 1,
                '!Func': 2,
                '!Arg': 3,
                '!Left': 1,
                '!Right': 1,
                '!DisjRole': 1,
                }

        for specifier, arity in specifier_dict.items():
            fixed_enc = nn.Parameter(torch.Tensor(self.hrr_size))
            fixed_encodings[specifier] = fixed_enc

            for i in range(arity):
                cov = nn.Parameter(torch.Tensor(self.hrr_size, self.hrr_size))
                specifier_covariances['cov_{}_{}'.format(specifier, i)] = cov

                merge_ratio = nn.Parameter(torch.Tensor(2))
                ground_vec_merge_ratios['ground_{}_{}'.format(specifier, i)] = merge_ratio

        self.fixed_encodings = nn.ParameterDict(fixed_encodings)
        self.specifier_covariances = nn.ParameterDict(specifier_covariances)
        self.ground_vec_merge_ratios = nn.ParameterDict(ground_vec_merge_ratios)


        self.func_weights = nn.Parameter(torch.Tensor(3))
        self.eq_weights = nn.Parameter(torch.Tensor(6))
        self.disj_weights = nn.Parameter(torch.Tensor(3))
        self.conj_weights = nn.Parameter(torch.Tensor(2))

        for p in self.fixed_encodings.values():
            nn.init.normal_(p)
            with torch.no_grad():
                p /= p.norm()
        for p in self.specifier_covariances.values():
            with torch.no_grad():
                torch.eye(*p.shape, out=p, requires_grad=True)
                p += torch.normal(0, 0.01, p.shape)
        for p in [
                *self.ground_vec_merge_ratios.values(),
                self.func_weights,
                self.eq_weights,
                self.disj_weights,
                self.conj_weights,
                ]:
            nn.init.uniform_(p)
            with torch.no_grad():
                p /= p.sum()

    def cache_clear(self):
        super(FlatTreeHRRTorch, self).cache_clear()

        self.get_ground_vector.cache_clear()
        self.var.cache_clear()
        self.const.cache_clear()
        self.dist.cache_clear()
        self.func.cache_clear()
        self.eq.cache_clear()
        self.disj.cache_clear()
        self.conj.cache_clear()

    @functools.lru_cache()
    def get_ground_vector(self, label):
        """ Deterministically generate a random vector from its label """

        if ':' in label:
            # This is an identifier

            parent, _, specifier = label.rpartition(':')
            top, _, _ = parent.partition(':')
            parentvec = self.get_ground_vector(parent)

            rs = np.random.RandomState(
                    zlib.adler32(
                        (str(self.hrr_size)+label).encode('utf-8')
                        ) & 0xffffffff)
            rs.randint(2)

            specifier_vec = normalize(
                    self.specifier_covariances['cov_{}_{}'.format(top, parent.count(':'))] @
                    torch.tensor(rs.standard_normal(self.hrr_size)).float())

            newvec = normalize(
                    self.ground_vec_merge_ratios['ground_{}_{}'.format(top, parent.count(':'))] @
                    torch.cat([
                        parentvec,
                        specifier_vec,
                        ]).reshape(-1, self.hrr_size))

            return newvec
        else:
            # Top level terms are fixed encodings

            return normalize(self.fixed_encodings[label])

    @functools.lru_cache()
    def var(self, name):
        """ Output a HRR vector given a variable name """
        return self.get_ground_vector('!Var:{}'.format(name))

    @functools.lru_cache()
    def const(self, name):
        """ Output a HRR vector given a constant name """
        return self.get_ground_vector('!Const:{}'.format(name))

    @functools.lru_cache()
    def dist(self, name):
        """ Output a HRR vector given a distinct object name """
        return self.get_ground_vector('!Dist:{}'.format(name))

    @functools.lru_cache()
    def func(self, name, vecs):
        """ Output a HRR vector given function name and HRR vectors of its inputs """
        arity = len(vecs)
        funcobj = self.get_ground_vector('!Func:{}:{}'.format(arity, name))

        argobjs = [
                associate(
                    self.get_ground_vector('!Arg:{}:{}:{}'.format(arity, name, i)),
                    vec)
                for i, vec in enumerate(vecs) ]

        result = normalize(
                self.func_weights @
                torch.cat([
                    funcobj,
                    merge(argobjs),
                    merge(vecs),
                    ]).reshape(-1, self.hrr_size))

        return result

    @functools.lru_cache()
    def eq(self, pos, vec1, vec2):
        """ Output a HRR vector given positivity of this literal (literals are equations), and HRR vectors of the left and right sides """
        leftobj = associate(self.get_ground_vector('!Left:{}'.format(pos)), vec1)
        rightobj = associate(self.get_ground_vector('!Right:{}'.format(pos)), vec2)
        result = normalize(
                self.eq_weights @
                torch.cat([
                    leftobj,
                    rightobj,
                    associate(leftobj, rightobj),
                    vec1,
                    vec2,
                    associate(vec2, vec2),
                    ]).reshape(-1, self.hrr_size))

        return result

    @functools.lru_cache()
    def disj(self, role, vecs):
        """ Output a HRR vector given role of this clause, and HRR vectors of all the literals """
        roleobj = self.get_ground_vector('!DisjRole:{}'.format(role))
        objs = [ associate(roleobj, vec) for vec in vecs ]
        mergedobjs = merge(objs)
        mergedvecs = merge(vecs)
        disjobj = associate(mergedobjs, mergedobjs) # Essentially associates each literal with each other
        return normalize(
                self.disj_weights @
                torch.cat([
                    disjobj,
                    mergedobjs,
                    mergedvecs,
                    ]).reshape(-1, self.hrr_size))

    @functools.lru_cache()
    def conj(self, vecs):
        """ Output a HRR vector given HRR vectors of all the clauses """
        mergedvecs = merge(vecs)
        conjobj = associate(mergedvecs, mergedvecs)
        return normalize(
                self.conj_weights @
                torch.cat([
                    conjobj,
                    mergedvecs,
                    ]).reshape(-1, self.hrr_size))

class FlatTreeHRRTorch2(FlatTreeHRRTorch):
    """
    Like FlatTreeHRRTorch but where the ground vectors for identity objects
    are created with an independent set of normal variables.
    """

    def __init__(self, hrr_size):
        super(FlatTreeHRRTorch2, self).__init__(hrr_size)

        specifier_dict = {
                '!Var': 1,
                '!Const': 1,
                '!Dist': 1,
                '!Func': 2,
                '!Arg': 3,
                '!Left': 1,
                '!Right': 1,
                '!DisjRole': 1,
                }

        specifier_variances = dict()

        for specifier, arity in specifier_dict.items():
            for i in range(arity):
                var = nn.Parameter(torch.Tensor(self.hrr_size))
                specifier_variances['var_{}_{}'.format(specifier, i)] = var

        self.specifier_variances = nn.ParameterDict(specifier_variances)

        for p in self.specifier_variances.values():
            nn.init.uniform_(p)

    @functools.lru_cache()
    def get_ground_vector(self, label):
        """ Deterministically generate a random vector from its label """

        if ':' in label:
            # This is an identifier

            parent, _, specifier = label.rpartition(':')
            top, _, _ = parent.partition(':')
            parentvec = self.get_ground_vector(parent)

            rs = np.random.RandomState(
                    zlib.adler32(
                        (str(self.hrr_size)+label).encode('utf-8')
                        ) & 0xffffffff)
            rs.randint(2)

            specifier_vec = normalize(
                    self.specifier_variances['var_{}_{}'.format(top, parent.count(':'))] *
                    torch.tensor(rs.standard_normal(self.hrr_size)).float())

            newvec = normalize(
                    self.ground_vec_merge_ratios['ground_{}_{}'.format(top, parent.count(':'))] @
                    torch.cat([
                        parentvec,
                        specifier_vec,
                        ]).reshape(-1, self.hrr_size))

            return newvec
        else:
            # Top level terms are fixed encodings

            return normalize(self.fixed_encodings[label])

# class TreeHRR(HRR):

# class FlatTreeHRR(HRR):

