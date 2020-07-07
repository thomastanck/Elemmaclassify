import collections
import functools
import zlib

import numpy
import scipy.stats

from AST import Var, Dist, Const, Func, Eq, Disj, Conj, fold_term
from CircularHRR import CircularHRR

CNFtoHRR1Settings = collections.namedtuple(
        'CNFtoHRR1Settings',
        '''
        var_base_weight
        dist_base_weight
        const_base_weight
        skolem_base_weight
        func_base_weight
        func_arity_weight
        arg_base_weight
        arg_arity_weight
        arg_index_weight
        superposition_func_funcobj_weight
        superposition_func_argobjs_weight
        superposition_func_copy_weight
        superposition_func_beta1
        superposition_func_beta2
        superposition_func_equalize_size_weight
        superposition_eq_pos_weight
        superposition_eq_assoc_weight
        superposition_eq_equalize_size_weight
        superposition_disj_assoc_weight
        superposition_disj_equalize_size_weight
        superposition_conj_assoc_weight
        superposition_conj_equalize_size_weight
        '''
        )

defaultCNFtoHRR1Settings = CNFtoHRR1Settings(
            var_base_weight                         = 0.5,
            dist_base_weight                        = 0.5,
            const_base_weight                       = 0.5,
            skolem_base_weight                      = 0.5,
            func_base_weight                        = 0.5,
            func_arity_weight                       = 0.25,
            arg_base_weight                         = 0.5,
            arg_arity_weight                        = 0.25,
            arg_index_weight                        = 0.125,
            superposition_func_funcobj_weight       = 0.2,
            superposition_func_argobjs_weight       = 0.4,
            superposition_func_copy_weight          = 0.4,
            superposition_func_beta1                = 1,
            superposition_func_beta2                = 1,
            superposition_func_equalize_size_weight = 0.5,
            superposition_eq_pos_weight             = 0.5,
            superposition_eq_assoc_weight           = 0.5,
            superposition_eq_equalize_size_weight   = 1,
            superposition_disj_assoc_weight         = 0.5,
            superposition_disj_equalize_size_weight = 1,
            superposition_conj_assoc_weight         = 0.5,
            superposition_conj_equalize_size_weight = 1,
            )

def randomCNFtoHRR1Settings_v1(seed):
    rs = numpy.random.RandomState(seed)

    var_base_weight                         = rs.uniform()
    dist_base_weight                        = rs.uniform()
    const_base_weight                       = rs.uniform()
    skolem_base_weight                      = rs.uniform()
    func_base_weight                        = rs.uniform()
    func_arity_weight                       = rs.uniform(high=1 - func_base_weight)
    arg_base_weight                         = rs.uniform()
    arg_arity_weight                        = rs.uniform(high=1 - arg_base_weight)
    arg_index_weight                        = rs.uniform(high=1 - arg_base_weight - arg_arity_weight)

    superposition_func_weights = rs.uniform(size=3)
    superposition_func_weights /= sum(superposition_func_weights)
    (superposition_func_funcobj_weight,
            superposition_func_argobjs_weight,
            superposition_func_copy_weight) = superposition_func_weights

    superposition_func_beta1                = rs.exponential()
    superposition_func_beta2                = rs.exponential()
    superposition_func_equalize_size_weight = rs.uniform()

    superposition_eq_weights = rs.uniform(size=2)
    superposition_eq_weights /= sum(superposition_eq_weights)
    superposition_eq_pos_weight             = rs.uniform()
    superposition_eq_assoc_weight           = rs.uniform()
    superposition_eq_equalize_size_weight   = rs.uniform()
    superposition_disj_assoc_weight         = rs.uniform()
    superposition_disj_equalize_size_weight = rs.uniform()
    superposition_conj_assoc_weight         = rs.uniform()
    superposition_conj_equalize_size_weight = rs.uniform()

    return CNFtoHRR1Settings(
            var_base_weight,
            dist_base_weight,
            const_base_weight,
            skolem_base_weight,
            func_base_weight,
            func_arity_weight,
            arg_base_weight,
            arg_arity_weight,
            arg_index_weight,
            superposition_func_funcobj_weight,
            superposition_func_argobjs_weight,
            superposition_func_copy_weight,
            superposition_func_beta1,
            superposition_func_beta2,
            superposition_func_equalize_size_weight,
            superposition_eq_pos_weight,
            superposition_eq_assoc_weight,
            superposition_eq_equalize_size_weight,
            superposition_disj_assoc_weight,
            superposition_disj_equalize_size_weight,
            superposition_conj_assoc_weight,
            superposition_conj_equalize_size_weight,
            )

class CNFtoHRR1:
    def __init__(self, hrr_size, settings):
        self.hrr_size = hrr_size
        self.settings = settings

        # prefix with ! so that these vectors never occur in normal cnf input
        self.var_base    = self.get_ground_vector('!Var_base')
        self.dist_base   = self.get_ground_vector('!Dist_base')
        self.const_base  = self.get_ground_vector('!Const_base')
        self.skolem_base = self.get_ground_vector('!Skolem_base')
        self.func_base   = self.get_ground_vector('!Func_base')
        self.arg_base    = self.get_ground_vector('!Arg_base')

        self.eq_assoc    = self.get_ground_vector('!Eq_assoc')
        self.disj_assoc    = self.get_ground_vector('!Disj_assoc')
        self.conj_assoc    = self.get_ground_vector('!Conj_assoc')

    def var_id(self, name):
        return self.get_ground_vector('!Var_id_{}'.format(name))
    def dist_id(self, name):
        return self.get_ground_vector('!Dist_id_{}'.format(name))
    def const_id(self, name):
        return self.get_ground_vector('!Const_id_{}'.format(name))
    def skolem_id(self, name):
        return self.get_ground_vector('!Skolem_id_{}'.format(name))
    def func_arity(self, arity):
        return self.get_ground_vector('!Func_arity_{}'.format(arity))
    def func_id(self, name):
        return self.get_ground_vector('!Func_id_{}'.format(name))
    def arg_arity(self, arity):
        return self.get_ground_vector('!Arg_arity_{}'.format(arity))
    def arg_index(self, arity, index):
        return self.get_ground_vector('!Arg_index_{}_{}'.format(arity, index))
    def arg_id(self, name):
        return self.get_ground_vector('!Arg_id_{}'.format(name))

    @functools.lru_cache()
    def get_ground_vector(self, label):
        """ Deterministically generate a random vector from its label """

        seed = zlib.adler32(
                (str(self.hrr_size)+label).encode('utf-8')
                ) & 0xffffffff

        return CircularHRR(self.hrr_size, seed)


    def fold_term(self, term):
        return fold_term(
                term,
                self.fvar,
                self.fdist,
                self.fconst,
                self.fskolem,
                self.ffunc,
                self.feq,
                self.fdisj,
                self.fconj)

    def fvar(self, name):
        ''' Creates HRR for variable node.

        Relevant settings:
        var_base_weight
        '''
        return (self.settings.var_base_weight * self.var_base +
                (1 - self.settings.var_base_weight) * self.var_id(name)
                ).normalize()

    def fdist(self, name):
        ''' Creates HRR for distinct constant node.

        Relevant settings:
        dist_base_weight
        '''
        return (self.settings.dist_base_weight * self.dist_base +
                (1 - self.settings.dist_base_weight) * self.dist_id(name)
                ).normalize()

    def fconst(self, name):
        ''' Creates HRR for constant node.

        Relevant settings:
        const_base_weight
        '''
        return (self.settings.const_base_weight * self.const_base +
                (1 - self.settings.const_base_weight) * self.const_id(name)
                ).normalize()

    def fskolem(self, name):
        ''' Creates HRR for skolem constant node.

        Relevant settings:
        skolem_base_weight
        '''
        return (self.settings.skolem_base_weight * self.skolem_base +
                (1 - self.settings.skolem_base_weight) * self.skolem_id(name)
                ).normalize()

    def ffunc(self, name, argsizes, arghrrs):
        ''' Creates HRR for functor node.

        Relevant settings:
        func_base_weight
        func_arity_weight
        arg_base_weight
        arg_arity_weight
        arg_index_weight
        superposition_func_funcobj_weight
        superposition_func_argobjs_weight
        superposition_func_copy_weight
        superposition_func_beta1
        superposition_func_beta2
        superposition_func_equalize_size_weight # 1 means all branches are weighted by size, 0 means branches are equally weighted
        '''
        arity = len(argsizes)
        funcobj = self.settings.superposition_func_funcobj_weight * (
                self.settings.func_base_weight * self.func_base +
                self.settings.func_arity_weight * self.func_arity(arity) +
                (1 - self.settings.func_base_weight - self.settings.func_arity_weight) * self.func_id(name)
                ).normalize()

        bias = scipy.stats.betabinom(arity - 1, self.settings.superposition_func_beta1, self.settings.superposition_func_beta2).pmf

        totalsize = sum(argsizes)

        argobjs = self.settings.superposition_func_argobjs_weight * sum(
                (self.settings.superposition_func_equalize_size_weight * size / totalsize +
                    (1 - self.settings.superposition_func_equalize_size_weight) * 1 / arity) *
                bias(i) *
                (self.settings.arg_base_weight * self.arg_base +
                    self.settings.arg_arity_weight * self.arg_arity(arity) +
                    self.settings.arg_index_weight * self.arg_index(arity, i) +
                    (1 - self.settings.arg_base_weight - self.settings.arg_arity_weight - self.settings.arg_index_weight) * self.arg_id(name)
                    ) @
                vec
                for i, (size, vec) in enumerate(zip(argsizes, arghrrs)) )

        copyobjs = self.settings.superposition_func_copy_weight * sum(
                (self.settings.superposition_func_equalize_size_weight * size / totalsize +
                    (1 - self.settings.superposition_func_equalize_size_weight) * 1 / arity) *
                bias(i) *
                vec
                for i, (size, vec) in enumerate(zip(argsizes, arghrrs)) )

        return (funcobj + argobjs + copyobjs).normalize()

    def feq(self, pos, leftsize, rightsize, lefthrr, righthrr):
        ''' Creates HRR for equality node.

        Relevant settings:
        superposition_eq_pos_weight
        superposition_eq_assoc_weight
        superposition_eq_equalize_size_weight
        '''
        magnitude = self.settings.superposition_eq_pos_weight
        if not pos:
            magnitude = 1 - magnitude

        totalsize = leftsize + rightsize

        assoc = self.settings.superposition_eq_assoc_weight * self.eq_assoc @ (lefthrr * righthrr)

        copy = (1 - self.settings.superposition_eq_assoc_weight) * sum(
                (self.settings.superposition_eq_equalize_size_weight * size / totalsize +
                    (1 - self.settings.superposition_eq_equalize_size_weight) * 1 / 2) *
                vec
                for size, vec in ((leftsize, lefthrr), (rightsize, righthrr)))

        return magnitude * (assoc + copy).normalize()

    def fdisj(self, eqsizes, eqhrrs):
        ''' Creates HRR for disj node.

        Relevant settings:
        superposition_disj_assoc_weight
        superposition_disj_equalize_size_weight
        '''
        arity = len(eqsizes)
        totalsize = sum(eqsizes)

        copy = sum(
                (self.settings.superposition_disj_equalize_size_weight * size / totalsize +
                    (1 - self.settings.superposition_disj_equalize_size_weight) * 1 / arity) *
                vec
                for size, vec in zip(eqsizes, eqhrrs))
        assoc = self.disj_assoc @ (copy * copy)

        assoc *= self.settings.superposition_disj_assoc_weight
        copy *= 1 - self.settings.superposition_disj_assoc_weight

        return (assoc + copy).normalize()

    def fconj(self, disjsizes, disjhrrs):
        ''' Creates HRR for variable node.

        Relevant settings:
        superposition_conj_assoc_weight
        superposition_conj_equalize_size_weight
        '''
        arity = len(disjsizes)
        totalsize = sum(disjsizes)

        copy = sum(
                (self.settings.superposition_conj_equalize_size_weight * size / totalsize +
                    (1 - self.settings.superposition_conj_equalize_size_weight) * 1 / arity) *
                vec
                for size, vec in zip(disjsizes, disjhrrs))
        assoc = self.conj_assoc @ (copy * copy)

        assoc *= self.settings.superposition_conj_assoc_weight
        copy *= 1 - self.settings.superposition_conj_assoc_weight

        return (assoc + copy).normalize()

