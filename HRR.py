from utils import persist_to_file
from AST import Var, Const, Dist, Func, Eq, Disj, Conj
import numpy as np
import functools
import zlib

def normalize(vec):
    return vec / np.linalg.norm(vec)

def normalize_comp(vec):
    return vec / np.linalg.norm(vec)

def associate(x, y):
    """ Create association between two objects """
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    z = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
    return normalize(z)

def associate_comp(x, y):
    """ Create association between two complex objects """
    return normalize_comp(x * y)

def involution(x):
    """ Flips array from second element onwards """
    return np.concatenate((x[0:1], x[-1:0:-1]))

def involution_comp(x):
    """ Equivalent to involution for complex HRRs """
    return x.conj()

def decode(x, xy):
    """ (Attempts to) Decode association of two objects """
    return associate(involution(x), xy)

def decode(x, xy):
    """ (Attempts to) Decode association of two complex objects """
    return associate_comp(involution_comp(x), xy)

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

def merge2_comp(x, y, weight=0.5):
    """ Merge two complex associations together """
    z = normalize_comp(weight * x + (1-weight) * y)
    return normalize_comp(weight * x + (1-weight) * y)

def merge_comp(it):
    """ Merge a bag of associations together """
    s = it[0].copy()
    for i in it[1:]:
        s += i
    s = normalize_comp(s)
    return s

class HRR:
    """ Interface for HRR manipulations """

    def __init__(
            self,
            hrr_size,
            complex_hrr=False,
            specifier_decay=0.5,
            randomise_variables=False,
            randomise_skolem_constants=False,
            randomise_theory_constants=False,
            randomise_distinct_objects=False,
            ):
        """
        hrr_size:                   Dimensionality of the HRR
        complex_hrr:                Whether to use phase HRRs (True) or real HRRs (False)
        specifier_decay:            The weightage of higher level specifiers (higher level = to the left) when generating ground vectors.
                                    0 means only the last specifier matters (all the way to the right)
                                    1 means only the top level specifier matters (all the way to the left)
        randomise_variables:        Whether variables should get different HRRs after cache_clear calls. Between cache_clear calls, variables will get consistent HRRs.
        randomise_skolem_constants: See randomise_variables. This option affects skolem constants instead of variables.
        randomise_theory_constants: See randomise_variables. This option affects theory constants instead of variables.
        randomise_distinct_objects: See randomise_variables. This option affects distinct objects instead of variables.
        """

        self.hrr_size = hrr_size
        self.complex_hrr = complex_hrr
        self.specifier_decay = specifier_decay
        self.randomise_variables = randomise_variables
        self.randomise_skolem_constants = randomise_skolem_constants
        self.randomise_theory_constants = randomise_theory_constants
        self.randomise_distinct_objects = randomise_distinct_objects

        # Used for generating completely randomised ids
        self.counter = 0

    def cache_clear(self):
        self.get_ground_vector.cache_clear()
        self.fold_term.cache_clear()

    @functools.lru_cache(maxsize=None)
    def get_ground_vector(self, label, is_identifier=True):
        """ Deterministically generate a random vector from its label """
        rs = np.random.RandomState(
                zlib.adler32(
                    (str(self.hrr_size)+label).encode('utf-8')
                    ) & 0xffffffff)
        rs.randint(2)

        if ':' in label:
            parent, _, specifier = label.rpartition(':')
        else:
            parent = None
            specifier = label

        if is_identifier:
            # Check if we should randomise this identifier.
            # If so, then we reseed the random state with the counter so it no longer depends on the name,
            # and every seed is distinct (even after cache_clear)
            if specifier[0].isupper():
                if self.randomise_variables:
                    rs.seed(self.counter)
                    self.counter += 1
            elif specifier.startswith('esk'):
                if self.randomise_skolem_constants:
                    rs.seed(self.counter)
                    self.counter += 1
            elif specifier.startswith('"') or specifier[0].isdigit():
                if self.randomise_distinct_objects:
                    rs.seed(self.counter)
                    self.counter += 1
            else: # not matching any of the above means it's a theory constant
                if self.randomise_theory_constants:
                    rs.seed(self.counter)
                    self.counter += 1

        if parent is not None:
            parentvec = self.get_ground_vector(parent, is_identifier=False)
            if self.complex_hrr:
                specifiervec = normalize(rs.standard_normal(self.hrr_size) + rs.standard_normal(self.hrr_size) * 1j)
            else:
                specifiervec = normalize(rs.standard_normal(self.hrr_size))
            newvec = merge2(parentvec, specifiervec, weight=self.specifier_decay)
            return newvec
        else:
            return normalize(rs.standard_normal(self.hrr_size))

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
        elif isinstance(term, Eq):
            return self.eq(
                term.pos,
                self.fold_term(term.t1),
                self.fold_term(term.t2))
        elif isinstance(term, Disj):
            return self.disj(
                    term.role,
                    [ self.fold_term(eq)
                      for eq in term.eqs ])
        elif isinstance(term, Conj):
            return self.conj(
                    [ self.fold_term(disj)
                      for disj in term.disjs ])

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
    def __init__(
            self,
            hrr_size,
            complex_hrr=False,
            specifier_decay=0.5,
            randomise_variables=False,
            randomise_skolem_constants=False,
            randomise_theory_constants=False,
            randomise_distinct_objects=False,
            recursive_role=True,
            ):
        """
        hrr_size:                   Dimensionality of the HRR
        complex_hrr:                Whether to use phase HRRs (True) or real HRRs (False)
        specifier_decay:            The weightage of higher level specifiers (higher level = to the left) when generating ground vectors.
                                    0 means only the last specifier matters (all the way to the right)
                                    1 means only the top level specifier matters (all the way to the left)
                                    (Note: setting this to 1 makes more sense if recursive_role is set to False, else Arg:0:2 and Arg:1:2 will get the same HRR, despite semantically being different)
        randomise_variables:        Whether variables should get different HRRs after cache_clear calls. Between cache_clear calls, variables will get consistent HRRs.
        randomise_skolem_constants: See randomise_variables. This option affects skolem constants instead of variables.
        randomise_theory_constants: See randomise_variables. This option affects theory constants instead of variables.
        randomise_distinct_objects: See randomise_variables. This option affects distinct objects instead of variables.
        recursive_role:             Whether role specifiers should be built recursively (so all the Arg:x:y are similar), or flat (so Arg:x:y is completely separate from one another)
        """
        super(FlatTreeHRR, self).__init__(
                hrr_size,
                complex_hrr,
                specifier_decay,
                randomise_variables,
                randomise_skolem_constants,
                randomise_theory_constants,
                randomise_distinct_objects,
                )
        self.recursive_role = recursive_role

    def var(self, name):
        """ Output a HRR vector given a variable name """
        return self.get_ground_vector('Var:{}-Var'.format(name))

    def const(self, name):
        """ Output a HRR vector given a constant name """
        return self.get_ground_vector('Const:{}-Const'.format(name))

    def dist(self, name):
        """ Output a HRR vector given a distinct object name """
        return self.get_ground_vector('Dist:{}-Dist'.format(name))

    def func(self, name, vecs):
        """ Output a HRR vector given function name and HRR vectors of its inputs """
        arity = len(vecs)
        funcobj = self.get_ground_vector(
                ('Func:{}:{}-Func-{}' if self.recursive_role else 'Func-{}:{}-Func-{}')
                .format(arity, name, arity))

        if self.complex_hrr:
            argobjs = [
                    associate_comp(
                        self.get_ground_vector(
                            ('Arg:{}:{}:{}-Arg-{}-{}' if self.recursive_role else 'Arg-{}-{}:{}-Arg-{}-{}')
                            .format(arity, i, name, arity, i)),
                        vec)
                    for i, vec in enumerate(vecs) ]
            result = merge_comp([funcobj, merge(argobjs), merge(vecs)])
        else:
            argobjs = [
                    associate(
                        self.get_ground_vector(
                            ('Arg:{}:{}:{}-Arg-{}-{}' if self.recursive_role else 'Arg-{}-{}:{}-Arg-{}-{}')
                            .format(arity, i, name, arity, i)),
                        vec)
                    for i, vec in enumerate(vecs) ]
            result = merge([funcobj, merge(argobjs), merge(vecs)])
        return result

    def eq(self, pos, vec1, vec2):
        """ Output a HRR vector given positivity of this literal (literals are equations), and HRR vectors of the left and right sides """
        if self.complex_hrr:
            return merge_comp([
                merge2_comp(
                    associate_comp(self.get_ground_vector('Eq:{}:Left'.format(pos), is_identifier=False), vec1),
                    associate_comp(self.get_ground_vector('Eq:{}:Right'.format(pos), is_identifier=False), vec2)),
                vec1,
                vec2])
        else:
            return merge([
                merge2(
                    associate(self.get_ground_vector('Eq:{}:Left'.format(pos), is_identifier=False), vec1),
                    associate(self.get_ground_vector('Eq:{}:Right'.format(pos), is_identifier=False), vec2)),
                vec1,
                vec2])

    def disj(self, role, vecs):
        """ Output a HRR vector given role of this clause, and HRR vectors of all the literals """
        roleobj = self.get_ground_vector('DisjRole:{}'.format(role), is_identifier=False)
        if self.complex_hrr:
            objs = [ associate_comp(roleobj, vec) for vec in vecs ]
            merged = merge_comp(objs)
            disjobj = associate_comp(merged, merged) # Essentially associates each literal with each other
            return merge2_comp(disjobj, merge_comp(vecs))

        else:
            objs = [ associate(roleobj, vec) for vec in vecs ]
            merged = merge(objs)
            disjobj = associate(merged, merged) # Essentially associates each literal with each other
            return merge2(disjobj, merge(vecs))

    def conj(self, vecs):
        """ Output a HRR vector given HRR vectors of all the clauses """
        if self.complex_hrr:
            return merge_comp(vecs)
        else:
            return merge(vecs)

# class TreeHRR(HRR):

# class FlatTreeHRR(HRR):

