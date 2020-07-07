import collections
import functools

# a term is either a Var, Const, or Func
Var = collections.namedtuple('Var', 'name')
Dist = collections.namedtuple('Dist', 'name')
Const = collections.namedtuple('Const', 'name')
Func = collections.namedtuple('Func', 'funcname, args')

# Eq relates two terms with equality or inequality
Eq = collections.namedtuple('Eq', 'pos, t1, t2')

# Disj is a list of Eqs
Disj = collections.namedtuple('Disj', 'name, role, eqs')

# Conj is a list of Disj
Conj = collections.namedtuple('Conj', 'disjs')

def fold_term(term, fvar, fdist, fconst, fskolem, ffunc, feq, fdisj, fconj):
    fvar, fdist, fconst, fskolem, ffunc, feq, fdisj, fconj = map(
            functools.lru_cache(maxsize=None),
            [fvar, fdist, fconst, fskolem, ffunc, feq, fdisj, fconj])

    def _fold_term(term):
        if isinstance(term, Var):
            return 1, fvar(term.name)
        elif isinstance(term, Dist):
            return 1, fdist(term.name)
        elif isinstance(term, Const):
            if term.name.startswith('esk'):
                return 1, fskolem(term.name)
            else:
                return 1, fconst(term.name)
        elif isinstance(term, Func):
            argsizes, arghrrs = zip(*[ _fold_term(x)
                      for x in term.args ])
            return sum(argsizes) + 1, ffunc(
                    term.funcname,
                    argsizes,
                    arghrrs)
        elif isinstance(term, Eq):
            leftsize, lefthrr = _fold_term(term.t1)
            rightsize, righthrr = _fold_term(term.t2)
            return leftsize + rightsize + 1, feq(
                    term.pos,
                    leftsize,
                    rightsize,
                    lefthrr,
                    righthrr)
        elif isinstance(term, Disj):
            eqsizes, eqhrrs = zip(*[ _fold_term(eq)
                      for eq in term.eqs ])
            return sum(eqsizes) + 1, fdisj(
                    eqsizes,
                    eqhrrs)
        elif isinstance(term, Conj):
            disjsizes, disjhrrs = zip(*[ _fold_term(disj)
                      for disj in term.disjs ])
            return sum(disjsizes) + 1, fconj(
                    disjsizes,
                    disjhrrs)
        raise RuntimeError('Invalid term {}'.format(term))

    _, resulthrr = _fold_term(term)

    map(
            lambda f: f.cache_clear(),
            [fvar, fdist, fconst, fskolem, ffunc, feq, fdisj, fconj])

    return resulthrr

