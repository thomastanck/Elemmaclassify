import collections

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

