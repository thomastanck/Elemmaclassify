import parsy as p
import collections

from AST import Var, Dist, Const, Func, Eq, Disj, Conj

def symbol(s):
    return p.whitespace.many() >> p.string(s) << p.whitespace.many()

def parenthesised(parser):
    @p.generate
    def a():
        # Handle parenthesised terms
        if (yield symbol('(').optional()):
            t = yield a
            yield symbol(')')
            return t
        else:
            return (yield parser)
    return a

alpha_numeric = p.regex('[a-zA-Z0-9_]')

lower_alpha = p.regex('[a-z]')
upper_alpha = p.regex('[A-Z]')

lower_word = p.seq(lower_alpha, alpha_numeric.many().concat()).concat()
upper_word = p.seq(upper_alpha, alpha_numeric.many().concat()).concat()

integer = p.seq(
            p.string_from('+', '-').optional(),
            p.decimal_digit.at_least(1).concat().map(int)
        ).combine(lambda sign, number: -number if sign == '-' else number)

sq_char = p.regex('[a-zA-Z0-9 _\\-/~!@#$%^&*(),."]')
single_quote = p.string("'")
single_quoted = single_quote >> sq_char.at_least(1).concat() << single_quote

dq_char = p.regex("[a-zA-Z0-9 _\\-/~!@#$%^&*(),.']")
double_quote = p.string('"')
double_quoted = double_quote >> dq_char.at_least(1).concat() << double_quote

atomic_word = lower_word | single_quoted

name = atomic_word | integer

formula_role = p.string_from(
    'axiom',
    'hypothesis',
    'definition',
    'assumption',
    'lemma',
    'theorem',
    'corollary',
    'conjecture',
    'negated_conjecture',
    'plain',
    'type',
    'fi_domain',
    'fi_functors',
    'fi_predicates',
    'unknown')

constant = (lower_word | p.string('$') + lower_word | p.string('$$') + lower_word).map(Const)
variable = upper_word.map(Var)
distinct = (integer.map(repr) | double_quoted.map(repr)).map(Dist)

@p.generate
def fof_term():
    v = yield variable.optional()
    if v:
        return v

    c = yield constant | distinct

    if not (yield symbol('(').optional()):
        return c

    args = yield function_args
    yield symbol(')')
    return Func(c.name, args)

fof_term = parenthesised(fof_term)

function_args = fof_term.sep_by(symbol(',')).map(tuple)

@p.generate
def literal():
    n = yield symbol('~').optional()
    t = yield fof_term
    eq = yield (symbol('=')|symbol('!=')).optional()
    if eq is None:
        if n:
            return Eq(False, t, Const('$true'))
        else:
            return Eq(True, t, Const('$true'))
    else:
        pos = n == (eq == '!=')
        t2 = yield fof_term
        return Eq(pos, t, t2)
literal = parenthesised(literal)

disjunction = literal.sep_by(symbol('|')).map(tuple)

cnf_formula = parenthesised(disjunction)

cnf_annotated = (symbol('cnf') >> symbol('(') >>
    p.seq(name, symbol(',') >> formula_role, symbol(',') >> cnf_formula).combine(Disj)
    << symbol(')') << symbol('.'))

cnf_list = cnf_annotated.many().map(tuple).map(Conj)

def parse_cnf_clause(s):
    return cnf_annotated.parse(s)

def parse_cnf_list(s):
    # Filter out comments
    s = '\n'.join(l for l in s.split('\n') if not l.startswith('#') and l)
    return cnf_list.parse(s)

def parse_cnf_file(filename):
    with open(filename, 'r') as f:
        return parse_cnf_list(f.read())

