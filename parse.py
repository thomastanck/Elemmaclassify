import parsy as p

def symbol(s):
    return p.whitespace.many() >> p.string(s) << p.whitespace.many()

alpha_numeric = p.regex('[a-zA-Z0-9_]')

lower_alpha = p.regex('[a-z]')
upper_alpha = p.regex('[A-Z]')

lower_word = p.seq(lower_alpha, alpha_numeric.many().concat()).concat()
upper_word = p.seq(upper_alpha, alpha_numeric.many().concat()).concat()

integer = p.seq(
            p.string_from(['+', '-']).optional(),
            p.decimal_digit.at_least(1).concat().map(int)
        ).combine(lambda sign, number: -number if sign == '-' else number)

sq_char = p.regex('[a-zA-Z0-9 _\\-/~!@#$%^&*(),."]')
single_quote = p.string("'")
single_quoted = single_quote >> sq_char.at_least(1).concat() << single_quote

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

constant = lower_word | p.string('$') + lower_word | p.string('$$') + lower_word
variable = upper_word

@p.generate
def fof_term():
    v = yield variable.optional()
    if v:
        return v
    c = yield constant
    if not (yield symbol('(').optional()):
        return c
    args = yield function_args
    yield symbol(')')
    return [c, *args]

function_args = fof_term.sep_by(symbol(','))

fof_neg_eqn = p.seq(fof_term, symbol('!='), fof_term).combine(lambda a, b, c: (b, a, c))
fof_eqn     = p.seq(fof_term, symbol('='), fof_term).combine(lambda a, b, c: (b, a, c))

literal = (p.seq(symbol('~').optional(), fof_eqn).combine(lambda a, b: b if a is None else (a, b)) |
           fof_neg_eqn |
           fof_term.map(lambda t: ('=', t, '$true')))

disjunction = literal.sep_by(symbol('|'))

cnf_formula = disjunction | symbol('(') >> disjunction << symbol(')')

cnf_annotated = (symbol('cnf') >> symbol('(') >>
    p.seq(name, symbol(',') >> formula_role, symbol(',') >> cnf_formula)
    << symbol(')') << symbol('.'))
