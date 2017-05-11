import pickle

from functools import partial
from itertools import chain
from collections import Counter
from collections import namedtuple
from collections import defaultdict
import json

from lark import Lark
from lark import InlineTransformer
from lark.lexer import Token
from lark.common import ParseError

import numpy as np

Node = namedtuple('Node', ['label', 'left', 'right', 'unit'])

parser_op_mapping = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'log': 'log',
    'exp': 'exp',
    'neg': '-'
}
parser = Lark(
    '''
    ?sum: product
         | sum "+" product   -> add
         | sum "-" product   -> sub

     ?product: item
         | product "*" item  -> mul
         | product "/" item  -> div

     ?item: NUMBER
          | NAME      
          | "-" item         -> neg
          | "cos" "(" sum ")"        -> cos
          | "sin" "(" sum ")"       -> sin
          | "tan" "(" sum ")"       -> tan
          | "exp" "(" sum ")"       -> exp
          | "log" "(" sum ")"       -> log
          | "(" sum ")"
    
     %import common.NUMBER
     %import common.WS
     %ignore WS
     %import common.CNAME -> NAME
''', start='sum')


class Unit(defaultdict):
    
    def __init__(self, d=None):
        if d is None:
            d = {}
        if (d) == int:
            d = {}
        super().__init__(int, d)

    def __add__(self, c):
        for k in self.keys():
            self[k] += c

    def __mul__(self, v):
        u = self
        out = Unit()
        for k in set(u.keys()) | set(v.keys()):
            out[k] = u[k] + v[k]
        return out

    def __floordiv__(self, v):
        u = self
        out = Unit()
        for k in set(u.keys()) | set(v.keys()):
            out[k] = u[k] - v[k]
        return out

    def __eq__(self, v):
        if not isinstance(v, Unit):
            return False
        u = self
        for k in set(u.keys()) | set(v.keys()):
            if u[k] != v[k]:
                return False
        return True
    
    def __repr__(self):
        return ' '.join(['{}:{}'.format(k, v) for k, v in self.items()])


def gen_formula_tree(symbols, units, min_depth=1, max_depth=2, unit_constraints=True, random_state=None):
    rng = np.random.RandomState(random_state)
    def _gen(rng, depth=0, force_unit=None):
        if depth >= max_depth:
            choices = ('cst', 'symbol')
        elif depth <= min_depth:
            choices = ('op',)
        else:
            choices = ('op', 'cst', 'symbol', 'unary_op')
        if unit_constraints and (force_unit is not None and force_unit != constant_unit):
            choices = tuple(c for c in choices if c != 'cst')
        action = rng.choice(choices)
        if action == 'op':
            op = rng.choice(('+', '-', '*', '/'))
            if op in ('+', '-'):
                if unit_constraints and (force_unit is not None and unit_constraints):
                    lnode = _gen(rng, depth=depth + 1, force_unit=force_unit)
                    rnode = _gen(rng, depth=depth + 1, force_unit=force_unit)
                    unit = force_unit
                else:
                    lnode = _gen(rng, depth=depth + 1)
                    rnode = _gen(rng, depth=depth + 1, force_unit=lnode.unit)
                    unit = lnode.unit
                if unit_constraints:
                    assert (lnode.unit == rnode.unit), (lnode.unit, rnode.unit, as_str(lnode), as_str(rnode), force_unit)
                return Node(label=op, unit=unit, left=lnode, right=rnode)
            elif op in ('*', '/'):
                if unit_constraints and force_unit is not None:
                    lnode = _gen(rng, depth=depth + 1)
                    if op == '*':
                        unit = force_unit // lnode.unit
                    elif op == '/':
                        unit = lnode.unit // force_unit
                    rnode = _gen(rng, depth=depth + 1, force_unit=unit)
                    assert rnode.unit == unit
                    if op == '*':
                        assert ((lnode.unit * rnode.unit) == force_unit)
                    elif op == '/':
                        assert (lnode.unit // rnode.unit) == force_unit, (lnode.unit, rnode.unit, (lnode.unit//rnode.unit), force_unit, as_str(lnode), as_str(rnode))
                    unit = force_unit
                else:
                    lnode = _gen(rng, depth=depth + 1)
                    rnode = _gen(rng, depth=depth + 1)
                    if op == '*':
                        unit = lnode.unit * rnode.unit
                    elif op == '/':
                        unit = lnode.unit // rnode.unit
            return Node(label=op, unit=unit, left=lnode, right=rnode)
        elif action == 'unary_op':
            choices = ('-', 'exp', 'sin', 'cos', 'tan', 'log')
            if unit_constraints  and (force_unit is not None and force_unit != constant_unit):
                choices = ('-',)
                unit = force_unit
            else:
                unit = constant_unit
            op = rng.choice(choices)
            left = _gen(rng, depth=depth + 1, force_unit=unit)
            if unit_constraints:
                assert unit == left.unit, (op, left.unit, unit, as_str(left))
            return Node(label=op, left=left, right=None, unit=unit)
        elif action == 'symbol':
            if unit_constraints and force_unit is not None:
                symbols_ = list(filter(lambda s:units[s] == force_unit, symbols))
                if len(symbols_) == 0:
                    raise ValueError('no symbols to choose from')
            else:
                symbols_ = symbols
            s = rng.choice(symbols_)
            unit = units[s]
            return Node(label=s, left=None, right=None, unit=unit)
        elif action == 'cst':
            val = _gen_constant()
            return Node(label=val, left=None, right=None, unit=constant_unit)

    def _gen_constant():
        return rng.randint(1, 10)/100.

    return _gen(rng)


constant_unit = Unit() 

def check_constraints(node):
    if node.left and node.right:
        if node.label in ('+', '-'):
            assert node.left.unit == node.right.unit
        elif node.label == '*':
            assert node.unit == node.left.unit * node.right.unit
        elif node.label == '/':
            assert node.unit == node.left.unit // node.right.unit
        check_constraints(node.left)
        check_constraints(node.right)
    elif node.left:
        if node.label != '-':
            assert node.left.unit == constant_unit
        assert node.unit == node.left.unit
        check_constraints(node.left)

def as_tree(s, units):
    t = parser.parse(s)
    def _as_tree(t):
        if isinstance(t, Token):
            if t in units.keys():
                unit = units[t]
            else:
                unit = constant_unit
            return Node(label=t.value, left=None, right=None, unit=unit)
        elif len(t.children) == 2:
            left, right = t.children
            left = _as_tree(left)
            right = _as_tree(right)
            op = t.data
            if op in ('add', 'sub'):
                unit = left.unit
            elif op == 'mul':
                unit = left.unit * right.unit
            elif op == 'div':
                unit = left.unit // right.unit
            op = parser_op_mapping[op]
            return Node(label=op, left=left, right=right, unit=unit)
        elif len(t.children) == 1:
            left, = t.children
            left = _as_tree(left)
            unit = left.unit
            op = t.data
            op = parser_op_mapping[op]
            return Node(label=op, left=left, right=None, unit=unit)
        else:
            raise ValueError('nb of children must be 1 or 2')
    return _as_tree(t)

def get_symbols(t):
    if t.left and t.right:
        return get_symbols(t.left) | get_symbols(t.right)
    elif t.left:
        return get_symbols(t.left)
    else:
        return set([t.label])


def as_theano(s, symbols):
    import theano.tensor as T
    import theano
    vars = {s: T.fvector(name=s) for s in symbols}
    for sym in symbols:
        locals()[sym] = vars[sym]
    ops = ['sin', 'cos', 'tan', 'log', 'exp']
    for op in ops:
        s = s.replace(op, 'T.' + op)
    s = 'vars["result"] = ' + s
    exec(s)
    return vars


def evaluate(s, symbol_values):
    from math import sin, cos, tan, exp, log
    for k, v in symbol_values.items():
        locals()[k] = v
    s = 'result = ' + s
    exec(s)
    return locals()['result']


def as_str(f):
    if f.left and f.right:
        s = as_str(f.left) + f.label + as_str(f.right)
        s = '(' + s + ')'
        return s
    elif f.left:
        if f.label == '-':
            return '(' + f.label + '(' + as_str(f.left) + '))'
        return f.label + '(' + as_str(f.left) + ')'
    else:
        return str(f.label)


def generate_dataset(generate_one, nb=1000, random_state=None):
    np.random.seed(random_state)
    data = []
    while len(data) < nb:
        try:
            inst = generate_one()
        except ValueError:
            continue
        else:
            data.append(inst)
    return data


def save_dataset(data, symbols, units, filename):
    content = {
        'symbols' : symbols,
        'units' : units,
        'data' : data
    }
    with open(filename, 'wb') as fd:
        pickle.dump(content, fd)


def load_dataset(filename):
    with open(filename, 'rb') as fd:
        content = pickle.load(fd)
    return content

if __name__ == '__main__':
    print(as_theano('log(x+1+y)', ('x', 'y')))
