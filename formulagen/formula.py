import numpy as np
from itertools import chain
from collections import Counter
from collections import namedtuple
from collections import defaultdict

import sympy

Node = namedtuple('Node', ['label', 'left', 'right', 'unit'])

class Unit(defaultdict):
    
    def __init__(self, d={}):
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

def gen_formula(symbols=('x',), units={'x': {'m': 1}}, min_depth=2, max_depth=4, unit_constraints=True, random_state=None):
    return _gen_formula_tree(symbols, units, min_depth=min_depth, max_depth=max_depth, 
                             unit_constraints=unit_constraints, random_state=random_state)

def _gen_formula_tree(symbols, units, min_depth=1, max_depth=2, random_state=None, unit_constraints=True):
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
            """
            elif op == '^':
                if force_unit is not None:
                    lnode = _gen(rng, depth=depth + 1)
                    rnode = Node(label=_gen_constant(), left=None, right=None, unit=constant_unit)
                    unit = lnode.unit + rnode.label
                else:
                    pass
            """
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
        return rng.randint(1, 10)

    return _gen(rng)

constant_unit = Unit() 

def as_str(f):
    if f.left and f.right:
        s = as_str(f.left) + ' ' + f.label + ' ' + as_str(f.right)
        s = '( ' + s + ' )'
        return s
    elif f.left:
        return f.label + '(' + as_str(f.left) + ')'
    else:
        return str(f.label)

def as_sympy(f, symbol_names):
    from sympy import symbols
    from sympy import cos, tan, sin, exp, log
    symbols = symbols(' '.join(symbol_names))
    for n, v in zip(symbol_names, symbols):
        locals()[n] = v
    s = 'result = {}'.format(as_str(f))
    exec(s)
    return locals()['result']
    
if __name__ == '__main__':
    symbols = ('x', 'y', 'b')
    units = {}
    units['x'] = Unit({'m': 1})
    units['y'] = Unit({'s': 1})
    units['b'] = Unit()
    seed = np.random.randint(1, 100)
    #seed = 22
    print(seed)
    f = gen_formula(symbols=symbols, units=units,  min_depth=2, max_depth=10, random_state=seed, unit_constraints=False)
    print(as_str(f))
    s = as_sympy(f, symbols)
    print(s)
