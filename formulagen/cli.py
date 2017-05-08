from clize import run
import logging
from functools import partial

import numpy as np

from machinedesign.text.ngram import NGram

from formulagen.formula import gen_formula_tree
from formulagen.formula import generate_dataset
from formulagen.formula import constant_unit
from formulagen.formula import load_dataset
from formulagen.formula import save_dataset
from formulagen.formula import Unit
from formulagen.formula import Node
from formulagen.formula import check_constraints
from formulagen.formula import as_tree
from formulagen.formula import as_str
from formulagen.formula import ParseError

fmt = ''
logging.basicConfig(format=fmt)
log = logging.getLogger(__name__)


def train(*, data='formulas.pkl', out_folder='out', log_level='INFO'):
    log.setLevel(log_level)
    log.info('Loading dataset...')
    dataset = load_dataset(data)
    data = dataset['data']
    units = dataset['units']
    data_str = list(map(as_str, data))
    max_size = max(map(len, data_str))
    model = _fit_model(data_str)
    rng = np.random
    nb = 1000
    G = [model.generate(rng, max_size=max_size) for _ in range(nb)]
    G = list(set(G))
    total = len(G)
    syntax_ok = 0
    constraints_ok = 0
    for s in G:
        if s is None:
            continue
        try:
            t = as_tree(s, units)
        except ParseError:
            continue
        syntax_ok += 1
        try:
            check_constraints(t)
        except Exception:
            continue
        constraints_ok += 1
    print('Syntax ok       : {}'.format(syntax_ok))
    print('Constraints ok  : {}'.format(constraints_ok))
    print('Total unique    : {}'.format(total))
    print('Total generated : {}'.format(nb))


def generate_data(*, nb=10000, min_depth=2, max_depth=10):
    symbols = ('x', 'y', 'z', 'b')
    units = {}
    units['x'] = Unit({'m': 1})
    units['y'] = Unit({'s': 1})
    units['z'] = Unit({'g': 1})
    units['b'] = constant_unit
    gen = partial(
        gen_formula_tree,
        symbols=symbols,
        units=units,
        min_depth=min_depth,
        max_depth=max_depth
    )
    gen_with_constraints = partial(gen, unit_constraints=True)
    data = generate_dataset(gen_with_constraints, nb=nb)
    data = list(data)
    save_dataset(data, symbols, units, 'formulas_constraints.pkl')
    for d in data:
        check_constraints(d)
    gen_without_constraints = partial(gen, unit_constraints=False)
    data = generate_dataset(gen_without_constraints, nb=nb)
    data = list(data)
    save_dataset(data, symbols, units, 'formulas.pkl')


def _fit_model(corpus):
    min_gram = 1
    max_gram = 10
    model = NGram(min_gram=min_gram, max_gram=max_gram, begin='^', end='$')
    log.info('fitting model...')
    model.fit(corpus)
    return model


if __name__ == '__main__':
    run(train, generate_data)
