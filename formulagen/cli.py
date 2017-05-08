import logging
from functools import partial
import os
import pickle
from clize import run

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
from formulagen.formula import as_theano

fmt = ''
logging.basicConfig(format=fmt)
log = logging.getLogger(__name__)
log.setLevel('INFO')

def full():
    generate_data()
    train(data='data/formulas.pkl', out_model='models/formulas.pkl')
    train(data='data/formulas_constraints.pkl', out_model='models/formulas_constraints.pkl')
    generate_formulas(points='data/dataset.npz', 
                      formulas='data/formulas_constraints.pkl', 
                      model='models/formulas_constraints.pkl', 
                      out='out/formulas_constraints.csv')
    generate_formulas(points='data/dataset.npz', 
                      formulas='data/formulas.pkl', 
                      model='models/formulas.pkl', 
                      out='out/formulas.csv')


def generate_data():
    import theano
    nb = 1000
    min_depth = 2
    max_depth = 10
    nb_points = 1000
    folder = 'data'

    symbols = ('x', 'y', 'z', 'b')
    units = {}
    units['x'] = Unit({'m': 1})
    units['y'] = Unit({'s': 1})
    units['z'] = Unit({'g': 1})
    units['b'] = constant_unit
    # define generic function generator
    gen = partial(
        gen_formula_tree,
        symbols=symbols,
        units=units,
        min_depth=min_depth,
        max_depth=max_depth
    )
    # generate with constraints
    gen_with_constraints = partial(gen, unit_constraints=True)
    data = generate_dataset(gen_with_constraints, nb=nb)
    data = list(data)
    # take one formula and use it as a held-out formula
    formula = data[0]
    data = data[1:]
    name = os.path.join(folder, 'formulas_constraints.pkl')
    log.info('Save dataset to {}'.format(name))
    save_dataset(data, symbols, units, name)
    for d in data:
        check_constraints(d)
    # generate without constraints
    gen_without_constraints = partial(gen, unit_constraints=False)
    data = generate_dataset(gen_without_constraints, nb=nb)
    data = list(data)
    name = os.path.join(folder, 'formulas.pkl')
    log.info('Save dataset to {}'.format(name))
    save_dataset(data, symbols, units, name)

    # generate points from one formula taken from the constrained formulas
    X = np.random.uniform(0, 1, size=(nb_points, len(symbols)))
    X = X.astype(np.float32)
    vars = as_theano(as_str(formula), symbols)
    pred = theano.function([vars[s] for s in symbols], vars['result'], on_unused_input='ignore')
    y = pred(*[x for x in X.T])
    np.savez(os.path.join(folder, 'dataset.npz'), X=X, y=y)


def train(*, data='data/formulas.pkl', out_model='models/model.pkl'):
    log.info('Loading dataset in {}...'.format(data))
    dataset = load_dataset(data)
    data = dataset['data']
    units = dataset['units']
    symbols = dataset['symbols']
    data_str = list(map(as_str, data))
    model = _fit_model(data_str)
    log.info('Save model to {}'.format(out_model))
    _save_model(model, out_model)


def generate_formulas(*, points='data/dataset.npz', formulas='data/formulas.pkl', model='models/model.pkl', out='out/formulas.csv'):
    import theano
    import pandas as pd
    nb = 1000
    rng = np.random

    log.info('Loading dataset in {}...'.format(formulas))
    dataset = load_dataset(formulas)
    D = np.load(points)
    X, y_true = D['X'], D['y']
    data = dataset['data']
    units = dataset['units']
    symbols = dataset['symbols']
    data_str = list(map(as_str, data))
 
    log.info('Generate...')
    max_size = max(map(len, data_str))
    model = _load_model(model)
    G = [model.generate(rng, max_size=max_size) for _ in range(nb)]
    G = list(set(G))
    total = len(G)
    syntax_ok = [False] * len(G)
    constraints_ok = [False] * len(G)
    for i, s in enumerate(G):
        if s is None:
            continue
        try:
            t = as_tree(s, units)
        except ParseError:
            continue
        syntax_ok[i] = True
        try:
            check_constraints(t)
        except Exception:
            continue
        constraints_ok[i] = True
    log.debug('Syntax ok       : {}'.format(sum(syntax_ok)))
    log.debug('Constraints ok  : {}'.format(sum(constraints_ok)))
    log.debug('Total unique    : {}'.format(total))
    log.debug('Total generated : {}'.format(nb))

    log.info('Regress...')
    df = []
    for i, formula in enumerate(G):
        if not syntax_ok[i]:
            continue
        vars = as_theano(formula, symbols)
        pred = theano.function([vars[s] for s in symbols], vars['result'], on_unused_input='ignore')
        y_pred = pred(*[x for x in X.T])
        mse = ((y_pred - y_true) ** 2).mean()
        r2 = mse / y_pred.var()
        df.append({'mse': mse, 'r2': r2})
    pd.DataFrame(df).to_csv(out)


def plot():
    import pandas as pd
    from bokeh.plotting import figure, output_file, show
    from bokeh.layouts import row

    d1 = 'out/formulas.csv'
    d2 = 'out/formulas_constraints.csv'
    out = 'out/scatter.html'
    output_file(out)

    df1 = pd.read_csv(d1, index_col=0)
    df1 = df1.dropna()
    df1 = df1.sort_values(by='r2')
    
    df2 = pd.read_csv(d2, index_col=0)
    df2 = df2.dropna()
    df2 = df2.sort_values(by='r2')
    
    p = figure(title="evolution R2 with iter")
    
    iter = np.arange(len(df2))
    val = df1['r2']
    p.line(iter, val, legend="without constraints", line_width=2, color='blue')

    iter = np.arange(len(df2))
    val = df2['r2']
    p.line(iter, val, legend="with constraints", line_width=2, color='red')
    show(p)



def _fit_model(corpus):
    min_gram = 1
    max_gram = 10
    model = NGram(min_gram=min_gram, max_gram=max_gram, begin='^', end='$')
    log.info('fitting model...')
    model.fit(corpus)
    return model


def _save_model(model, filename):
    with open(filename, 'wb') as fd:
        pickle.dump(model, fd)


def _load_model(filename):
    with open(filename, 'rb') as fd:
        model = pickle.load(fd)
    return model



if __name__ == '__main__':
    run(full, train, generate_data, generate_formulas, plot)
