import logging
from functools import partial
import os
import pickle
from clize import run
from uuid import uuid4

import numpy as np

from machinedesign.text.ngram import NGram

from formulagen.formula import get_symbols
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
from formulagen.formula import evaluate

fmt = ''
logging.basicConfig(format=fmt)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def full():
    """
    do the full pipeline
    """
    instance = str(uuid4())
    folder = 'instances/{}'.format(instance)
    os.mkdir(folder)
    data_folder = os.path.join(folder, 'data')
    models_folder = os.path.join(folder, 'models')
    out_folder = os.path.join(folder, 'out')
    os.mkdir(data_folder)
    os.mkdir(models_folder)
    os.mkdir(out_folder)
    nb_formulas = 100
    nb_points = 1000
    nb_generated = 100

    # generate formulas with and without constraints
    # take 1 formula from the constrained ones, use it to
    # generate a numerical dataset
    generate_data(folder=data_folder, nb_formulas=nb_formulas, nb_points=nb_points)
    # train a generator on formulas
    train(data='{}/formulas.pkl'.format(data_folder), 
          out_model='{}/formulas.pkl'.format(models_folder))
    # train a generator on constrained formulas
    train(data='{}/formulas_constraints.pkl'.format(data_folder), 
          out_model='{}/formulas_constraints.pkl'.format(models_folder))
    # generate formulas from the model trained on unconstrained formulas and and evaluate 
    # on the numerical dataset
    generate_formulas(points='{}/dataset.npz'.format(data_folder), 
                      formulas='{}/formulas_constraints.pkl'.format(data_folder), 
                      model='{}/formulas_constraints.pkl'.format(models_folder), 
                      out='{}/formulas_constraints.csv'.format(out_folder),
                      nb_generated=nb_generated)
    # generate formulas from the model trained on constrained formulas and evaluate
    # on the numerical dataset
    generate_formulas(points='{}/dataset.npz'.format(data_folder), 
                      formulas='{}/formulas.pkl'.format(data_folder), 
                      model='{}/formulas.pkl'.format(models_folder), 
                      out='{}/formulas.csv'.format(out_folder),
                      nb_generated=nb_generated)
    plot(folder=out_folder)

def generate_data(folder='data', nb_formulas=1000, nb_points=1000):
    """
    generate:
        1) a dataset unconstrained formulas
        2) a dataset of constrained formulas
        3) a dataset of points with their corresponding outputs using a randomly chosen formula
           from the constrained ones
    """
    import theano
    import theano.tensor as T
    min_depth = 2
    max_depth = 10
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
    data = generate_dataset(gen_with_constraints, nb=nb_formulas)
    data = list(data)

    # generate points from one formula taken from the constrained formulas
    log.info('Generate points from the held-out formula...')
    X = np.random.uniform(0, 1, size=(nb_points, len(symbols)))
    X = X.astype(np.float32)

    # take one formula and use it as a held-out formula
    y = None
    for i, formula in enumerate(data):
        syms = get_symbols(formula)
        if len(set(['x', 'y', 'z', 'b']) & syms) == 4:
            try:
                y = _evaluate_dataset(X, as_str(formula), symbols)
            except ValueError:
                continue
            except ZeroDivisionError:
                continue
            log.info('Held-out formula : ' + as_str(formula))
            break
    assert y is not None, 'No suitable formula found to hold out'
    formula = data[i]
    data = data[0:i] + data[i + 1:]
    name = os.path.join(folder, 'formulas_constraints.pkl')
    log.info('Save dataset to {}'.format(name))
    save_dataset(data, symbols, units, name)
    for d in data:
        check_constraints(d)
    # generate without constraints
    gen_without_constraints = partial(gen, unit_constraints=False)
    data = generate_dataset(gen_without_constraints, nb=nb_formulas)
    data = list(data)
    name = os.path.join(folder, 'formulas.pkl')
    log.info('Save dataset to {}'.format(name))
    save_dataset(data, symbols, units, name)

    name = os.path.join(folder, 'dataset.npz')
    log.debug('save point to {}'.format(name))
    np.savez(name, X=X, y=y)

def _evaluate_dataset(X, formula, symbols):
    y = []
    for x in X:
        vals = {s: v for v, s in zip(x, symbols)}
        out = evaluate(formula, vals)
        y.append(out)
    y = np.array(y)
    return y


def train(*, data='data/formulas.pkl', out_model='models/model.pkl'):
    """
    train a formula generator on the dataset 'data' and save the model
    on out_model
    """
    log.info('Loading dataset in {}...'.format(data))
    dataset = load_dataset(data)
    data = dataset['data']
    units = dataset['units']
    symbols = dataset['symbols']
    data_str = list(map(as_str, data))
    model = _fit_model(data_str)
    log.info('Save model to {}'.format(out_model))
    _save_model(model, out_model)


def generate_formulas(*, points='data/dataset.npz', formulas='data/formulas.pkl', 
                      model='models/model.pkl', out='out/formulas.csv',
                      nb_generated=1000):
    """
    1) generate a set of formulas from a generator 'model' trained on the dataset 'formulas'
    2) evaluate each generated formula on the dataset 'points'
    3) write the results of the eevaluation on 'out'
    """
    import theano
    import theano.tensor as T
    import pandas as pd
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
    G = [model.generate(rng, max_size=max_size) for _ in range(nb_generated)]
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
    log.debug('Total generated : {}'.format(nb_generated))

    log.info('Regress...')
    df = []
    for i, formula in enumerate(G):
        if not syntax_ok[i]:
            continue
        try:
            y_pred = _evaluate_dataset(X, formula, symbols)
        except ValueError:
            continue
        except ZeroDivisionError:
            continue
        mse = ((y_pred - y_true) ** 2).mean()
        r2 = 1.0 - mse / y_true.var()
        df.append({'mse': mse, 'r2': r2, 'formula': formula})
    
    df = pd.DataFrame(df)
    df.to_csv(out)


def plot(folder='out'):
    """
    plot evolution of MSE for the model trained on unconstrained
    formulas and for the model trained on constrained formulas.
    """
    import pandas as pd
    from bokeh.plotting import figure, output_file, show
    from bokeh.charts import Histogram, Bar, BoxPlot
    from bokeh.layouts import row

    d1 = '{}/formulas.csv'.format(folder)
    d2 = '{}/formulas_constraints.csv'.format(folder)

    df1 = pd.read_csv(d1, index_col=0)
    df1 = df1.replace([np.inf, -np.inf], np.nan)
    df1 = df1.dropna(subset=['mse'])
    df1 = df1.sort_values(by='mse', ascending=False)
    df1['log_mse'] = np.log(1 + df1['mse'])
    log.info('without constraints : ' + df1['formula'].iloc[-1])

    df2 = pd.read_csv(d2, index_col=0)
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna(subset=['mse'])
    df2 = df2.sort_values(by='mse', ascending=False)
    df2['log_mse'] = np.log(1 + df2['mse'])
    log.info('with constraints : ' + df2['formula'].iloc[-1])

    output_file('{}/scatter.html'.format(folder))
    p = figure(title="evolution of log.MSE with iter")
    p.line(np.arange(len(df1)), df1['log_mse'], legend="without constraints", line_width=2, color='blue')
    p.line(np.arange(len(df2)), df2['log_mse'], legend="with constraints", line_width=2, color='red')
    show(p)
    df = ([{'val': v, 'type': 'without constraints'} for v in df1['log_mse']] + 
          [{'val': v, 'type': 'with constraints'} for v in df2['log_mse']])
    df = pd.DataFrame(df)

    output_file('{}/bar.html'.format(folder))
    p = Bar(df, 'type', values='val', color='type', title="diff between with/without constraints", agg='mean')
    show(p)

    output_file('{}/boxplot.html'.format(folder))
    p = BoxPlot(df, values='val', label='type', title="diff between with/without constraints")
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
