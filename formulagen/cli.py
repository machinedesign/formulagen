import time
import logging
from functools import partial
import os
import pickle
from clize import run
from uuid import uuid4
from glob import glob

from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

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

from formulagen.treegram import TreeGram


fmt = ''
logging.basicConfig(format=fmt)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')
EPS = 1e-7

def full(*, folder='instances', algo='eda', nb_formulas=1000, nb_points=1000, nb_generated=1000, nb_iterations=10):
    instance = str(uuid4())
    folder = '{}/{}'.format(folder, instance)
    os.mkdir(folder)
    print(folder)
    _full(folder=folder, 
          nb_formulas=nb_formulas, 
          nb_points=nb_points, 
          nb_generated=nb_generated,
          nb_iterations=nb_iterations,
          algo=algo)


def _full(folder='out', nb_formulas=1000, nb_points=1000, nb_generated=1000, nb_iterations=1, algo='eda'):
    """
    do the full pipeline once
    """
    data_folder = os.path.join(folder, 'data')
    models_folder = os.path.join(folder, 'models')
    out_folder = os.path.join(folder, 'out')
    os.mkdir(data_folder)
    os.mkdir(models_folder)
    os.mkdir(out_folder)
    generate_data(folder=data_folder, nb_formulas=nb_formulas, nb_points=nb_points)
    train(data='{}/formulas.pkl'.format(data_folder), 
          out_model='{}/formulas.pkl'.format(models_folder))
    train(data='{}/formulas_constraints.pkl'.format(data_folder), 
          out_model='{}/formulas_constraints.pkl'.format(models_folder))
    optimize_formulas(points='{}/dataset.npz'.format(data_folder), 
                      points_test='{}/test.npz'.format(data_folder),
                      formulas='{}/formulas_constraints.pkl'.format(data_folder), 
                      model='{}/formulas_constraints.pkl'.format(models_folder), 
                      out='{}/formulas_constraints.csv'.format(out_folder),
                      nb_generated=nb_generated,
                      nb_iterations=nb_iterations,
                      algo=algo)
    optimize_formulas(points='{}/dataset.npz'.format(data_folder),
                      points_test='{}/test.npz'.format(data_folder),
                      formulas='{}/formulas.pkl'.format(data_folder), 
                      model='{}/formulas.pkl'.format(models_folder), 
                      out='{}/formulas.csv'.format(out_folder),
                      nb_generated=nb_generated,
                      nb_iterations=nb_iterations,
                      algo=algo)
    clean(folder=folder)


def generate_data(folder='data', nb_formulas=1000, nb_points=1000):
    """
    generate:
        1) a dataset unconstrained formulas
        2) a dataset of constrained formulas
        3) a dataset of points with their corresponding outputs using a randomly chosen formula
           from the constrained ones
    """
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
    X = np.random.uniform(0, 1, size=(nb_points * 2, len(symbols)))#train/test
    X = X.astype(np.float32)

    # take one formula and use it as a held-out formula
    y = None
    for i, formula in enumerate(data):
        try:
            y = _evaluate_dataset(X, as_str(formula), symbols)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not _is_valid_output(y):
            continue
        log.info('Held-out formula : ' + as_str(formula))
        break
    assert y is not None, 'No suitable formula found to hold out'
    with open(os.path.join(folder, 'held_out'), 'w') as fd:
        fd.write(as_str(formula))
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
    np.savez(name, X=X[0:nb_points], y=y[0:nb_points])

    name = os.path.join(folder, 'test.npz')
    log.debug('save point to {}'.format(name))
    np.savez(name, X=X[nb_points:], y=y[nb_points:])



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


def optimize_formulas(*, points='data/dataset.npz', points_test='data/test.npz', formulas='data/formulas.pkl', 
                      model='models/model.pkl', out='out/formulas.csv',
                      nb_generated=1000, nb_iterations=1,
                      algo='eda'):
    """
    1) generate a set of formulas from a generator 'model' trained on the dataset 'formulas'
    2) evaluate each generated formula on the dataset 'points'
    3) write the results of the eevaluation on 'out'
    """
    rng = np.random
    log.info('Loading dataset in {}...'.format(formulas))

    dataset = load_dataset(formulas)
    D = np.load(points)
    X, y_true = D['X'], D['y']
    mean = y_true.mean()
    sigma = y_true.std()
    Dt = np.load(points_test)
    X_test, y_test = Dt['X'], Dt['y']

    data = dataset['data']
    units = dataset['units']
    symbols = dataset['symbols']
    data_str = list(map(as_str, data))
 
    max_size = max(map(len, data_str))
    model = _load_model(model)
    log.info('Optimize...')
    
    def sample_valid_formulas(nb, model):
        G = [model.generate(rng, max_size=max_size) for _ in range(nb)]
        G = filter(_not_none, G)
        G = filter(partial(_syntax_ok, units=units), G)
        G = list(G)
        return G
    
    # types of optimizations

    def onestep():
        G = sample_valid_formulas(nb_generated, model)
        G = _drop_duplicates(G)
        df = _evaluate_formulas(G, X, y_true, mean, sigma, symbols)
        df = df[df['is_valid']]
        return df
    
    def random():
        all = []
        all_vals = []
        for i in range(nb_iterations):
            log.info('Eval...')
            G = sample_valid_formulas(nb_generated, model)
            G = _drop_duplicates(G)
            df = _evaluate_formulas(G, X, y_true, mean, sigma, symbols)
            df = df[df['is_valid']]
            df.to_csv(out + '_it{:03d}'.format(i))
            G = df['formula'].values.tolist()
            vals = df['mse'].values.tolist()
            all.extend(G)
            all_vals.extend(vals)
            print(np.min(vals))
        df = pd.DataFrame({'formula': all, 'mse': all_vals})
        return df

    def eda(warm_start=False):
        cur_model = model
        if warm_start:
            df = _evaluate_formulas(data_str, X, y_true, mean, sigma, symbols)
            df = df[df['is_valid']]
            best = df['formula'].values.tolist()
            best_vals = df['mse'].values.tolist()
        else:
            best = []
            best_vals = []
        for i in range(nb_iterations):
            log.info('Eval...')
            G = sample_valid_formulas(nb_generated, cur_model)
            G = _drop_duplicates(G)
            df = _evaluate_formulas(G, X, y_true, mean, sigma, symbols)
            df = df[df['is_valid']]
            df.to_csv(out + '_it{:03d}'.format(i))
            df = df.sort_values(by='mse')
            G = df['formula'].values.tolist()
            vals = df['mse'].values.tolist()
            # append generated to best
            best += G
            best_vals += vals
            # remove duplicates from best
            dup = _is_dup(best)
            best = [b for b, d in zip(best, dup) if not d]
            best_vals = [v for v, d in zip(best_vals, dup) if not d]
            # take only best nb_generated top
            indices = np.argsort(best_vals)
            best = [best[i] for i in indices]
            best_vals = [best_vals[i] for i in indices]
            best = best[0:nb_generated]
            best_vals = best_vals[0:nb_generated]
            print(np.min(best_vals), len(best))
            for b in best[0:10]:
                print(b)
            # fit a new model using best
            cur_model = _fit_model(best)
        df = pd.DataFrame({'formula': best, 'mse': best_vals})
        return df


    def eda_weights():
        df = _evaluate_formulas(data_str, X, y_true, mean, sigma, symbols)
        df = df[df['is_valid']]
        all = df['formula'].values.tolist()
        all_vals = df['mse'].values.tolist()
        cur_model = model

        for i in range(nb_iterations):
            log.info('Eval...')
            G = sample_valid_formulas(nb_generated, cur_model)
            G = _drop_duplicates(G)
            df = _evaluate_formulas(G, X, y_true, mean, sigma, symbols)
            df = df[df['is_valid']]
            df.to_csv(out + '_it{:03d}'.format(i))
            G = df['formula'].values.tolist()
            vals = df['mse'].values.tolist()
            all += G
            all_vals += vals
            # remove duplicates from best
            dup = _is_dup(all)
            all = [b for b, d in zip(all, dup) if not d]
            all_vals = [v for v, d in zip(all_vals, dup) if not d]
            print(np.min(all_vals), len(all_vals))
            pr = np.array(all_vals)
            pr = (1. / pr)
            pr /= pr.sum()
            inds = np.random.multinomial(1, pr, size=nb_generated).argmax(axis=1)
            best = [all[i] for i in inds]
            # fit a new model using best
            cur_model = _fit_model(best)
        df = pd.DataFrame({'formula': all, 'mse': all_vals})
        return df

    if algo == 'eda':
        df = eda()
    elif algo == 'eda_warm_start':
        df = eda(warm_start=True)
    elif algo == 'eda_weights':
        df = eda_weights()
    elif algo == 'eda_thresh':
        df = eda_thresh()
    elif algo == 'one_step':
        df = one_step()
    elif algo == 'bayesopt':
        df = bayesopt()
    print(df['mse'].min())
    G = df['formula'].values.tolist()
    df_test = _evaluate_formulas(G, X_test, y_test, mean, sigma, symbols)
    df['mse_test'] = df_test['mse']
    df.to_csv(out)


def _evaluate_formulas(G, X, y_true, mean, sigma, symbols):
    y_true_ = (y_true - mean) / sigma
    df = []
    for i, formula in enumerate(G):
        try:
            y_pred = _evaluate_dataset(X, formula, symbols)
        except (ValueError, ZeroDivisionError, OverflowError, AssertionError) as exc:
            df.append({'error': str(exc), 'formula': formula, 'is_valid': False})
            continue
        if not _is_valid_output(y_pred):
            df.append({'error': 'output_not_valid', 'formula': formula, 'is_valid': False})
            continue
        y_pred_ = (y_pred - mean) / sigma
        mse = ((y_pred_ - y_true_) ** 2).mean()
        df.append({'mse': mse, 'formula': formula, 'error': '', 'is_valid': True})
    df = pd.DataFrame(df)
    return df


_not_none = lambda x:x is not None

def _syntax_ok(s, units):
    try:
        t = as_tree(s, units=units)
    except ParseError:
        return False
    else:
        return True


def _as_tree_or_none(s, *args, **kwargs):
    try:
        t = as_tree(s, *args, **kwargs)
    except ParseError:
        return None
    else:
        return t


def _is_dup(G):
    s = set()
    d = []
    for g in G:
        if g in s:
            d.append(True)
        else:
            d.append(False)
            s.add(g)
    return d


def _drop_duplicates(G):
    return list(set(G))



def plot(*, folder='out'):
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
    df1 = df1.sort_values(by='mse', ascending=False)
    
    df2 = pd.read_csv(d2, index_col=0)
    df2 = df2.sort_values(by='mse', ascending=False)

    output_file('{}/scatter.html'.format(folder))
    p = figure(title="evolution of MSE with iter")
    p.line(np.arange(len(df1)), df1['mse'], legend="without constraints", line_width=2, color='blue')
    p.line(np.arange(len(df2)), df2['mse'], legend="with constraints", line_width=2, color='red')
    show(p)
    df = ([{'val': v, 'type': 'without constraints'} for v in df1['mse']] + 
          [{'val': v, 'type': 'with constraints'} for v in df2['mse']])
    df = pd.DataFrame(df)

    output_file('{}/bar.html'.format(folder))
    p = Bar(df, 'type', values='val', color='type', title="diff between with/without constraints", agg='mean')
    show(p)

    output_file('{}/boxplot.html'.format(folder))
    p = BoxPlot(df, values='val', label='type', title="diff between with/without constraints")
    show(p)
    
    m1s, m2s = _get_curves(folder)
    output_file('{}/iter.html'.format(folder))
    p = figure(title="evolution of MSE with iter")
    p.line(np.arange(len(m1s)), m1s, legend="without constraints", line_width=2, color='blue')
    p.line(np.arange(len(m2s)), m2s, legend="with constraints", line_width=2, color='red')
    show(p)


def global_heldout_eval(*, folder='instances'):
    symbols = ('x', 'y', 'z', 'b')
    for f in os.listdir(folder):
        dataf = os.path.join(folder, f, 'data')
        outf = os.path.join(folder, f, 'out')
        name = os.path.join(folder, f, 'data', 'held_out')
        if not os.path.exists(name):
            continue
        if not os.path.exists(os.path.join(outf, 'formulas.csv')):
            continue
        if not os.path.exists(os.path.join(outf, 'formulas_constraints.csv')):
            continue
        formula = open(name).read()
        data = np.load(os.path.join(dataf, 'dataset.npz'))
        data_test = np.load(os.path.join(dataf, 'test.npz'))
        X_train = data['X']
        y_train = data['y']
        X_test = data_test['X']
        y_test = data_test['y']
        m, s = y_train.mean(), y_train.std()
        y_train = (y_train - m) / s
        y_test = (y_test - m) / s

        c = pd.read_csv(os.path.join(outf, 'formulas_constraints.csv'))
        wc =  pd.read_csv(os.path.join(outf, 'formulas.csv'))
        
        fc = c.sort_values(by='mse').iloc[0]
        fwc = wc.sort_values(by='mse').iloc[0]
        v = _evaluate_dataset(X_train, fc['formula'], symbols)
        v = (v - v.mean()) / v.std()
        c_mse = ((v - y_train) ** 2).mean()




def global_plots(*, folder='instances'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    p = figure(title="evolution of MSE with iter")
    m1 = []
    m2 = []
    nb_iter = 30
    for i, instance in enumerate(os.listdir(folder)):
        f = os.path.join(folder, instance, 'out')
        m1s, m2s = _get_curves(f)
        if not(len(m1s) == nb_iter and len(m2s) == nb_iter):
            continue
        m1.append(m1s)
        m2.append(m2s)
    m1 = np.array(m1)
    m2  = np.array(m2)
    u_m1 = np.mean(m1, axis=0)
    u_m2 = np.mean(m2, axis=0)
    s_m1 = np.std(m1, axis=0)
    s_m2 = np.std(m2, axis=0)
    fig = plt.figure()
    ax = sns.tsplot(m1, time=np.arange(1, nb_iter + 1), color='blue')
    ax = sns.tsplot(m2, time=np.arange(1, nb_iter + 1), color='red')
    ax.legend(["without constraints", "with constraints"])
    plt.xlabel('Iteration')
    plt.ylabel('Normalized MSE')
    plt.savefig('{}/iter_global.png'.format(folder))



def global_fit_numerical_data(*, folder='instances', model_type='neuralnet'):
    for f in os.listdir(folder):
        print(f)
        dataf = os.path.join(folder, f, 'data')
        outf = os.path.join(folder, f, 'out')
        #if os.path.exists(os.path.join(outf, 'neuralnet.csv')):
        #    continue
        if not os.path.exists(os.path.join(outf, 'formulas.csv')):
            continue
        if not os.path.exists(os.path.join(outf, 'formulas_constraints.csv')):
            continue
        _fit_numerical_data(folder=dataf, out=outf, model_type=model_type)



def _fit_numerical_data(*, folder='data', out='out', model_type='neuralnet'):
    df_c = pd.read_csv(os.path.join(out, 'formulas_constraints.csv'))
    df_wc = pd.read_csv(os.path.join(out, 'formulas.csv'))
    print('with constraints    : {:.6f}'.format(df_c['mse'].min()))
    print('without constraints : {:.6f}'.format(df_wc['mse'].min()))
    formula = open(os.path.join(folder, 'held_out')).read()
    formula_t = as_tree(formula)
    
    data = np.load(os.path.join(folder, 'dataset.npz'))
    data_test = np.load(os.path.join(folder, 'test.npz'))

    X_train = data['X']
    y_train = data['y']
    X_test = data_test['X']
    y_test = data_test['y']
    m, s = y_train.mean(), y_train.std()
    y_train = (y_train - m) / s
    y_test = (y_test - m) / s

    if model_type == 'neuralnet':
        model = _fit_neural_net(X_train, y_train, folder=folder)
    elif model_type == 'genetic':
        model = _fit_genetic(X_train, y_train, folder=folder)
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()
    train_mse  = ((y_pred_train - y_train) ** 2).mean()
    test_mse = ((y_pred_test - y_test) ** 2).mean()
    print('Train MSE : {:.6f} Test MSE : {:.6f}'.format(train_mse, test_mse))
    mse= []
    mse.append({'train': train_mse, 'test': test_mse})
    mse = pd.DataFrame(mse)
    mse.to_csv(os.path.join(out, '{}.csv'.format(model_type)))
    print(mse.mean())



def _fit_neural_net(X, y, folder='.'):
    from keras.layers import Dense
    from keras.layers import Input
    from keras.layers import Activation
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint
    nb_layers = 2
    nb_units = 1000
    inp = Input((X.shape[1],))
    h = inp
    for _ in range(nb_layers):
        h = Dense(nb_units, activation='relu')(h)
    out = Dense(1)(h)
    model = Model(inp, out)
    filename = os.path.join(folder, 'nnet.hd5')
    cb = [
        ModelCheckpoint(filename, monitor='val_loss', save_best_only=True, verbose=0)
    ]
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3))
    model.fit(X, y, verbose=0, nb_epoch=200, batch_size=128, validation_split=0.2, callbacks=cb)
    model.load_weights(filename)
    os.remove(filename)
    return model



def _fit_genetic(X, y, folder='.'):
    from gplearn.genetic import SymbolicRegressor
    reg = SymbolicRegressor(
        function_set=('add', 'sub', 'mul', 'div', 'neg', 'sin', 'cos', 'tan'), 
        generations=30,
        const_range=(0.0, 1.0),
        metric='mse',
        verbose=1
    )
    reg.fit(X, y)
    return reg


def global_stats(*, folder='instances'):
    df = _read_global(folder)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df['rmse'] = np.sqrt(df['mse'])
    thresh = 1
    print('\nTop MSE < {}'.format(thresh))
    def f(x):
        return x.min()
        #return x[x < thresh].mean()
    d = df.groupby(['id', 'type']).agg(f).reset_index()
    print('Total Nb of repetitions : {}'.format(len(df['id'].unique())))
    print(d.groupby('type').mean())
    print(d.groupby('type').std())
    vals = d
    c = vals[vals['type']=='with_constraints']
    wc = vals[vals['type'] == 'without_constraints']
    c = c.sort_values(by='id')
    wc = wc.sort_values(by='id')
    print(len(c), len(wc))
    _, pvalue = ttest_ind(c['rmse'], wc['rmse'], equal_var=False)
    print('P-value : {}'.format(pvalue))
    c = c.reset_index().set_index('id')
    wc = wc.reset_index().set_index('id')
    diff = c['rmse'] - wc['rmse']
    print('Diff mean {} std {}'.format(diff.mean(), diff.std()))
    print('ratio of c<wc : {}'.format((c<wc).mean()['rmse']))
    # count nb of duplicates


def _read_global(folder):
    import pandas as pd
    dfg = []
    for instance in os.listdir(folder):
        f1 = os.path.join(folder, instance, 'out', 'formulas.csv')
        if not os.path.exists(f1):
            continue
        f2 = os.path.join(folder, instance, 'out', 'formulas_constraints.csv')
        if not os.path.exists(f2):
            continue
        idx = instance
        for filename, t in ((f1, 'without_constraints'), (f2, 'with_constraints')):
            df = pd.read_csv(filename)
            d = {'mse': df['mse'], 'type': [t] * len(df), 'id': [idx] * len(df)}
            d = pd.DataFrame(d)
            dfg.append(d)
    dfg = pd.concat(dfg)
    return dfg


def clean(folder):
    for filename in glob(os.path.join(folder, 'data', '*.pkl')):
        os.remove(filename)
    for filename in glob(os.path.join(folder, 'models', '*.pkl')):
        os.remove(filename)


def global_compare(*, folder='instances', model_type='neuralnet'):
    nb_better = 0
    total = 0
    rows = []
    for f in os.listdir(folder):
        outf = os.path.join(folder, f, 'out')
        filename = os.path.join(outf, model_type + '.csv') 
        if not os.path.exists(filename):
            continue
        if not os.path.exists(os.path.join(outf, 'formulas.csv')):
            continue
        if not os.path.exists(os.path.join(outf, 'formulas_constraints.csv')):
            continue
        formula = open(os.path.join(folder, f, 'data', 'held_out')).read()
        df = pd.read_csv(filename)
        df_c = pd.read_csv(os.path.join(outf, 'formulas_constraints.csv'))
        df_wc = pd.read_csv(os.path.join(outf, 'formulas.csv'))
        print('with constraints    : {:.6f}'.format(df_c['mse'].min()))
        print('without constraints : {:.6f}'.format(df_wc['mse'].min()))
        print('Neural net          : {:.6f}'.format(df['test'].min()))
        print('\n')
        c = df_c['mse'].min()
        wc = df_wc['mse'].min()
        o = df['test'].min()
        better = min(c, wc) < o
        rows.append({'c': c, 'wc': wc, 'nnet': o, 'f': formula, 'better': better})
    df = pd.DataFrame(rows)
    print('better at:')
    for f in df[df['better']]['f'].values.tolist():
        print(f)
    print('worse at:')
    for f in df[~df['better']]['f'].values.tolist():
        print(f)


def print_heldout(*, folder='instances'):
    for f in os.listdir(folder):
        name = os.path.join(folder, f, 'data', 'held_out')
        if os.path.exists(name):
            formula = open(name).read()
            print(formula)


def _evaluate_dataset(X, formula, symbols):
    y = []
    for x in X:
        vals = {s: v for v, s in zip(x, symbols)}
        out = evaluate(formula, vals)
        y.append(out)
    y = np.array(y)
    return y


def _fit_model(corpus):
    min_gram = 1
    max_gram = 10
    model = NGram(min_gram=min_gram, max_gram=max_gram)
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


def _is_valid_output(y):
    if np.any(np.isinf(y)):
        return False
    if np.any(np.isnan(y)):
        return False
    if y.var() <= EPS:
        return False
    return True


def _get_curves(folder):
    m1s = []
    m2s = []
    for i in range(100):
        d1 = '{}/formulas.csv_it{:03d}'.format(folder, i)
        d2 = '{}/formulas_constraints.csv_it{:03d}'.format(folder, i)
        if not os.path.exists(d1):
            continue
        if not os.path.exists(d2):
            continue
        df1 = pd.read_csv(d1, index_col=0)
        df2 = pd.read_csv(d2, index_col=0)
        m1 = df1['mse'].min()
        m2 = df2['mse'].min()
        m1s.append(m1)
        m2s.append(m2)
    m1s = np.array(m1s)
    m2s = np.array(m2s)
    m1s = np.minimum.accumulate(m1s)
    m2s = np.minimum.accumulate(m2s)
    return m1s, m2s



if __name__ == '__main__':
    run(full,  
        global_stats, 
        global_plots, 
        global_fit_numerical_data, 
        global_compare, 
        global_heldout_eval,
        print_heldout,)
