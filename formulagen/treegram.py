import numpy as np
from collections import defaultdict

from formulagen.formula import as_str
from formulagen.formula import as_tree
from formulagen.formula import gen_formula_tree
from formulagen.formula import Unit
from formulagen.formula import Node

def count():
    return defaultdict(int)

class TreeGram:
    
    def __init__(self, min_gram=1, max_gram=5):
        self.min_gram = min_gram
        self.max_gram = max_gram
        self._counts = defaultdict(count)
        self._nb_children = defaultdict(count)
        self.vocab_ = set()

    def fit(self, corpus):
        self.vocab_ = set()
        for doc in corpus:
            tree = as_tree(doc)
            _count(tree, counts=self._counts, vocab=self.vocab_, nb_children=self._nb_children, min_gram=self.min_gram, max_gram=self.max_gram)
        self.vocab_ = list(self.vocab_)
    
    def generate(self, rng, max_size=10):
        tree = _generate(self._counts, self.vocab_, self._nb_children, min_gram=self.min_gram, max_gram=self.max_gram, rng=rng)
        return as_str(tree)


def _count(root, counts, vocab, nb_children, child_idx=None, min_gram=1, max_gram=5):
    def _tree_count(t, context=tuple(), child_idx=0):
        if len(context) == 0:
            counts[(None, context)][t.label] += 1 # root
        # others
        first = min_gram
        last = min(max_gram, len(context))
        for i in range(first, last + 1):
            ctx = context[len(context) - i:]
            counts[(child_idx, ctx)][t.label] += 1
        if t.left and t.right:
            children = [t.left, t.right]
        elif t.left:
            children = [t.left]
        else:
            children = []
        nb_children[t.label][len(children)] += 1
        vocab.add(t.label)
        for ic, c in enumerate(children):
            _tree_count(c, context=context + (t.label,), child_idx=ic)
    return _tree_count(root)


def _generate(counts, vocab, nb_children, child_idx=None, min_gram=1, max_gram=2, rng=np.random):
    def _gen_tree(context=tuple(), child_idx=None):
        if child_idx is None: #root
            found = (None, context) in counts
            ctx = context
        else:#others
            first = min_gram
            last = min(max_gram, len(context))
            degs = reversed(range(first, last + 1))
            degs = list(degs)
            found = False
            for i in degs:
                ctx = context[len(context) - i:]
                if (child_idx, ctx) in counts:
                    found = True
                    break
        if found:
            cnt = counts[(child_idx, ctx)]
            label = _sample(cnt, rng)
        else:
            label = rng.choice(vocab)
        nb = _sample(nb_children[label], rng)
        children = []
        for idx in range(nb):
            child = _gen_tree(context=context + (label,), child_idx=idx)
            children.append(child)
        if len(children) == 2:
            left, right = children
        elif len(children) == 1:
            left, = children
            right = None
        else:
            left = None
            right = None
        return Node(label=label, left=left, right=right, unit=None)
    return _gen_tree()


def _sample(counts, rng):
    labels = list(counts.keys())
    probas = list(counts.values())
    probas = np.array(probas).astype(np.float32)
    probas /= probas.sum()
    label_idx = rng.multinomial(1, probas).argmax()
    label = labels[label_idx]
    return label


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('instances/67e6bff2-4dd0-4d43-b140-b57f21e06c60/out/formulas.csv')
    corpus = df['formula']
    print(sum(map(len, corpus)) / len(corpus))
    model = TreeGram(min_gram=1, max_gram=10)
    model.fit(corpus)
    gen = []
    for i in range(1000):
        gen.append(model.generate())
    print(sum(map(len, gen)) / len(gen))
