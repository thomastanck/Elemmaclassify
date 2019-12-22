import HRR
import parse
import utils

import collections
import functools
import zlib

import numpy as np
import sklearn as sk

import sklearn.svm as svm
import sklearn.preprocessing as pre
import sklearn.linear_model as lm

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

@utils.persist_to_file('ex1data/usefulness.pickle')
def get_usefulness():
    print('getting usefulness')
    with open('E_conj/statistics', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        usefulness = collections.defaultdict(dict)
        for l in ls:
            if not l.strip():
                continue
            psr, problem, lemmaname, *_ = l.split(':')
            psr = float(psr)
            lemmaname = lemmaname.split('.')[0]
            usefulness[problem][lemmaname] = psr

    return usefulness

@functools.lru_cache(maxsize=1)
def problem_hrr(problemname):
    print('getting problem hrr for {}'.format(problemname))
    return HRR.FlatTreeHRR.encode_conj(parse_problem(problemname))

def lemma_hrr(lemmastr):
    print('getting lemma hrr for {}'.format(lemmastr[:30]))
    return HRR.FlatTreeHRR.encode_disj(parse.parse_cnf_clause(lemmastr))

# problem -> lemmaname -> psr
# usefulness = get_usefulness()

@functools.lru_cache(maxsize=1)
def get_problemnames():
    problemnames = set()
    with open('E_conj/lemmas') as f:
        for l in f:
            name, lemma = l.split(':')
            _, problemname, lemmaname = name.split('/')
            problemnames.add(problemname)
    return sorted(list(problemnames))

@functools.lru_cache(maxsize=1)
def get_lemmanames():
    lemmanames = set()
    with open('E_conj/lemmas') as f:
        for l in f:
            name, lemma = l.split(':')
            _, problemname, lemmaname = name.split('/')
            lemmanames.add((problemname, lemmaname))
    return sorted(list(lemmanames))

@functools.lru_cache(maxsize=None)
def get_hrr_encoder(hrr_class, hrr_size):
    return HRR.FlatTreeHRR(hrr_size)

def process_problem(args):
    hrr_class, hrr_size, pname = args
    return (pname,
            get_hrr_encoder(hrr_class, hrr_size).encode_conj(
                parse.parse_cnf_file('E_conj/problems/{}'.format(pname))))

def process_lemma(args):
    hrr_class, hrr_size, pname, lname, lemma = args
    return (pname, lname,
            get_hrr_encoder(hrr_class, hrr_size).encode_disj(
                parse.parse_cnf_clause(lemma)))

@utils.persist_to_file('ex1data/hrrs-{}.pickle')
def get_hrrs(hrr_class, hrr_size):
    import multiprocessing as mp

    print('getting hrrs')
    # problem -> hrr
    problemhrrs = dict()
    # problem -> lemmaname -> hrr
    lemmahrrs = collections.defaultdict(dict)

    def get_lemmas():
        with open('E_conj/lemmas') as f:
            for l in f:
                name, lemma = l.split(':')
                _, problemname, lemmaname = name.split('/')
                yield (problemname, lemmaname, lemma)

    pool = mp.Pool()

    i = 0
    for pname, hrr in pool.imap_unordered(
            process_problem,
            ((hrr_class, hrr_size, pname) for pname in get_problemnames())):
        print(i, pname)
        i += 1
        problemhrrs[pname] = hrr

    i = 0
    for pname, lname, hrr in pool.imap_unordered(
            process_lemma,
            ((hrr_class, hrr_size, *args) for args in get_lemmas())):
        print(i, pname, lname)
        i += 1
        lemmahrrs[pname][lname] = hrr

    pool.close()

    return problemhrrs, lemmahrrs

def shuffle_by_hash(l, key=str):
    return sorted(l,
            key=lambda x:
                zlib.adler32(
                    key(x).encode('utf-8')
                    ) & 0xffffffff)

def get_training_data(hrr_class, hrr_size, training_size):
    problemhrrs, lemmahrrs = get_hrrs(hrr_class, hrr_size)
    usefulness = get_usefulness()

    # Segment training/test data by name
    problemlemmanames = shuffle_by_hash(get_lemmanames()[:-12000])[:training_size]
    # Segment training/test data by hashed name
    # problemlemmanames = shuffle_by_hash(get_lemmanames())[:training_size]

    trainingX = np.array([ np.concatenate((problemhrrs[problemname], lemmahrrs[problemname][lemmaname])) for problemname, lemmaname in problemlemmanames ])
    trainingy = np.array([ 1 if usefulness[problemname][lemmaname] < 1 else 0 for problemname, lemmaname in problemlemmanames ])

    scaler = pre.StandardScaler().fit(trainingX)

    trainingX_scaled = scaler.transform(trainingX)

    return scaler, trainingX_scaled, trainingy

def get_test_data(hrr_class, hrr_size, scaler, test_size):
    problemhrrs, lemmahrrs = get_hrrs(hrr_class, hrr_size)
    usefulness = get_usefulness()

    # Segment training/test data by name
    problemlemmanames = get_lemmanames()[-test_size:]
    # Segment training/test data by hashed name
    # problemlemmanames = shuffle_by_hash(get_lemmanames())[-test_size:]

    testX = np.array([ np.concatenate((problemhrrs[problemname], lemmahrrs[problemname][lemmaname])) for problemname, lemmaname in problemlemmanames ])
    testy = np.array([ 1 if usefulness[problemname][lemmaname] < 1 else 0 for problemname, lemmaname in problemlemmanames ])
    testXpos = np.array([ np.concatenate((problemhrrs[problemname], lemmahrrs[problemname][lemmaname])) for problemname, lemmaname in problemlemmanames if usefulness[problemname][lemmaname] < 1 ])
    testypos = np.array([ 1 for problemname, lemmaname in problemlemmanames if usefulness[problemname][lemmaname] < 1 ])
    testXneg = np.array([ np.concatenate((problemhrrs[problemname], lemmahrrs[problemname][lemmaname])) for problemname, lemmaname in problemlemmanames if usefulness[problemname][lemmaname] >= 1 ])
    testyneg = np.array([ 0 for problemname, lemmaname in problemlemmanames if usefulness[problemname][lemmaname] >= 1 ])

    testX_scaled = scaler.transform(testX)
    testXpos_scaled = scaler.transform(testXpos)
    testXneg_scaled = scaler.transform(testXneg)

    return testX_scaled, testy, testXpos_scaled, testypos, testXneg_scaled, testyneg

@utils.persist_to_file('ex1data/lrweights-{}.pickle')
def train_lr(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        params=lm.LogisticRegression(max_iter=5000).get_params()):
    """
    Trains a logistic regression model

    With training/test data split by name:

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.352   0.375      0.332   0.561
    64        5000           0.359   0.377      0.343   0.560
    64        10000          0.328   0.343      0.314   0.537
    64        50000          0.239   0.411      0.168   0.614
    64        -12000         0.148   0.413      0.090   0.627
    256       1000           0.384   0.374      0.396   0.545
    256       5000           0.396   0.385      0.407   0.553
    256       10000          0.362   0.358      0.366   0.537
    256       50000          0.354   0.410      0.312   0.591
    256       -12000         0.336   0.424      0.279   0.605
    1024      1000           0.399   0.359      0.448   0.514
    1024      5000           0.379   0.365      0.394   0.536
    1024      10000          0.439   0.385      0.510   0.531
    1024      50000          0.387   0.356      0.424   0.517
    1024      -12000         0.378   0.387      0.369   0.563

    With training/test data split by hashed name:

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.408   0.435      0.385   0.618
    64        5000           0.380   0.351      0.413   0.537
    64        10000          0.325   0.308      0.343   0.511
    64        50000          0.161   0.276      0.113   0.594
    64        -12000         0.119   0.353      0.071   0.637
    256       1000           0.280   0.276      0.283   0.500
    256       5000           0.378   0.330      0.442   0.501
    256       10000          0.286   0.272      0.301   0.485
    256       50000          0.265   0.293      0.242   0.540
    256       -12000         0.333   0.462      0.261   0.642
    1024      1000           0.336   0.303      0.375   0.490
    1024      5000           0.356   0.316      0.406   0.495
    1024      10000          0.360   0.295      0.462   0.438
    1024      50000          0.330   0.290      0.382   0.467
    1024      -12000         0.487   0.543      0.442   0.681
    """


    scaler, trainingX_scaled, trainingy = get_training_data(hrr_size, training_size)

    model = lm.LogisticRegression(max_iter=5000)
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/pacweights-{}.pickle')
def train_pac(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        params=lm.PassiveAggressiveClassifier().get_params()):
    """
    Trains a passive aggressive classifier

    With training/test split by name:

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.333   0.356      0.313   0.550
    64        5000           0.386   0.371      0.402   0.541
    64        10000          0.405   0.349      0.482   0.492
    64        50000          0.380   0.381      0.380   0.556
    64        -12000         0.332   0.385      0.292   0.578
    256       1000           0.374   0.365      0.384   0.539
    256       5000           0.394   0.370      0.421   0.534
    256       10000          0.387   0.371      0.405   0.540
    256       50000          0.441   0.390      0.507   0.538
    256       -12000         0.401   0.380      0.424   0.545
    1024      1000           0.423   0.353      0.528   0.482
    1024      5000           0.370   0.370      0.370   0.548
    1024      10000          0.394   0.369      0.421   0.534
    1024      50000          0.388   0.376      0.401   0.546
    1024      -12000         0.401   0.383      0.420   0.549

    With training/test split by hashed name:

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.410   0.418      0.403   0.603
    64        5000           0.360   0.310      0.429   0.477
    64        10000          0.378   0.321      0.461   0.480
    64        50000          0.437   0.395      0.490   0.568
    64        -12000         0.392   0.371      0.417   0.557
    256       1000           0.292   0.283      0.302   0.497
    256       5000           0.385   0.343      0.437   0.520
    256       10000          0.390   0.340      0.457   0.509
    256       50000          0.382   0.364      0.402   0.554
    256       -12000         0.353   0.313      0.405   0.491
    1024      1000           0.351   0.283      0.460   0.416
    1024      5000           0.323   0.283      0.377   0.459
    1024      10000          0.332   0.298      0.375   0.483
    1024      50000          0.373   0.333      0.424   0.512
    1024      -12000         0.448   0.505      0.402   0.660
    """

    scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)

    model = lm.PassiveAggressiveClassifier()
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/svcweights-{}.pickle')
def train_svc(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        params=svm.SVC().get_params()):
    """
    Trains a support vector machine classifier

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.319   0.411      0.261   0.600
    64        5000           0.258   0.338      0.209   0.569
    64        10000          0.282   0.371      0.228   0.584
    64        50000          0.275   0.429      0.202   0.617
    64        -12000         0.323   0.467      0.247   0.628
    1024      1000           0.254   0.419      0.182   0.616
    1024      5000           0.306   0.379      0.257   0.582
    1024      10000          0.282   0.360      0.232   0.576
    """

    print('Training SVC', hrr_size, training_size, params)

    scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)

    model = svm.SVC()
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

def test_model(hrr_class, hrr_size, scaler, model):
    """
    Tests a model
    """

    testX_scaled, testy, testXpos_scaled, testypos, testXneg_scaled, testyneg = get_test_data(hrr_class, hrr_size, scaler, 12000)

    acc = model.score(testX_scaled, testy)

    posacc = model.score(testXpos_scaled, testypos)
    negacc = model.score(testXneg_scaled, testyneg)

    precision = posacc * len(testypos) / (posacc * len(testypos) + (1-negacc) * len(testyneg))
    recall = posacc
    fscore = 2 * precision * recall / (precision + recall)

    # print('fscore, precision, recall, acc')
    # print(fscore, precision, recall, acc)
    return fscore, precision, recall, acc

for model_train in [
        train_lr,
        train_pac,
        # train_svc,
        ]:
    print('Training', model_train.__name__)
    print('hrr_size, training_size, fscore, precision, recall, acc')
    for hrr_size in [
            # 16,
            # 32,
            64,
            256,
            1024,
            ]:
        for training_size in [
                1000,
                5000,
                10000,
                50000,
                -12000,
                ]:
            scaler, model = model_train(hrr_size, training_size)
            print(hrr_size, training_size,
                    *('{:0.3f}'.format(x)
                        for x in test_model(HRR.FlatTreeHRR, hrr_size, scaler, model)))
