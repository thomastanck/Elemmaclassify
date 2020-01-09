import HRR
import parse
import utils
import lemmadata

import sys
import collections
import functools
import zlib

import numpy as np
import sklearn as sk

import sklearn.svm as svm
import sklearn.preprocessing as pre
import sklearn.linear_model as lm

import xgboost

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

@utils.persist_to_file('ex1data/usefulness.pickle', key=lambda *params: str(params))
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
    return HRR.FlatTreeHRR.fold_term(parse_problem(problemname))

def lemma_hrr(lemmastr):
    print('getting lemma hrr for {}'.format(lemmastr[:30]))
    return HRR.FlatTreeHRR.fold_term(parse.parse_cnf_clause(lemmastr))

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

@functools.lru_cache(maxsize=1)
def get_lemmanamesindex(num_hrrs=1):
    lemmanames = set()
    with open('E_conj/lemmas') as f:
        for l in f:
            name, lemma = l.split(':')
            _, problemname, lemmaname = name.split('/')
            for index in range(num_hrrs):
                lemmanames.add((problemname, lemmaname, index))
    return sorted(list(lemmanames))

@functools.lru_cache(maxsize=None)
def get_hrr_encoder(hrr_class, hrr_size, **hrr_args):
    return hrr_class(hrr_size, **hrr_args)

def process_problem(args):
    hrr_class, hrr_size, hrr_args, pname = args
    return (pname,
            get_hrr_encoder(hrr_class, hrr_size, **hrr_args).fold_term(
                parse.parse_cnf_file('E_conj/problems/{}'.format(pname))))

def process_lemma(args):
    hrr_class, hrr_size, hrr_args, pname, lname, lemma = args
    return (pname, lname,
            get_hrr_encoder(hrr_class, hrr_size, **hrr_args).fold_term(
                parse.parse_cnf_clause(lemma)))

@utils.persist_to_file('ex1data/comphrrs-{1}.pickle', key=lambda *params: str(params))
def get_hrrs_complex(hrr_class, hrr_size):
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
            ((hrr_class, hrr_size, {'complex_hrr': True}, pname) for pname in get_problemnames())):
        print(i, pname)
        i += 1
        hrr = np.concatenate((hrr.real, hrr.imag))
        problemhrrs[pname] = hrr

    i = 0
    for pname, lname, hrr in pool.imap_unordered(
            process_lemma,
            ((hrr_class, hrr_size, {'complex_hrr': True}, *args) for args in get_lemmas())):
        print(i, pname, lname)
        i += 1
        hrr = np.concatenate((hrr.real, hrr.imag))
        lemmahrrs[pname][lname] = hrr

    pool.close()

    return problemhrrs, lemmahrrs

@utils.persist_to_file('ex1data/hrrs-{1}.pickle', key=lambda *params: str(params))
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
            ((hrr_class, hrr_size, {}, pname) for pname in get_problemnames())):
        print(i, pname)
        i += 1
        problemhrrs[pname] = hrr

    i = 0
    for pname, lname, hrr in pool.imap_unordered(
            process_lemma,
            ((hrr_class, hrr_size, {}, *args) for args in get_lemmas())):
        print(i, pname, lname)
        i += 1
        lemmahrrs[pname][lname] = hrr

    pool.close()

    return problemhrrs, lemmahrrs

def process_problem_lemma_pair(args):
    hrr_class, hrr_size, hrr_args, pname, lname, problem, lemma = args
    problem = parse.parse_cnf_file('E_conj/problems/{}'.format(pname))
    hrr_enc = hrr_class(hrr_size, **hrr_args)
    hrr_enc.cache_clear()
    phrr = hrr_enc.fold_term(problem)
    lhrr = hrr_enc.fold_term(lemma)
    return (pname, lname, phrr, lhrr)

def process_problem_lemma_pair_complex(args):
    hrr_class, hrr_size, hrr_args, pname, lname, problem, lemma = args
    problem = parse.parse_cnf_file('E_conj/problems/{}'.format(pname))
    hrr_enc = hrr_class(hrr_size, **hrr_args)
    hrr_enc.cache_clear()
    phrr = hrr_enc.fold_term(problem)
    lhrr = hrr_enc.fold_term(lemma)
    phrr = np.concatenate((phrr.real, phrr.imag))
    lhrr = np.concatenate((lhrr.real, lhrr.imag))
    return (pname, lname, phrr, lhrr)

@utils.persist_to_file('ex1data/hrrs-pairs-{1}-{2}-{3}.pickle', key=lambda *params: str((params[0], params[1], tuple(sorted(params[2].items())))))
def get_hrrs_pairs(hrr_class, hrr_size, hrr_args=dict(), index=0):
    import multiprocessing as mp

    print('getting hrrs (pairs)', hrr_class, hrr_size, hrr_args)

    complex = False
    if 'complex_hrr' in hrr_args and hrr_args['complex_hrr']:
        complex = True

    pool = mp.Pool()

    def get_lemmas():
        with open('E_conj/lemmas') as f:
            for l in f:
                name, lemma = l.split(':')
                _, problemname, lemmaname = name.split('/')
                yield (problemname, lemmaname, lemma)

    pairshrrs = collections.defaultdict(list)

    i = 0
    for pname, lname, phrr, lhrr in pool.imap_unordered(
            process_problem_lemma_pair_complex if complex else process_problem_lemma_pair,
            ((hrr_class, hrr_size, hrr_args, *args) for args in lemmadata.get_problemslemmas())
            ):
        print(i, pname, lname)
        i += 1
        pairshrrs[(pname, lname)].append((phrr, lhrr))

    pool.close()

    return pairshrrs

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
    # problemlemmanames = shuffle_by_hash(get_lemmanames()[:-12000])[:training_size]
    # Segment training/test data by hashed name
    problemlemmanames = shuffle_by_hash(get_lemmanames())[:training_size]

    trainingX = np.array([ np.concatenate((problemhrrs[problemname], lemmahrrs[problemname][lemmaname])) for problemname, lemmaname in problemlemmanames ])
    trainingy = np.array([ 1 if usefulness[problemname][lemmaname] < 1 else 0 for problemname, lemmaname in problemlemmanames ])

    scaler = pre.StandardScaler().fit(trainingX)

    trainingX_scaled = scaler.transform(trainingX)

    return scaler, trainingX_scaled, trainingy

def get_test_data(hrr_class, hrr_size, scaler, test_size):
    problemhrrs, lemmahrrs = get_hrrs(hrr_class, hrr_size)
    usefulness = get_usefulness()

    # Segment training/test data by name
    # problemlemmanames = get_lemmanames()[-test_size:]
    # Segment training/test data by hashed name
    problemlemmanames = shuffle_by_hash(get_lemmanames())[-test_size:]

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

def get_training_data_pairs(hrr_class, hrr_size, training_size, shuffle, num_hrrs, hrr_args):
    pairshrrs = [ get_hrrs_pairs(hrr_class, hrr_size, hrr_args, index) for index in range(num_hrrs) ]
    usefulness = get_usefulness()

    if shuffle:
        # Segment training/test data by hashed name
        problemlemmanames = shuffle_by_hash(get_lemmanamesindex())[:training_size]
    else:
        # Segment training/test data by name
        problemlemmanames = shuffle_by_hash(get_lemmanamesindex()[:-12000*num_hrrs])[:training_size]

    trainingX = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] ])
    trainingy = np.array([ 1 if usefulness[problemname][lemmaname] < 1 else 0 for problemname, lemmaname, index in problemlemmanames ])

    scaler = pre.StandardScaler().fit(trainingX)

    trainingX_scaled = scaler.transform(trainingX)

    return scaler, trainingX_scaled, trainingy

def get_test_data_pairs(hrr_class, hrr_size, scaler, test_size, shuffle, num_hrrs, hrr_args):
    pairshrrs = [ get_hrrs_pairs(hrr_class, hrr_size, hrr_args, index) for index in range(num_hrrs) ]
    usefulness = get_usefulness()

    if shuffle:
        # Segment training/test data by hashed name
        problemlemmanames = shuffle_by_hash(get_lemmanamesindex())[-test_size:]
    else:
        # Segment training/test data by name
        problemlemmanames = get_lemmanamesindex()[-test_size:]

    testX = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] ])
    testy = np.array([ 1 if usefulness[problemname][lemmaname] < 1 else 0 for problemname, lemmaname, index in problemlemmanames ])
    testXpos = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] if usefulness[problemname][lemmaname] < 1 ])
    testypos = np.array([ 1 for problemname, lemmaname, index in problemlemmanames if usefulness[problemname][lemmaname] < 1 ])
    testXneg = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] if usefulness[problemname][lemmaname] >= 1 ])
    testyneg = np.array([ 0 for problemname, lemmaname, index in problemlemmanames if usefulness[problemname][lemmaname] >= 1 ])

    testX_scaled = scaler.transform(testX)
    testXpos_scaled = scaler.transform(testXpos)
    testXneg_scaled = scaler.transform(testXneg)

    return testX_scaled, testy, testXpos_scaled, testypos, testXneg_scaled, testyneg

@utils.persist_to_file('ex1data/lrweights-{}.pickle', key=lambda *params: str(tuple(tuple(sorted(p.items())) if isinstance(p, dict) else p for p in params)))
def train_lr(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        shuffle=False,
        hrr_args=dict(),
        num_hrrs=1,
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

    With training/test data split by name, randomised variables and skolem constants:

    With training/test data split by name, complex hrrs, randomised variables and skolem constants:

    With training/test data split by hashed name, randomised variables and skolem constants:

    With training/test data split by hashed name, complex hrrs, randomised variables and skolem constants:
    """

    params = dict(params)

    print('Training LR', hrr_size, training_size, params)

    # scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)
    scaler, trainingX_scaled, trainingy = get_training_data_pairs(
            hrr_class,
            hrr_size,
            training_size,
            shuffle,
            num_hrrs,
            hrr_args)

    model = lm.LogisticRegression()
    model.set_params(**params)
    model.set_params(n_jobs=4)
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/pacweights-{}.pickle', key=lambda *params: str(tuple(tuple(sorted(p.items())) if isinstance(p, dict) else p for p in params)))
def train_pac(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        shuffle=False,
        hrr_args=dict(),
        num_hrrs=1,
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

    params = dict(params)

    print('Training PAC', hrr_size, training_size, params)

    # scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)
    scaler, trainingX_scaled, trainingy = get_training_data_pairs(
            hrr_class,
            hrr_size,
            training_size,
            shuffle,
            num_hrrs,
            hrr_args)

    model = lm.PassiveAggressiveClassifier()
    model.set_params(**params)
    model.set_params(n_jobs=4)
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/svcweights-{}.pickle', key=lambda *params: str(tuple(tuple(sorted(p.items())) if isinstance(p, dict) else p for p in params)))
def train_svc(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        shuffle=False,
        hrr_args=dict(),
        num_hrrs=1,
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

    params = dict(params)

    print('Training SVC', hrr_size, training_size, params)

    # scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)
    scaler, trainingX_scaled, trainingy = get_training_data_pairs(
            hrr_class,
            hrr_size,
            training_size,
            shuffle,
            num_hrrs,
            hrr_args)

    model = svm.SVC()
    model.set_params(**params)
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/xgbweights-{}.pickle', key=lambda *params: str(tuple(tuple(sorted(p.items())) if isinstance(p, dict) else p for p in params)))
def train_xgb(
        hrr_size=1024,
        training_size=-12000,
        hrr_class=HRR.FlatTreeHRR,
        shuffle=False,
        hrr_args=dict(),
        num_hrrs=1,
        params=tuple(sorted(list(xgboost.XGBClassifier().get_params().items())))):
    """
    Trains a XGBoost classifier

    with max_depth=3 (default)

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.286   0.374      0.231   0.585
    64        5000           0.287   0.404      0.222   0.603
    64        10000          0.277   0.387      0.215   0.596
    64        50000          0.122   0.496      0.070   0.640
    64        -12000         0.082   0.439      0.045   0.636

    with max_depth=32

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.313   0.389      0.262   0.587
    64        5000           0.319   0.381      0.273   0.580
    64        10000          0.314   0.390      0.262   0.588
    64        50000          0.297   0.420      0.230   0.609
    64        -12000         0.238   0.462      0.160   0.631
    256       1000           0.289   0.400      0.226   0.600
    256       5000           0.227   0.410      0.157   0.616
    256       10000          0.293   0.333      0.262   0.546
    256       50000          0.271   0.432      0.197   0.618
    256       -12000         0.230   0.442      0.155   0.626
    1024      1000           0.294   0.351      0.253   0.564
    1024      5000           0.248   0.382      0.184   0.600
    1024      10000          0.295   0.371      0.244   0.580
    1024      50000          0.257   0.443      0.181   0.624
    1024      -12000         0.256   0.488      0.174   0.638

    with max_depth=32, mixed train/test

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.255   0.358      0.198   0.603
    64        5000           0.368   0.470      0.303   0.644
    64        10000          0.410   0.427      0.393   0.611
    64        50000          0.263   0.383      0.200   0.615
    64        -12000         0.525   0.562      0.493   0.694
    256       1000           0.223   0.403      0.154   0.631
    256       5000           0.213   0.346      0.154   0.610
    256       10000          0.233   0.302      0.189   0.572
    256       50000          0.174   0.263      0.130   0.577
    256       -12000         0.573   0.689      0.491   0.749
    1024      1000           0.338   0.449      0.272   0.636
    1024      5000           0.221   0.308      0.173   0.583
    1024      10000          0.206   0.222      0.193   0.490
    1024      50000          0.160   0.294      0.110   0.604
    1024      -12000         0.574   0.705      0.485   0.754

    with max_depth=32, complex HRR (i think this is nonmixed, but can't remember. rerunning anyway.)

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.208   0.338      0.151   0.607
    64        5000           0.346   0.407      0.300   0.610
    64        10000          0.286   0.360      0.237   0.594
    64        50000          0.140   0.279      0.093   0.606
    64        -12000         0.594   0.700      0.515   0.758
    256       1000           0.184   0.300      0.133   0.597
    256       5000           0.224   0.252      0.201   0.521
    256       10000          0.289   0.335      0.254   0.571
    256       50000          0.276   0.396      0.212   0.619
    256       -12000         0.522   0.589      0.468   0.706
    1024      1000           0.237   0.370      0.174   0.615
    1024      5000           0.264   0.318      0.225   0.569
    1024      10000          0.252   0.284      0.226   0.539

    with max_depth=32, complex HRR, separate train/test

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.265   0.355      0.212   0.579
    64        5000           0.244   0.376      0.181   0.598
    64        10000          0.283   0.368      0.230   0.581
    64        50000          0.254   0.449      0.177   0.626
    64        -12000         0.292   0.497      0.206   0.640
    256       1000           0.231   0.327      0.178   0.573
    256       5000           0.339   0.361      0.320   0.552
    256       10000          0.339   0.378      0.308   0.569
    256       50000          0.257   0.427      0.184   0.618
    256       -12000         0.272   0.452      0.195   0.626
    1024      1000           0.220   0.336      0.164   0.583
    1024      5000           0.217   0.346      0.158   0.591
    1024      10000          0.347   0.426      0.293   0.604
    1024      50000          0.250   0.449      0.173   0.627
    1024      -12000         0.256   0.500      0.172   0.641

    max_depth=32, complex, mixed train/test

    hrr_size, training_size, fscore, precision, recall, acc
    64        1000           0.246   0.403      0.177   0.628
    64        5000           0.290   0.459      0.212   0.644
    64        10000          0.262   0.307      0.228   0.559
    64        50000          0.246   0.362      0.186   0.609
    64        -12000         0.571   0.705      0.480   0.753
    256       1000           0.163   0.398      0.102   0.639
    256       5000           0.247   0.350      0.191   0.601
    256       10000          0.244   0.282      0.216   0.542
    256       50000          0.174   0.242      0.136   0.557
    256       -12000         0.572   0.682      0.492   0.747
    1024      1000           0.141   0.273      0.095   0.603
    1024      5000           0.204   0.298      0.155   0.585
    1024      10000          0.230   0.294      0.190   0.566
    1024      50000          0.260   0.398      0.193   0.623
    1024      -12000         0.573   0.693      0.488   0.750
    """

    params = dict(params)

    print('Training XGB', hrr_size, training_size, params)

    # scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)
    scaler, trainingX_scaled, trainingy = get_training_data_pairs(
            hrr_class,
            hrr_size,
            training_size,
            shuffle,
            num_hrrs,
            hrr_args)

    model = xgboost.XGBClassifier()
    model.set_params(**params)
    model.set_params(n_jobs=4)
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

def test_model(hrr_class, hrr_size, scaler, model, shuffle, num_hrrs, hrr_args):
    """
    Tests a model
    """

    # testX_scaled, testy, testXpos_scaled, testypos, testXneg_scaled, testyneg = get_test_data(hrr_class, hrr_size, scaler, 12000)
    testX_scaled, testy, testXpos_scaled, testypos, testXneg_scaled, testyneg = get_test_data_pairs(
            hrr_class,
            hrr_size,
            scaler,
            12000,
            shuffle,
            num_hrrs,
            hrr_args)

    acc = model.score(testX_scaled, testy)

    posacc = model.score(testXpos_scaled, testypos)
    negacc = model.score(testXneg_scaled, testyneg)

    precision = 0 if posacc * len(testypos) == 0 else posacc * len(testypos) / (posacc * len(testypos) + (1-negacc) * len(testyneg))
    recall = posacc
    fscore = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)

    # print('fscore, precision, recall, acc')
    # print(fscore, precision, recall, acc)
    return fscore, precision, recall, acc

## Preprocessing

# args = [ (hrr_size, hrr_args, index)
#     for hrr_size in [
#             16,
#             32,
#             64,
#             256,
#             # 1024,
#             ]
#     for hrr_args in [
#             {'complex_hrr': True,  'randomise_variables': True,  'randomise_skolem_constants': True, },
#             {'complex_hrr': False, 'randomise_variables': True,  'randomise_skolem_constants': True, },
#             {'complex_hrr': True,  'randomise_variables': False, 'randomise_skolem_constants': False, },
#             {'complex_hrr': False, 'randomise_variables': False, 'randomise_skolem_constants': False, },
#                 ]
#     for index in range(1)
#     ]
# print(sys.argv[1], 'out of', len(args))
# actualargs = args[int(sys.argv[1])]
# print(actualargs)
# get = get_hrrs_pairs(HRR.FlatTreeHRR, actualargs[0], actualargs[1], actualargs[2])

## Training/testing

num_hrrs = 1
hrr_class = HRR.FlatTreeHRR
for shuffle in [False, True]:
    for model_train, params in [
            # (train_lr, tuple(sorted(list(lm.LogisticRegression(max_iter=5000).get_params().items())))),
            # (train_pac, tuple(sorted(list(lm.PassiveAggressiveClassifier().get_params().items())))),
            # # (train_svc, tuple(sorted(list(svm.SVC().get_params().items())))),
            # (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32).get_params().items())))),
            (train_lr, tuple(sorted(list(lm.LogisticRegression(max_iter=5000, class_weight={False: 1, True: 10}).get_params().items())))),
            (train_pac, tuple(sorted(list(lm.PassiveAggressiveClassifier(class_weight={False: 1, True: 10}).get_params().items())))),
            # (train_svc, tuple(sorted(list(svm.SVC(class_weight={False: 1, True: 10}).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, class_weight={False: 1, True: 10}).get_params().items())))),
            ]:
        for hrr_args in [
                {'complex_hrr': True,  'randomise_variables': True,  'randomise_skolem_constants': True, },
                {'complex_hrr': False, 'randomise_variables': True,  'randomise_skolem_constants': True, },
                {'complex_hrr': True,  'randomise_variables': False, 'randomise_skolem_constants': False, },
                {'complex_hrr': False, 'randomise_variables': False, 'randomise_skolem_constants': False, },
                    ]:
            print('Training', model_train.__name__, 'complex' if hrr_args['complex_hrr'] else '', 'random' if hrr_args['randomise_variables'] else '', 'mixed test/train' if shuffle else 'nonmixed test/train')
            print('hrr_size, training_size, fscore, precision, recall, acc')
            for hrr_size in [
                    16,
                    32,
                    64,
                    # 256,
                    # 1024,
                    ]:
                for training_size in [
                        100 * num_hrrs,
                        300 * num_hrrs,
                        1000 * num_hrrs,
                        5000 * num_hrrs,
                        10000 * num_hrrs,
                        50000 * num_hrrs,
                        -12000 * num_hrrs,
                        ]:
                    scaler, model = model_train(hrr_size, training_size, hrr_class, shuffle, hrr_args, num_hrrs, params)
                    print(hrr_size, training_size,
                            *('{:0.3f}'.format(x)
                                for x in test_model(HRR.FlatTreeHRR, hrr_size, scaler, model, shuffle, num_hrrs, hrr_args)))
            print()

"""

Training train_xgb complex random nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.302   0.367      0.257   0.574
16        300            0.292   0.359      0.245   0.572
16        1000           0.267   0.376      0.207   0.592
16        5000           0.209   0.403      0.141   0.616
16        10000          0.296   0.384      0.240   0.589
16        50000          0.223   0.418      0.152   0.619
16        -12000         0.193   0.429      0.125   0.626
32        100            0.281   0.388      0.220   0.595
32        300            0.199   0.354      0.138   0.600
32        1000           0.313   0.375      0.268   0.576
32        5000           0.224   0.359      0.163   0.595
32        10000          0.312   0.375      0.267   0.577
32        50000          0.234   0.423      0.161   0.620
32        -12000         0.187   0.410      0.121   0.622
64        100            0.240   0.388      0.174   0.605
64        300            0.243   0.401      0.174   0.610
64        1000           0.266   0.368      0.208   0.587
64        5000           0.253   0.397      0.185   0.606
64        10000          0.305   0.395      0.248   0.593
64        50000          0.241   0.443      0.166   0.625
64        -12000         0.194   0.438      0.125   0.628

Training train_xgb  random nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.426   0.377      0.490   0.525
16        300            0.255   0.362      0.197   0.587
16        1000           0.309   0.359      0.271   0.564
16        5000           0.233   0.396      0.165   0.610
16        10000          0.298   0.380      0.245   0.585
16        50000          0.210   0.397      0.143   0.614
16        -12000         0.181   0.399      0.117   0.620
32        100            0.335   0.359      0.314   0.552
32        300            0.175   0.364      0.115   0.610
32        1000           0.283   0.371      0.228   0.584
32        5000           0.205   0.375      0.141   0.607
32        10000          0.266   0.368      0.208   0.588
32        50000          0.167   0.367      0.108   0.613
32        -12000         0.151   0.377      0.095   0.619
64        100            0.265   0.379      0.204   0.594
64        300            0.248   0.370      0.186   0.594
64        1000           0.304   0.372      0.257   0.577
64        5000           0.205   0.361      0.143   0.601
64        10000          0.273   0.362      0.220   0.581
64        50000          0.208   0.406      0.140   0.617
64        -12000         0.161   0.392      0.102   0.621

Training train_xgb complex  nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.292   0.346      0.252   0.560
16        300            0.216   0.356      0.155   0.596
16        1000           0.271   0.401      0.205   0.604
16        5000           0.358   0.421      0.311   0.599
16        10000          0.377   0.406      0.352   0.582
16        50000          0.312   0.482      0.230   0.634
16        -12000         0.302   0.488      0.219   0.637
32        100            0.294   0.341      0.258   0.555
32        300            0.227   0.338      0.171   0.582
32        1000           0.284   0.361      0.234   0.576
32        5000           0.285   0.363      0.235   0.577
32        10000          0.364   0.419      0.322   0.596
32        50000          0.283   0.413      0.215   0.608
32        -12000         0.262   0.483      0.180   0.636
64        100            0.294   0.404      0.231   0.602
64        300            0.283   0.367      0.231   0.581
64        1000           0.322   0.371      0.285   0.569
64        5000           0.248   0.380      0.184   0.599
64        10000          0.290   0.378      0.235   0.586
64        50000          0.241   0.438      0.166   0.624
64        -12000         0.255   0.446      0.179   0.625

Training train_xgb   nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.315   0.356      0.283   0.559
16        300            0.304   0.384      0.251   0.586
16        1000           0.305   0.369      0.259   0.575
16        5000           0.237   0.377      0.173   0.600
16        10000          0.357   0.400      0.323   0.583
16        50000          0.263   0.393      0.197   0.602
16        -12000         0.228   0.414      0.158   0.617
32        100            0.259   0.389      0.194   0.601
32        300            0.198   0.365      0.135   0.605
32        1000           0.305   0.379      0.255   0.582
32        5000           0.329   0.383      0.288   0.578
32        10000          0.331   0.377      0.294   0.572
32        50000          0.269   0.435      0.195   0.620
32        -12000         0.262   0.462      0.183   0.630
64        100            0.242   0.357      0.183   0.588
64        300            0.283   0.397      0.220   0.600
64        1000           0.350   0.390      0.318   0.576
64        5000           0.304   0.406      0.243   0.600
64        10000          0.343   0.396      0.303   0.583
64        50000          0.301   0.416      0.236   0.607
64        -12000         0.230   0.444      0.155   0.627

Training train_lr complex random nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.374   0.373      0.374   0.549
16        300            0.459   0.371      0.601   0.491
16        1000           0.514   0.366      0.859   0.416
16        5000           0.529   0.360      0.997   0.361
16        10000          0.529   0.359      1.000   0.359
16        50000          0.529   0.359      1.000   0.359
16        -12000         0.529   0.359      1.000   0.359
32        100            0.271   0.352      0.220   0.574
32        300            0.378   0.353      0.407   0.519
32        1000           0.473   0.356      0.707   0.434
32        5000           0.528   0.360      0.986   0.366
32        10000          0.529   0.359      1.000   0.360
32        50000          0.529   0.359      1.000   0.359
32        -12000         0.529   0.359      1.000   0.359
64        100            0.266   0.373      0.207   0.590
64        300            0.448   0.381      0.543   0.519
64        1000           0.424   0.374      0.489   0.522
64        5000           0.521   0.366      0.902   0.404
64        10000          0.527   0.360      0.986   0.365
64        50000          0.529   0.360      0.999   0.361
64        -12000         0.529   0.359      1.000   0.359

Training train_lr  random nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.472   0.371      0.647   0.479
16        300            0.498   0.370      0.760   0.450
16        1000           0.520   0.360      0.932   0.381
16        5000           0.529   0.359      1.000   0.359
16        10000          0.529   0.359      1.000   0.359
16        50000          0.529   0.359      1.000   0.359
16        -12000         0.529   0.359      1.000   0.359
32        100            0.408   0.355      0.480   0.500
32        300            0.435   0.363      0.542   0.494
32        1000           0.496   0.361      0.790   0.423
32        5000           0.528   0.359      0.999   0.360
32        10000          0.529   0.359      1.000   0.359
32        50000          0.529   0.359      1.000   0.359
32        -12000         0.529   0.359      1.000   0.359
64        100            0.396   0.370      0.427   0.533
64        300            0.417   0.366      0.484   0.513
64        1000           0.459   0.353      0.655   0.445
64        5000           0.528   0.360      0.990   0.363
64        10000          0.529   0.359      1.000   0.359
64        50000          0.529   0.359      1.000   0.359
64        -12000         0.529   0.359      1.000   0.359

Training train_lr complex  nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.389   0.360      0.424   0.523
16        300            0.448   0.380      0.546   0.517
16        1000           0.443   0.364      0.567   0.489
16        5000           0.491   0.349      0.829   0.382
16        10000          0.525   0.359      0.977   0.365
16        50000          0.529   0.360      1.000   0.361
16        -12000         0.529   0.359      1.000   0.359
32        100            0.373   0.335      0.420   0.492
32        300            0.400   0.337      0.492   0.469
32        1000           0.381   0.339      0.435   0.492
32        5000           0.471   0.342      0.752   0.392
32        10000          0.506   0.353      0.892   0.373
32        50000          0.530   0.361      0.997   0.364
32        -12000         0.528   0.359      0.999   0.359
64        100            0.377   0.348      0.411   0.512
64        300            0.412   0.357      0.485   0.502
64        1000           0.461   0.373      0.604   0.493
64        5000           0.417   0.349      0.519   0.479
64        10000          0.479   0.356      0.735   0.427
64        50000          0.531   0.363      0.987   0.373
64        -12000         0.530   0.360      0.998   0.363

Training train_lr   nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.362   0.348      0.377   0.523
16        300            0.431   0.366      0.526   0.502
16        1000           0.464   0.348      0.697   0.422
16        5000           0.528   0.361      0.980   0.370
16        10000          0.529   0.360      0.999   0.362
16        50000          0.529   0.359      1.000   0.359
16        -12000         0.529   0.359      1.000   0.359
32        100            0.303   0.339      0.274   0.547
32        300            0.456   0.351      0.652   0.442
32        1000           0.464   0.371      0.619   0.486
32        5000           0.522   0.365      0.917   0.397
32        10000          0.522   0.357      0.972   0.361
32        50000          0.528   0.359      0.997   0.360
32        -12000         0.529   0.359      1.000   0.359
64        100            0.181   0.282      0.133   0.567
64        300            0.435   0.374      0.519   0.515
64        1000           0.455   0.360      0.617   0.468
64        5000           0.486   0.354      0.774   0.411
64        10000          0.509   0.355      0.896   0.379
64        50000          0.529   0.360      0.997   0.363
64        -12000         0.528   0.359      0.999   0.359

Training train_pac complex random nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.374   0.370      0.379   0.545
16        300            0.463   0.350      0.684   0.429
16        1000           0.492   0.361      0.769   0.428
16        5000           0.480   0.353      0.753   0.415
16        10000          0.517   0.363      0.897   0.398
16        50000          0.475   0.355      0.715   0.431
16        -12000         0.512   0.366      0.852   0.417
32        100            0.336   0.347      0.326   0.537
32        300            0.433   0.362      0.538   0.493
32        1000           0.473   0.350      0.726   0.417
32        5000           0.484   0.354      0.762   0.415
32        10000          0.501   0.363      0.812   0.420
32        50000          0.504   0.360      0.842   0.405
32        -12000         0.510   0.371      0.817   0.437
64        100            0.370   0.380      0.361   0.559
64        300            0.456   0.368      0.598   0.487
64        1000           0.451   0.366      0.586   0.487
64        5000           0.494   0.372      0.734   0.460
64        10000          0.511   0.362      0.871   0.401
64        50000          0.498   0.361      0.804   0.417
64        -12000         0.501   0.359      0.831   0.406

Training train_pac  random nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.484   0.357      0.747   0.426
16        300            0.502   0.365      0.803   0.428
16        1000           0.507   0.357      0.875   0.390
16        5000           0.503   0.357      0.853   0.394
16        10000          0.464   0.359      0.657   0.455
16        50000          0.512   0.355      0.923   0.369
16        -12000         0.521   0.358      0.954   0.369
32        100            0.408   0.354      0.481   0.499
32        300            0.472   0.368      0.657   0.472
32        1000           0.493   0.362      0.773   0.429
32        5000           0.502   0.363      0.813   0.421
32        10000          0.502   0.359      0.835   0.406
32        50000          0.504   0.356      0.860   0.391
32        -12000         0.504   0.364      0.822   0.419
64        100            0.433   0.372      0.518   0.513
64        300            0.470   0.359      0.683   0.447
64        1000           0.438   0.344      0.600   0.446
64        5000           0.499   0.361      0.807   0.419
64        10000          0.514   0.362      0.889   0.397
64        50000          0.495   0.356      0.814   0.404
64        -12000         0.500   0.370      0.770   0.447

Training train_pac complex  nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.387   0.348      0.436   0.504
16        300            0.457   0.396      0.541   0.538
16        1000           0.472   0.362      0.676   0.457
16        5000           0.471   0.345      0.741   0.402
16        10000          0.470   0.353      0.704   0.431
16        50000          0.504   0.368      0.799   0.434
16        -12000         0.503   0.364      0.815   0.422
32        100            0.363   0.340      0.388   0.510
32        300            0.463   0.350      0.683   0.431
32        1000           0.415   0.356      0.497   0.497
32        5000           0.473   0.353      0.715   0.427
32        10000          0.496   0.353      0.837   0.390
32        50000          0.508   0.367      0.825   0.426
32        -12000         0.498   0.365      0.784   0.433
64        100            0.253   0.335      0.203   0.569
64        300            0.468   0.369      0.642   0.477
64        1000           0.479   0.384      0.634   0.503
64        5000           0.459   0.357      0.641   0.457
64        10000          0.475   0.357      0.712   0.436
64        50000          0.486   0.358      0.759   0.423
64        -12000         0.503   0.367      0.799   0.433

Training train_pac   nonmixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.414   0.366      0.478   0.514
16        300            0.455   0.353      0.639   0.450
16        1000           0.479   0.353      0.745   0.418
16        5000           0.502   0.363      0.810   0.421
16        10000          0.521   0.361      0.938   0.382
16        50000          0.512   0.360      0.888   0.391
16        -12000         0.510   0.367      0.837   0.423
32        100            0.193   0.298      0.142   0.572
32        300            0.450   0.349      0.634   0.443
32        1000           0.432   0.386      0.491   0.537
32        5000           0.510   0.366      0.842   0.418
32        10000          0.516   0.367      0.867   0.415
32        50000          0.520   0.370      0.873   0.421
32        -12000         0.496   0.356      0.820   0.402
64        100            0.172   0.291      0.122   0.578
64        300            0.475   0.374      0.649   0.484
64        1000           0.489   0.374      0.704   0.470
64        5000           0.478   0.365      0.695   0.455
64        10000          0.474   0.353      0.723   0.424
64        50000          0.502   0.362      0.820   0.415
64        -12000         0.508   0.363      0.851   0.409

Training train_xgb complex random mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.352   0.327      0.381   0.515
16        300            0.169   0.355      0.111   0.623
16        1000           0.200   0.349      0.140   0.612
16        5000           0.162   0.277      0.115   0.590
16        10000          0.227   0.333      0.173   0.594
16        50000          0.195   0.319      0.141   0.599
16        -12000         0.295   0.564      0.200   0.670
32        100            0.298   0.329      0.272   0.556
32        300            0.199   0.295      0.151   0.582
32        1000           0.259   0.285      0.238   0.530
32        5000           0.200   0.316      0.146   0.595
32        10000          0.294   0.323      0.271   0.551
32        50000          0.199   0.346      0.140   0.611
32        -12000         0.275   0.547      0.184   0.665
64        100            0.278   0.319      0.247   0.557
64        300            0.171   0.358      0.113   0.623
64        1000           0.167   0.287      0.118   0.593
64        5000           0.311   0.394      0.257   0.606
64        10000          0.256   0.290      0.228   0.540
64        50000          0.193   0.320      0.138   0.600
64        -12000         0.311   0.556      0.216   0.669

Training train_xgb  random mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.330   0.315      0.346   0.514
16        300            0.223   0.314      0.172   0.584
16        1000           0.227   0.305      0.180   0.575
16        5000           0.189   0.316      0.135   0.600
16        10000          0.270   0.357      0.217   0.594
16        50000          0.199   0.382      0.135   0.625
16        -12000         0.184   0.387      0.121   0.630
32        100            0.349   0.326      0.374   0.517
32        300            0.146   0.328      0.094   0.620
32        1000           0.178   0.321      0.123   0.607
32        5000           0.194   0.351      0.134   0.615
32        10000          0.234   0.337      0.179   0.594
32        50000          0.182   0.369      0.121   0.624
32        -12000         0.197   0.484      0.124   0.651
64        100            0.407   0.356      0.476   0.521
64        300            0.163   0.348      0.106   0.622
64        1000           0.319   0.404      0.263   0.611
64        5000           0.237   0.330      0.185   0.588
64        10000          0.274   0.334      0.232   0.575
64        50000          0.191   0.345      0.133   0.613
64        -12000         0.202   0.454      0.130   0.645

Training train_xgb complex  mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.374   0.314      0.461   0.466
16        300            0.257   0.388      0.192   0.616
16        1000           0.163   0.303      0.112   0.604
16        5000           0.215   0.296      0.169   0.573
16        10000          0.290   0.343      0.251   0.575
16        50000          0.177   0.311      0.124   0.602
16        -12000         0.527   0.596      0.472   0.707
32        100            0.403   0.364      0.451   0.537
32        300            0.199   0.271      0.157   0.562
32        1000           0.251   0.314      0.209   0.569
32        5000           0.239   0.292      0.203   0.554
32        10000          0.255   0.386      0.191   0.615
32        50000          0.256   0.469      0.176   0.646
32        -12000         0.564   0.696      0.474   0.747
64        100            0.343   0.354      0.333   0.559
64        300            0.171   0.270      0.126   0.580
64        1000           0.257   0.437      0.182   0.636
64        5000           0.288   0.395      0.227   0.612
64        10000          0.305   0.326      0.287   0.548
64        50000          0.162   0.325      0.108   0.614
64        -12000         0.557   0.703      0.461   0.746

Training train_xgb   mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.310   0.300      0.320   0.506
16        300            0.179   0.257      0.137   0.565
16        1000           0.253   0.342      0.201   0.590
16        5000           0.222   0.314      0.171   0.584
16        10000          0.253   0.336      0.202   0.586
16        50000          0.244   0.397      0.176   0.623
16        -12000         0.553   0.640      0.487   0.728
32        100            0.276   0.291      0.262   0.524
32        300            0.051   0.226      0.029   0.630
32        1000           0.198   0.278      0.154   0.569
32        5000           0.257   0.346      0.204   0.591
32        10000          0.259   0.337      0.211   0.584
32        50000          0.184   0.326      0.128   0.607
32        -12000         0.560   0.670      0.481   0.739
64        100            0.320   0.350      0.295   0.566
64        300            0.220   0.310      0.171   0.582
64        1000           0.281   0.395      0.218   0.614
64        5000           0.262   0.345      0.212   0.588
64        10000          0.422   0.436      0.409   0.612
64        50000          0.209   0.387      0.144   0.625
64        -12000         0.524   0.586      0.474   0.702

Training train_lr complex random mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.305   0.299      0.312   0.509
16        300            0.295   0.290      0.300   0.504
16        1000           0.475   0.347      0.750   0.425
16        5000           0.511   0.345      0.989   0.346
16        10000          0.514   0.346      1.000   0.348
16        50000          0.514   0.346      1.000   0.346
16        -12000         0.514   0.346      1.000   0.346
32        100            0.318   0.317      0.318   0.528
32        300            0.283   0.320      0.254   0.555
32        1000           0.475   0.347      0.751   0.426
32        5000           0.513   0.346      0.990   0.349
32        10000          0.514   0.346      1.000   0.346
32        50000          0.514   0.346      1.000   0.346
32        -12000         0.514   0.346      1.000   0.346
64        100            0.262   0.351      0.209   0.593
64        300            0.321   0.310      0.332   0.513
64        1000           0.395   0.324      0.505   0.464
64        5000           0.480   0.333      0.863   0.355
64        10000          0.514   0.347      0.994   0.351
64        50000          0.515   0.346      1.000   0.348
64        -12000         0.514   0.346      1.000   0.346

Training train_lr  random mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.428   0.349      0.554   0.488
16        300            0.469   0.341      0.752   0.411
16        1000           0.504   0.344      0.940   0.359
16        5000           0.514   0.346      1.000   0.346
16        10000          0.514   0.346      1.000   0.346
16        50000          0.514   0.346      1.000   0.346
16        -12000         0.514   0.346      1.000   0.346
32        100            0.404   0.322      0.543   0.446
32        300            0.289   0.350      0.247   0.581
32        1000           0.483   0.356      0.752   0.444
32        5000           0.514   0.346      0.998   0.346
32        10000          0.514   0.346      1.000   0.346
32        50000          0.514   0.346      1.000   0.346
32        -12000         0.514   0.346      1.000   0.346
64        100            0.379   0.339      0.429   0.514
64        300            0.316   0.321      0.311   0.534
64        1000           0.486   0.360      0.748   0.454
64        5000           0.514   0.346      0.996   0.348
64        10000          0.514   0.346      1.000   0.346
64        50000          0.514   0.346      1.000   0.346
64        -12000         0.514   0.346      1.000   0.346

Training train_lr complex  mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.316   0.311      0.320   0.519
16        300            0.403   0.353      0.468   0.519
16        1000           0.474   0.393      0.595   0.542
16        5000           0.487   0.353      0.787   0.427
16        10000          0.517   0.354      0.962   0.379
16        50000          0.518   0.350      1.000   0.357
16        -12000         0.517   0.348      1.000   0.353
32        100            0.358   0.306      0.431   0.465
32        300            0.231   0.232      0.231   0.469
32        1000           0.346   0.284      0.441   0.423
32        5000           0.457   0.327      0.755   0.378
32        10000          0.485   0.337      0.861   0.367
32        50000          0.527   0.358      0.998   0.380
32        -12000         0.520   0.352      1.000   0.362
64        100            0.405   0.323      0.542   0.449
64        300            0.369   0.354      0.386   0.545
64        1000           0.310   0.285      0.339   0.477
64        5000           0.462   0.336      0.739   0.405
64        10000          0.500   0.353      0.859   0.407
64        50000          0.487   0.334      0.900   0.343
64        -12000         0.524   0.355      0.999   0.372

Training train_lr   mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.282   0.282      0.283   0.503
16        300            0.284   0.270      0.300   0.477
16        1000           0.477   0.349      0.754   0.429
16        5000           0.498   0.340      0.931   0.350
16        10000          0.514   0.346      0.999   0.347
16        50000          0.514   0.346      1.000   0.347
16        -12000         0.514   0.346      1.000   0.346
32        100            0.313   0.298      0.328   0.500
32        300            0.171   0.239      0.133   0.553
32        1000           0.438   0.355      0.573   0.493
32        5000           0.484   0.335      0.865   0.361
32        10000          0.519   0.353      0.977   0.374
32        50000          0.519   0.350      1.000   0.358
32        -12000         0.514   0.346      1.000   0.347
64        100            0.271   0.362      0.216   0.597
64        300            0.323   0.358      0.295   0.573
64        1000           0.426   0.346      0.553   0.484
64        5000           0.499   0.346      0.890   0.381
64        10000          0.522   0.356      0.973   0.383
64        50000          0.515   0.346      1.000   0.347
64        -12000         0.521   0.353      1.000   0.365

Training train_pac complex random mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.309   0.291      0.329   0.490
16        300            0.315   0.296      0.338   0.493
16        1000           0.480   0.357      0.730   0.453
16        5000           0.456   0.340      0.693   0.429
16        10000          0.518   0.370      0.863   0.445
16        50000          0.422   0.306      0.677   0.358
16        -12000         0.483   0.341      0.825   0.389
32        100            0.381   0.320      0.470   0.472
32        300            0.303   0.318      0.290   0.539
32        1000           0.424   0.319      0.631   0.407
32        5000           0.483   0.346      0.800   0.408
32        10000          0.492   0.344      0.861   0.384
32        50000          0.499   0.346      0.895   0.379
32        -12000         0.490   0.347      0.831   0.401
64        100            0.285   0.332      0.250   0.567
64        300            0.336   0.319      0.355   0.515
64        1000           0.454   0.343      0.674   0.441
64        5000           0.470   0.347      0.732   0.430
64        10000          0.442   0.335      0.650   0.433
64        50000          0.498   0.349      0.866   0.395
64        -12000         0.476   0.342      0.786   0.402

Training train_pac  random mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.482   0.360      0.733   0.457
16        300            0.461   0.338      0.724   0.415
16        1000           0.475   0.356      0.711   0.455
16        5000           0.458   0.333      0.732   0.401
16        10000          0.490   0.350      0.817   0.413
16        50000          0.487   0.336      0.885   0.354
16        -12000         0.464   0.334      0.758   0.395
32        100            0.370   0.298      0.486   0.426
32        300            0.433   0.343      0.588   0.468
32        1000           0.477   0.355      0.727   0.449
32        5000           0.493   0.348      0.849   0.397
32        10000          0.492   0.350      0.832   0.406
32        50000          0.521   0.362      0.927   0.411
32        -12000         0.475   0.344      0.770   0.412
64        100            0.370   0.334      0.415   0.511
64        300            0.407   0.344      0.496   0.499
64        1000           0.468   0.352      0.699   0.451
64        5000           0.499   0.351      0.860   0.402
64        10000          0.492   0.349      0.838   0.403
64        50000          0.470   0.346      0.734   0.428
64        -12000         0.483   0.348      0.786   0.417

Training train_pac complex  mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.237   0.278      0.207   0.540
16        300            0.285   0.308      0.266   0.540
16        1000           0.458   0.380      0.578   0.528
16        5000           0.521   0.384      0.810   0.485
16        10000          0.466   0.339      0.747   0.409
16        50000          0.505   0.345      0.941   0.361
16        -12000         0.478   0.340      0.800   0.395
32        100            0.259   0.253      0.265   0.475
32        300            0.307   0.266      0.363   0.433
32        1000           0.361   0.306      0.441   0.460
32        5000           0.424   0.314      0.651   0.388
32        10000          0.482   0.339      0.833   0.382
32        50000          0.481   0.351      0.761   0.431
32        -12000         0.485   0.351      0.788   0.422
64        100            0.390   0.313      0.517   0.440
64        300            0.359   0.308      0.429   0.469
64        1000           0.360   0.296      0.460   0.435
64        5000           0.434   0.320      0.675   0.391
64        10000          0.492   0.356      0.798   0.431
64        50000          0.512   0.361      0.882   0.418
64        -12000         0.476   0.349      0.747   0.430

Training train_pac   mixed test/train weighted

hrr_size, training_size, fscore, precision, recall, acc
16        100            0.293   0.313      0.275   0.540
16        300            0.265   0.265      0.266   0.491
16        1000           0.365   0.278      0.533   0.359
16        5000           0.419   0.310      0.646   0.380
16        10000          0.511   0.345      0.982   0.349
16        50000          0.471   0.329      0.829   0.357
16        -12000         0.501   0.345      0.912   0.372
32        100            0.261   0.271      0.251   0.507
32        300            0.285   0.304      0.268   0.535
32        1000           0.427   0.309      0.687   0.361
32        5000           0.516   0.373      0.837   0.456
32        10000          0.524   0.371      0.895   0.438
32        50000          0.444   0.316      0.747   0.353
32        -12000         0.502   0.355      0.855   0.412
64        100            0.236   0.396      0.168   0.624
64        300            0.369   0.382      0.356   0.578
64        1000           0.485   0.359      0.747   0.450
64        5000           0.488   0.343      0.847   0.386
64        10000          0.492   0.345      0.855   0.389
64        50000          0.463   0.339      0.729   0.414
64        -12000         0.425   0.335      0.583   0.455

"""
