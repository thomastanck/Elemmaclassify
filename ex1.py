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

@utils.persist_to_file('ex1data/hrrs-pairs-{1}-{2}-{3}.pickle', key=lambda *params: str(params[:3]))
def get_hrrs_pairs(hrr_class, hrr_size, hrr_args=dict(), index=0):
    import multiprocessing as mp

    print('getting hrrs (pairs)')

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
            process_problem_lemma_pair,
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
        problemlemmanames = shuffle_by_hash(get_lemmanamesindex())[:training_size]
    else:
        # Segment training/test data by name
        problemlemmanames = shuffle_by_hash(get_lemmanamesindex()[:-12000*num_hrrs])[:training_size]

    testX = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] ])
    testy = np.array([ 1 if usefulness[problemname][lemmaname] < 1 else 0 for problemname, lemmaname, index in problemlemmanames ])
    testXpos = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] if usefulness[problemname][lemmaname] < 1 ])
    testypos = np.array([ 1 for problemname, lemmaname, index in problemlemmanames if usefulness[problemname][lemmaname] < 1 ])
    testXpos = np.array([ np.concatenate(plhrr) for problemname, lemmaname, index in problemlemmanames for plhrr in pairshrrs[index][(problemname, lemmaname)] if usefulness[problemname][lemmaname] >= 1 ])
    testyneg = np.array([ 0 for problemname, lemmaname, index in problemlemmanames if usefulness[problemname][lemmaname] >= 1 ])

    testX_scaled = scaler.transform(testX)
    testXpos_scaled = scaler.transform(testXpos)
    testXneg_scaled = scaler.transform(testXneg)

    return testX_scaled, testy, testXpos_scaled, testypos, testXneg_scaled, testyneg

@utils.persist_to_file('ex1data/lrweights-{}.pickle', key=lambda *params: str(params))
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
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/pacweights-{}.pickle', key=lambda *params: str(params))
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
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

@utils.persist_to_file('ex1data/svcweights-{}.pickle', key=lambda *params: str(params))
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

@utils.persist_to_file('ex1data/mixed-xgbweights-{}.pickle', key=lambda *params: str(params))
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

args = [ (hrr_size, hrr_args, index)
    for hrr_size in [
            16,
            32,
            64,
            256,
            # 1024,
            ]
    for hrr_args in [
            {'complex_hrr': True,  'randomise_variables': True,  'randomise_skolem_constants': True, },
            {'complex_hrr': False, 'randomise_variables': True,  'randomise_skolem_constants': True, },
            {'complex_hrr': True,  'randomise_variables': False, 'randomise_skolem_constants': False, },
            {'complex_hrr': False, 'randomise_variables': False, 'randomise_skolem_constants': False, },
                ]
    for index in range(1)
    ]
print(sys.argv[1], 'out of', len(args))
actualargs = args[int(sys.argv[1])]

## Training/testing

# num_hrrs = 1
# for model_train, params in [
#         (train_lr, tuple(sorted(list(lm.LogisticRegression(max_iter=5000).get_params().items())))),
#         (train_pac, tuple(sorted(list(lm.PassiveAggressiveClassifier().get_params().items())))),
#         (train_svc, tuple(sorted(list(svm.SVC().get_params().items())))),
#         (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32).get_params().items())))),
#         ]:
#     print('Training', model_train.__name__)
#     print('hrr_size, training_size, fscore, precision, recall, acc')
#     for hrr_size in [
#             # 16,
#             # 32,
#             64,
#             # 256,
#             # 1024,
#             ]:
#         for training_size in [
#                 100 * num_hrrs,
#                 # 300 * num_hrrs,
#                 1000 * num_hrrs,
#                 # 5000 * num_hrrs,
#                 # 10000 * num_hrrs,
#                 # 50000 * num_hrrs,
#                 # -12000 * num_hrrs,
#                 ]:
#             for shuffle in [True, False]:
#                 for hrr_args in [
#                         {'complex_hrr': True,  'randomise_variables': True,  'randomise_skolem_constants': True, },
#                         {'complex_hrr': False, 'randomise_variables': True,  'randomise_skolem_constants': True, },
#                         {'complex_hrr': True,  'randomise_variables': False, 'randomise_skolem_constants': False, },
#                         {'complex_hrr': False, 'randomise_variables': False, 'randomise_skolem_constants': False, },
#                             ]:
#                     for hrr_class in [HRR.FlatTreeHRR]:
#                         scaler, model = model_train(hrr_size, training_size, hrr_class, shuffle, hrr_args, num_hrrs, params)
#                         print(hrr_size, training_size,
#                                 *('{:0.3f}'.format(x)
#                                     for x in test_model(HRR.FlatTreeHRR, hrr_size, scaler, model, shuffle, num_hrrs, hrr_args)))
