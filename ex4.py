import utils
import lemmadatav2

import sys
import zlib

import torch

import numpy as np
import sklearn as sk

import sklearn.svm as svm
import sklearn.preprocessing as pre
import sklearn.linear_model as lm

import xgboost

def get_shuf_ind(n):
    l = list(range(n))
    np.random.default_rng(0).shuffle(l)
    return l

def get_train_dataset(dataset, shuf, training_size):
    if shuf == True:
        ind = get_shuf_ind(122356)[:training_size]
    elif shuf == False:
        ind = range(122356)[:training_size]
    elif shuf == 'postshuf':
        ind = get_shuf_ind(range(122356-12000))[:training_size]
    return torch.utils.data.Subset(dataset, ind)

def get_test_dataset(dataset, shuf):
    if shuf == True:
        ind = get_shuf_ind(122356)[-12000:]
    elif shuf == False:
        ind = range(122356)[-12000:]
    elif shuf == 'postshuf':
        ind = range(122356)[-12000:]
    return torch.utils.data.Subset(dataset, ind)

@utils.persist_to_file('ex4data/xgbweights-{}.pickle', key=lambda *params: str(tuple(tuple(sorted(p.items())) if isinstance(p, dict) else p for p in params)))
def train_xgb(
        hrr_size=1024,
        num_hrrs=16,
        training_size=-12000,
        shuffle=False,
        params=tuple(sorted(list(xgboost.XGBClassifier().get_params().items())))):
    """
    Trains a XGBoost classifier
    """

    params = dict(params)

    print('Training XGB', hrr_size, num_hrrs, training_size, shuffle, params)

    origdataset = lemmadatav2.MultiHRRDataset1()
    truncdataset = lemmadatav2.MultiHRRDataset1Truncated(origdataset, hrr_size, num_hrrs)
    trainingdataset = get_train_dataset(truncdataset, shuffle, training_size)

    # scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)
    scaler, trainingX_scaled, trainingy = lemmadatav2.dataset_to_Xy(trainingdataset)

    model = xgboost.XGBClassifier()
    model.set_params(**params)
    model.set_params(n_jobs=24)
    model.fit(trainingX_scaled, trainingy)

    return scaler, model

def test_model(hrr_size, num_hrrs, shuffle, scaler, model):
    """
    Tests a model
    """

    origdataset = lemmadatav2.MultiHRRDataset1()
    truncdataset = lemmadatav2.MultiHRRDataset1Truncated(origdataset, hrr_size, num_hrrs)
    testdataset = get_test_dataset(truncdataset, shuffle)

    # scaler, trainingX_scaled, trainingy = get_training_data(hrr_class, hrr_size, training_size)
    _, testX_scaled, testy = lemmadatav2.dataset_to_Xy(testdataset, scaler)

    acc = model.score(testX_scaled, testy)

    # print('fscore, precision, recall, acc')
    # print(fscore, precision, recall, acc)
    return (acc,)

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

for shuffle in [False]:
    for model_train, params in [
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=1).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=2).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=3).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=4).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=5).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=10).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=20).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4, n_estimators=50).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=4).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=1).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=2).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=3).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=4).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=5).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=10).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=20).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8, n_estimators=50).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=8).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=1).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=2).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=3).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=4).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=5).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=10).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=20).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, n_estimators=50).get_params().items())))),
            (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32).get_params().items())))),
            # (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=128).get_params().items())))),
            # (train_xgb, tuple(sorted(list(xgboost.XGBClassifier(max_depth=32, class_weight={False: 1, True: 10}).get_params().items())))),
            ]:
            print('Training', model_train.__name__, 'mixed test/train' if shuffle else 'nonmixed test/train', 'nonweighted', params)
            print('hrr_size, training_size, acc')
            for num_hrrs, hrr_size in [
                    # (2, 16),
                    (2, 32),
                    # (2, 64),
                    # (2, 256),
                    (2, 1024),
                    # (8, 16),
                    # (8, 32),
                    # (8, 64),
                    # (8, 256),
                    # (16, 16),
                    (16, 32),
                    # (16, 64),
                    (16, 128),
                    ]:
                for training_size in [
                        # 100,
                        # 300,
                        # 1000,
                        # 5000,
                        # 10000,
                        # 50000,
                        -12000,
                        ]:
                    scaler, model = model_train(hrr_size, num_hrrs, training_size, shuffle, params)
                    print(num_hrrs, hrr_size, training_size,
                            *('{:0.3f}'.format(x)
                                for x in test_model(hrr_size, num_hrrs, shuffle, scaler, model)))
            print()
