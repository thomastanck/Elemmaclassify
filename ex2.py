import functools
import itertools
import collections
import zlib

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.tensorboard as torchboard

import parse
import utils
import HRRTorch

torch.set_num_threads(1)

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

@utils.persist_to_file('ex2data/usefulness.pickle')
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
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

def _process_problemslemmas(l):
    name, lemma = l.split(':')
    _, problemname, lemmaname = name.split('/')
    return (
        problemname,
        lemmaname,
        parse_problem(problemname),
        parse.parse_cnf_clause(lemma),
        )


@utils.persist_to_file('ex2data/problemslemmas.pickle')
def get_problemslemmas():
    print('parsing problems and lemmas')
    import multiprocessing

    with multiprocessing.Pool() as pool:
        with open('E_conj/lemmas') as f:
            return pool.map(_process_problemslemmas, f, 32)

def shuffle_by_hash(l, key=str):
    return sorted(l,
            key=lambda x:
                zlib.crc32(key(x).encode('utf-8')) & 0xffffffff)

def train_test_split(names, test_split, sortkey):
    shuf = sorted(names, key=sortkey)
    train_size = int(len(shuf) * (1-test_split))
    return shuf[:train_size], shuf[train_size:]

def pos_neg_split(names):
    pos, neg = [], []
    for pname, lname, problem, lemma in names:
        if usefulness[pname][lname] < 1:
            pos.append((pname, lname, problem, lemma))
        else:
            neg.append((pname, lname, problem, lemma))
    return pos, neg

def MLP_classifier(feature_size, layer_sizes):
    model = nn.Sequential()
    numneurons = feature_size
    for i, size in enumerate(layer_sizes):
        model.add_module(str(i) + 'Linear', nn.Linear(numneurons, size))
        model.add_module(str(i) + 'LeakyReLU', nn.LeakyReLU())
        numneurons = size
    model.add_module(str(len(layer_sizes)) + 'Linear', nn.Linear(numneurons, 1))
    return model

class Cat_featurizer(nn.Module):
    def forward(self, problemhrr, lemmahrr):
        return torch.cat([problemhrr, lemmahrr])

class Decoder_featurizer(nn.Module):
    def __init__(self, hrr_size, num_decoders):
        super(Decoder_featurizer, self).__init__()

        decoders = [ nn.Parameter(torch.Tensor(hrr_size)) for i in range(num_decoders) ]
        self.decoders = nn.ParameterList(decoders)

    def forward(self, problemhrr, lemmahrr):
        return torch.cat([
            problemhrr,
            lemmahrr,
            *(HRRTorch.associate(decoder, problemhrr) for decoder in self.decoders),
            *(HRRTorch.associate(decoder, lemmahrr) for decoder in self.decoders),
            ])

class HRRClassifier(nn.Module):

    def __init__(self, hrrmodel, featurizer, classifier):
        super(HRRClassifier, self).__init__()

        self.hrrmodel = hrrmodel
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, problem, lemma):
        problemhrr = self.hrrmodel(problem)
        lemmahrr = self.hrrmodel(lemma)
        features = self.featurizer(problemhrr, lemmahrr)
        out = self.classifier(features)
        return out

    def cache_clear(self):
        self.hrrmodel.cache_clear()

usefulness = get_usefulness()
train, test = train_test_split(get_problemslemmas(), 0.1, lambda x: x)
train_pos, train_neg = pos_neg_split(train)
train_pos = shuffle_by_hash(train_pos, key=lambda x: str(x[:2]))
train_neg = shuffle_by_hash(train_neg, key=lambda x: str(x[:2]))
num_examples = min(len(train_pos), len(train_neg))
train_pos = train_pos[:num_examples]
train_neg = train_neg[:num_examples]

# hrr_size = 16
# model = HRRClassifier(
#         HRRTorch.FlatTreeHRRTorch2(hrr_size),
#         Cat_featurizer(),
#         MLP_classifier(hrr_size * 2, [hrr_size*4] * 2))
# loss_func = nn.BCEWithLogitsLoss()
# opt = optim.Adam(model.parameters())
# writer = torchboard.SummaryWriter(comment='FlatTreeHRRTorch2_3LP_BCEWithLogitsLoss_HRR16_eqclasses')

hrr_size = 16
num_decoders = 16
num_classifier_layers = 3
model = HRRClassifier(
        HRRTorch.FlatTreeHRRTorch2(hrr_size),
        Decoder_featurizer(hrr_size, num_decoders),
        MLP_classifier((num_decoders + 1) * hrr_size * 2, [hrr_size*4] * (num_classifier_layers-1)))
loss_func = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters())
writer = torchboard.SummaryWriter(
        comment='FlatTreeHRRTorch2_Decoder{}_{}LP_BCEWithLogitsLoss_HRR{}_eqclasses'.format(
            num_decoders,
            num_classifier_layers,
            hrr_size,
            ))

hrr_size = 16
num_decoders = 16
num_classifier_layers = 3
model = HRRClassifier(
        HRRTorch.FlatTreeHRRTorch2(hrr_size),
        Decoder_featurizer(hrr_size, num_decoders),
        MLP_classifier((num_decoders + 1) * hrr_size * 2, [hrr_size*4] * (num_classifier_layers-1)))
loss_func = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters())
writer = torchboard.SummaryWriter(
        comment='FlatTreeHRRTorch2_Decoder{}_{}LP_BCEWithLogitsLoss_HRR{}_eqclasses'.format(
            num_decoders,
            num_classifier_layers,
            hrr_size,
            ))

def calc_stats(numtruepos, numtrueneg, numfalsepos, numfalseneg):
    total = numtruepos + numtrueneg + numfalsepos + numfalseneg
    classified_pos = numtruepos + numfalsepos
    actual_pos = numtruepos + numfalseneg
    acc = 0 if total == 0 else (numtruepos + numtrueneg) / total
    precision = 0 if classified_pos == 0 else numtruepos / classified_pos
    recall = 0 if actual_pos == 0 else numtruepos / actual_pos
    fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return acc, precision, recall, fscore

epochs = 2
bs = 8
for epoch in range(epochs):
    numtruepos = 0
    numtrueneg = 0
    numfalsepos = 0
    numfalseneg = 0
    for i, batchnum in enumerate(shuffle_by_hash(range((num_examples - 1) // bs + 1))):
        start_i = batchnum * bs
        end_i = start_i + bs
        loss = torch.tensor([0.])

        if i % 5 == 0:
            acc, precision, recall, fscore = calc_stats(numtruepos, numtrueneg, numfalsepos, numfalseneg)
            writer.add_scalar('Acc/train', acc, i)
            writer.add_scalar('Precision/train', precision, i)
            writer.add_scalar('Recall/train', recall, i)
            writer.add_scalar('F-Score/train', fscore, i)
            numtruepos = 0
            numtrueneg = 0
            numfalsepos = 0
            numfalseneg = 0

            # Add weights
            def twodfy(tensor):
                if len(tensor.shape) == 1:
                    return tensor.reshape((1, -1))
                return tensor
            for name, weight in model.classifier.named_parameters():
                writer.add_image('classifier_{}'.format(name), twodfy(weight), dataformats='HW')
            for name, weight in model.featurizer.named_parameters():
                writer.add_image('featurizer_{}'.format(name), twodfy(weight), dataformats='HW')
            for name, weight in model.hrrmodel.named_parameters():
                writer.add_image('hrr_{}'.format(name), twodfy(weight), dataformats='HW')

        for problemname, lemmaname, problem, lemma in [
                *train_pos[start_i:end_i],
                *train_neg[start_i:end_i],
                ]:
            print(problemname, lemmaname)

            pred = model(problem, lemma)
            with torch.no_grad():
                if pred > 0 and usefulness[problemname][lemmaname] < 1:
                    numtruepos += 1
                elif pred <= 0 and usefulness[problemname][lemmaname] < 1:
                    numfalseneg += 1
                elif pred > 0 and usefulness[problemname][lemmaname] >= 1:
                    numfalsepos += 1
                elif pred <= 0 and usefulness[problemname][lemmaname] >= 1:
                    numtrueneg += 1
                acc, precision, recall, fscore = calc_stats(numtruepos, numtrueneg, numfalsepos, numfalseneg)
                print('{:0.3f}'.format(float(pred)), '{:0.3f}'.format(usefulness[problemname][lemmaname]), '|', numtruepos, numtrueneg, numfalsepos, numfalseneg, '|', '{:0.3f} {:0.3f} {:0.3f} {:0.3f}'.format(acc, precision, recall, fscore))

            loss += loss_func(pred, torch.tensor([1. if usefulness[problemname][lemmaname] < 1 else 0.]))
        loss.backward()
        model.cache_clear()

        writer.add_scalar('Loss/train', loss, i)

        opt.step()
        opt.zero_grad()
