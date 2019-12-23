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
            return pool.map(_process_problemslemmas, f)

def shuffle_by_hash(l, key=str):
    return sorted(l,
            key=lambda x:
                zlib.crc32(key(x).encode('utf-8')) & 0xffffffff)

def train_test_split(names, test_split, sortkey):
    shuf = sorted(names, key=sortkey)
    train_size = int(len(shuf) * (1-test_split))
    return shuf[:train_size], shuf[train_size:]

class HRRClassifier(nn.Module):

    def __init__(self, hrr_size):
        super(HRRClassifier, self).__init__()

        self.hrrmodel = HRRTorch.FlatTreeHRRTorch2(hrr_size)
        self.classifier = nn.Linear(hrr_size * 2, 1)

    def forward(self, problem, lemma):
        hrrs = torch.cat([
            self.hrrmodel(problem),
            self.hrrmodel(lemma),
            ])
        out = self.classifier(hrrs)
        return out

    def cache_clear(self):
        self.hrrmodel.cache_clear()

hrr_size = 16

model = HRRClassifier(hrr_size)
opt = optim.Adam(model.parameters())

usefulness = get_usefulness()
train, test = train_test_split(get_problemslemmas(), 0.1, lambda x: x)
train = shuffle_by_hash(train, key=lambda x: str(x[:2]))

loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))

writer = torchboard.SummaryWriter(comment='FlatTreeHRRTorch2_BCEWithLogitsLoss_HRR16')

def calc_stats(numtruepos, numtrueneg, numfalsepos, numfalseneg):
    total = numtruepos + numtrueneg + numfalsepos + numfalseneg
    classified_pos = numtruepos + numfalsepos
    actual_pos = numtruepos + numfalseneg
    acc = (numtruepos + numtrueneg) / total
    precision = 0 if classified_pos == 0 else numtruepos / classified_pos
    recall = 0 if actual_pos == 0 else numtruepos / actual_pos
    fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return acc, precision, recall, fscore

epochs = 2
bs = 10
for epoch in range(epochs):
    numtruepos = 0
    numtrueneg = 0
    numfalsepos = 0
    numfalseneg = 0
    for i, batchnum in enumerate(shuffle_by_hash(range((len(train) - 1) // bs + 1))):
        start_i = batchnum * bs
        end_i = start_i + bs
        loss = torch.tensor([0.])
        for problemname, lemmaname, problem, lemma in train[start_i:end_i]:
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
        if i % 5 == 4:
            acc, precision, recall, fscore = calc_stats(numtruepos, numtrueneg, numfalsepos, numfalseneg)
            writer.add_scalar('Acc/train', acc, i)
            writer.add_scalar('Precision/train', precision, i)
            writer.add_scalar('Recall/train', recall, i)
            writer.add_scalar('F-Score/train', fscore, i)
            numtruepos = 0
            numtrueneg = 0
            numfalsepos = 0
            numfalseneg = 0

        opt.step()
        opt.zero_grad()
