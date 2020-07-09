import functools
import collections
import shelve

import torch
import numpy

import parse
import datautils
import utils
import lemmadata
import HRRv2

@utils.persist_to_shelf('lemmadatav2/problems.shelf')
def get_problem(pname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(pname))

@functools.lru_cache()
def get_cnftohrrs(settings):
    cnftohrrs = [
            HRRv2.CNFtoHRR1(settings.hrr_size, HRRv2.defaultCNFtoHRR1Settings)
            if i == 0 else
            HRRv2.CNFtoHRR1(settings.hrr_size, HRRv2.randomCNFtoHRR1Settings_v1(i))
            for i in range(settings.num_hrrs) ]

    return cnftohrrs

def process_problem_multihrr(args):
    settings, pname = args
    print('working on', pname)
    problem = get_problem(pname)
    cnftohrrs = get_cnftohrrs(settings)
    return (pname,
            numpy.concatenate([cnftohrr.fold_term(problem).to_real_vec()
                        for cnftohrr in cnftohrrs]))

def process_lemma_multihrr(args):
    settings, pname, lname, lemma = args
    print('working on', pname, lname)
    cnftohrrs = get_cnftohrrs(settings)
    return (pname,
            lname,
            numpy.concatenate([cnftohrr.fold_term(lemma).to_real_vec()
                        for cnftohrr in cnftohrrs]))

def preprocess_dataset_multihrr(settings):
    with shelve.open('lemmadatav2/lemmas-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs)) as lemma_shelf:
        with shelve.open('lemmadatav2/problems-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs)) as problem_shelf:
            if 'done' not in lemma_shelf or 'done' not in problem_shelf:
                import multiprocessing

                problemnames = set()
                pnamelnames = list()
                with open('E_conj/lemmas') as f:
                    for l in f:
                        name, lemma = l.split(':')
                        _, problemname, lemmaname = name.split('/')
                        problemnames.add(problemname)
                        pnamelnames.append((problemname, lemmaname))

                problemslemmas = lemmadata.get_problemslemmas()
                usefulness = lemmadata.get_usefulness()

                if 'done' not in lemma_shelf:
                    with multiprocessing.Pool() as pool:
                        print('preproc lemmas')
                        for i, (pname, lname, hrr) in enumerate(pool.imap_unordered(
                                process_lemma_multihrr,
                                ((settings, pname, lname, lemma)
                                    for (pname, lname, problem, lemma) in problemslemmas
                                    if pname+'/'+lname not in lemma_shelf),
                                16)):
                            print(i, pname, lname)
                            lemma_shelf[pname+'/'+lname] = hrr
                    lemma_shelf['done'] = True

                if 'done' not in lemma_shelf:
                    with multiprocessing.Pool() as pool:
                        print('preproc problems')
                        for i, (pname, hrr) in enumerate(pool.imap_unordered(
                                process_problem_multihrr,
                                ((settings, pname)
                                    for pname in problemnames
                                    if pname not in problem_shelf),
                                16)):
                            print(i, pname)
                            problem_shelf[pname] = hrr
                    problem_shelf['done'] = True

                print('preproc done')

MultiHRRDataset1Settings = collections.namedtuple('MultiHRRDataset1Settings',
        '''
        hrr_size
        num_hrrs
        ''')

class MultiHRRDataset1(torch.utils.data.Dataset):
    def __init__(
            self,
            settings=MultiHRRDataset1Settings(1024, 16),
            ):
        self.lemma_shelf = shelve.open('lemmadatav2/lemmas-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs))
        self.problem_shelf = shelve.open('lemmadatav2/problems-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs))
        self.usefulness = lemmadata.get_usefulness()
        self.pln = lemmadata.get_problemslemmas()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def close(self):
        self.lemma_shelf.close()
        self.problem_shelf.close()

    def __getitem__(self, idx):
        pname, lname = self.pln[idx]
        return numpy.concatenate([self.problem_shelf[pname], self.lemma_shelf[pname+'/'+lname]]), float(self.usefulness[pname][lname] < 1)

    def __len__(self):
        return len(self.pln)

def dataset_to_Xy(dataset):
    Xs, ys = [], []
    for X, y in dataset:
        Xs.append(X)
        ys.append(y)
    return Xs, ys
