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
    problem = get_problem(pname)
    cnftohrrs = get_cnftohrrs(settings)
    return (pname,
            numpy.concatenate([cnftohrr.fold_term(problem).to_real_vec()
                        for cnftohrr in cnftohrrs]))

def process_lemma_multihrr(args):
    settings, pname, lname, lemma = args
    cnftohrrs = get_cnftohrrs(settings)
    return (pname,
            lname,
            numpy.concatenate([cnftohrr.fold_term(lemma).to_real_vec()
                        for cnftohrr in cnftohrrs]))

def get_dataset_multihrr(settings):
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

    with multiprocessing.Pool() as pool:
        print('preproc lemmas')
        with shelve.open('lemmadatav2/lemmas-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs)) as lemma_shelf:
            for i, (pname, lname, hrr) in enumerate(pool.imap_unordered(
                    process_lemma_multihrr,
                    ((settings, pname, lname, lemma)
                        for (pname, lname, problem, lemma) in problemslemmas
                        if pname+'/'+lname not in lemma_shelf),
                    16)):
                print(i, pname, lname)
                lemma_shelf[pname+'/'+lname] = hrr

        print('preproc problems')
        with shelve.open('lemmadatav2/problems-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs)) as problem_shelf:
            for i, (pname, hrr) in enumerate(pool.imap_unordered(
                    process_problem_multihrr,
                    ((settings, pname)
                        for pname in problemnames
                        if pname not in problem_shelf),
                    16)):
                print(i, pname)
                problem_shelf[pname] = hrr

        print('preproc done')

    with shelve.open('lemmadatav2/problems-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs)) as problem_shelf:
        with shelve.open('lemmadatav2/lemmas-multihrr-{}-{}.shelf'.format(settings.hrr_size, settings.num_hrrs)) as lemma_shelf:
            for pname, lname in pnamelnames:
                yield (numpy.concatenate((problem_shelf[pname], lemma_shelf[pname+'/'+lname])), usefulness[pname][lname])

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
        raise NotImplementedError()

    def __getitem__(self, idx):
        return self.data[idx][0], int(self.data[idx][1])

    def __len__(self):
        return len(self.data)
