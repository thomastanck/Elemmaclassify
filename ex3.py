'''
Experiment 3 runner

Usage:
    ex3.py <experiment_id>
'''

import sys
import collections

import docopt
import torch
import torch.nn as nn
import torch.optim as optim

import HRRTorch
import HRRClassifier
import trainv1
import lemmadata

ExperimentSettings = collections.namedtuple(
        'ExperimentSettings',
        '''
        hrr_size
        num_decoders
        num_classifier_layers
        num_classifier_hidden_neurons
        classifier_nonlinearity
        unitvecloss_weight
        sumvecloss_weight
        mlpweightloss_weight

        adam_lr
        adam_beta1
        adam_beta2
        adam_weight_decay
        ''')

experiments = [
        ExperimentSettings(16, 0,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 0,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 0,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 0,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 16, 2, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 16, 3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 0,  2, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 0,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  2, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 0,  2, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 0,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  2, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 16, 2, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 16, 3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-3, 0.9, 0.999, 0),

        # Adam LR experiments
        ExperimentSettings(16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.1, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings(16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings(64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.1, 1e-1, 0.9, 0.999, 0),
        ]

def make_model(experimentsettings):
    hrr_size = experimentsettings.hrr_size
    num_decoders = experimentsettings.num_decoders
    num_classifier_layers = experimentsettings.num_classifier_layers
    num_classifier_hidden_neurons = experimentsettings.num_classifier_hidden_neurons
    classifier_nonlinearity = experimentsettings.classifier_nonlinearity
    hrrmodel = HRRTorch.FlatTreeHRRTorch2(hrr_size)
    featurizer = HRRClassifier.Decoder_featurizer(hrr_size, num_decoders)
    nonlinearity = {
            'Sigmoid': nn.Sigmoid(),
            'LeakyReLU': nn.LeakyReLU(),
            }[classifier_nonlinearity]
    classifier = HRRClassifier.MLP_classifier(
            featurizer.get_output_size(),
            [num_classifier_hidden_neurons] * (num_classifier_layers - 1),
            nonlinearity,
            )

    return HRRClassifier.HRRClassifier(
            hrrmodel,
            featurizer,
            classifier,
            )

def make_loss(model, experimentsettings):
    bceloss = nn.BCEWithLogitsLoss()
    def loss_func(model, pred, target):
        errorloss = bceloss(pred, target)

        # Add regularisation for unit vectors (should have norm 1)
        unitvecloss = torch.tensor([0.])
        count = 0
        for param in (
                *model.hrrmodel.fixed_encodings.values(),
                *model.featurizer.decoders,
                ):
            sqsum = torch.sum(param ** 2)
            # Sum squared error = (norm - 1) ** 2
            #                   = (norm**2 - 2*norm + 1)
            unitvecloss += sqsum - 2 * sqsum ** 0.5 + 1
            count += 1
        unitvecloss /= count

        # Add regularisation for mixing vectors (should sum to 1)
        sumvecloss = torch.tensor([0.])
        count = 0
        for param in (
                *model.hrrmodel.ground_vec_merge_ratios.values(),
                model.hrrmodel.func_weights,
                model.hrrmodel.eq_weights,
                model.hrrmodel.disj_weights,
                model.hrrmodel.conj_weights,
                ):
            sum_ = torch.sum(param)
            # Sum squared error = (sum - 1) ** 2
            sumvecloss += (sum_ - 1) ** 2
            count += 1
        sumvecloss /= count

        # Add regularisation for variance vectors?
        # Let's not bother (looking at the data, it looks well behaved)

        # Add regularisation for MLP weights
        mlpweightloss = torch.tensor([0.])
        count = 0
        for param in model.classifier.parameters():
            mlpweightloss += param.norm()
            count += 1
        mlpweightloss /= count

        return (errorloss
                + experimentsettings.unitvecloss_weight * unitvecloss
                + experimentsettings.sumvecloss_weight * sumvecloss
                + experimentsettings.mlpweightloss_weight * mlpweightloss
                )

    return loss_func

def make_opt(model, experimentsettings):
    lr = experimentsettings.adam_lr
    beta1 = experimentsettings.adam_beta1
    beta2 = experimentsettings.adam_beta2
    weight_decay = experimentsettings.adam_weight_decay

    return optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

if __name__ == '__main__':
    settings = docopt.docopt(__doc__)

    experiment_id = int(settings['<experiment_id>'])
    experimentsettings = experiments[experiment_id]

    model = make_model(experimentsettings)
    loss_func = make_loss(model, experimentsettings)
    opt = make_opt(model, experimentsettings)

    trainv1.train(
            'ex3data',
            'ex3',
            model,
            loss_func,
            opt,
            lemmadata.get_train,
            lemmadata.get_crossval,
            experimentsettings,
            )

