import torch
import torch.nn as nn
import HRRTorch

def MLP_classifier(feature_size, layer_sizes, nonlinearity=nn.LeakyReLU()):
    model = nn.Sequential()
    numneurons = feature_size
    for i, size in enumerate(layer_sizes):
        model.add_module(str(i) + 'Linear', nn.Linear(numneurons, size))
        model.add_module(str(i) + 'Nonlinear', nonlinearity)
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

        self.output_size = hrr_size * 2 * (1 + num_decoders)

        for p in self.decoders:
            nn.init.normal_(p)
            with torch.no_grad():
                p /= p.norm()

    def forward(self, problemhrr, lemmahrr):
        return torch.cat([
            problemhrr,
            lemmahrr,
            *(HRRTorch.associate(decoder, problemhrr) for decoder in self.decoders),
            *(HRRTorch.associate(decoder, lemmahrr) for decoder in self.decoders),
            ])

    def get_output_size(self):
        return self.output_size

class HRRClassifier(nn.Module):

    def __init__(self, hrrmodel, featurizer, classifier):
        super(HRRClassifier, self).__init__()

        self.hrrmodel = hrrmodel
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, points):
        out = []
        for problem, lemma in points:
            problemhrr = self.hrrmodel(problem)
            lemmahrr = self.hrrmodel(lemma)
            features = self.featurizer(problemhrr, lemmahrr)
            out.append(self.classifier(features))
        return torch.cat(out)
