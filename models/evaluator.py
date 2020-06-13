import torch.nn as nn
from opts import *


class MLP_block(nn.Module):

    def __init__(self, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output


class Evaluator(nn.Module):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block(output_dim=output_dim)
        elif model_type == 'MUSDL':
            if not num_judges:
                raise NotImplementedError
            else:
                self.evaluator = nn.ModuleList([MLP_block(output_dim=output_dim) for _ in range(num_judges)])
        else:
            raise NotImplementedError

    def forward(self, feats_avg):  # data: NCTHW

        if self.model_type == 'USDL':
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        elif self.model_type == 'MUSDL':
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        else:
            raise NotImplementedError
        return probs

