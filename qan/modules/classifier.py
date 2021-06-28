import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm1d(in_features),
                                   nn.Dropout(p=0.25, inplace=False),
                                   nn.Linear(in_features, 512, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(512),
                                   nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(512, out_features, bias=False))
        
    def forward(self, inputs):
        return self.model(inputs)