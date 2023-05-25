import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class _RegionSplicerNetBase(nn.Module):
    """
    Base class for spliced region prediction.
    """

    def __init__(
        self,
        encoder="resnet18",
        pretrained=True,
        dims=[512, 512, 512, 512, 512, 512, 512, 512, 128],
        num_class=6,
        freeze_layers=False,
    ):

        """

        :encoder[str]: - fetches model from torchvision
        :pretrained[Bool]: - if True uses pretrained model
        :dims: [list]: - dimension of projection head
        :num_class[int]: - number of classes in the output layer
        :freeze_layers[Bool]: - if True freezes the encoder layers until layer2.
        """

        super().__init__()
        self.encoder = getattr(models, encoder)(pretrained=pretrained)
        if freeze_layers:
            # freezing specific to layer input and 1st layer of resnet18
            self.freeze("layer2")

        if (not pretrained) and (not freeze_layers):
            print("randomizing weights")
            self.encoder.apply(self.rand_init)

        last_layer = list(self.encoder.named_modules())[-1][0].split(".")[0]
        setattr(self.encoder, last_layer, nn.Identity())

        proj_layers = []
        for d in dims[:-1]:
            proj_layers.append(nn.Linear(d, d, bias=False)),
            proj_layers.append((nn.BatchNorm1d(d))),
            proj_layers.append(nn.ReLU(inplace=True))

        embeds = nn.Linear(dims[-2], dims[-1], bias=int(num_class) > 0)
        proj_layers.append(embeds)
        self.head = nn.Sequential(*proj_layers)
        self.out = nn.Linear(dims[-1], int(num_class))

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits, embeds

    def freeze(self, layer_name):
        """freeze encoder until layer_name"""
        for i, (param_name, param) in enumerate(self.encoder.named_parameters()):
            if not param_name.startswith(layer_name):
                param.requires_grad = False
            else:
                break

    def rand_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.randn(m.weight.size()) * 0.01
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.weight.data = torch.randn(m.weight.size()) * 0.01
            m.bias.data.fill_(0)

    def create_graph_model(self):
        return create_feature_extractor(model=self, return_nodes=["head", "out"])


class RegionSplicerNet(_RegionSplicerNetBase):
    """
    RegionSplicerNet class for spliced region prediction.
    """

    def __init__(
        self,
        encoder="resnet18",
        pretrained=True,
        dims=[512, 512, 512, 512, 512, 512, 512, 512, 128],
        num_class=6,
        freeze_layers=False,
    ):
        super().__init__(encoder, pretrained, dims, num_class, freeze_layers)
        return

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits, embeds, features
