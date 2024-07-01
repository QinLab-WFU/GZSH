import os

import torch

from G_MLZSL.config import get_config
from G_MLZSL.networks.HASH_model import HashNet
from G_MLZSL.networks.VGG_model import FeatNet
from G_MLZSL.util import build_trans, weights_init
from _utils import init
from _utils_zs import init_my_eval


class TestNet(torch.nn.Module):
    def __init__(self, input_dim, n_bits, pretrained=True):
        super(TestNet, self).__init__()
        self.feat_net = FeatNet()
        self.hash_net = HashNet(input_dim, n_bits)
        if pretrained:
            self.hash_net.apply(weights_init)

    def forward(self, x):
        x = self.feat_net(x)
        x = self.hash_net(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        return self.hash_net.load_state_dict(state_dict, **kwargs)


def build_model(args, pretrained=True):
    net = TestNet(args.res_size, args.n_bits, pretrained)
    return net.cuda()


if __name__ == "__main__":
    init("1")

    proj_name = "G_MLZSL"

    # TODO: "TrainingTime"
    evals = ["mAP", "NDCG", "PR-curve", "TopN-precision", "P@H≤2", "EncodingTime", "SelfCheck"]

    datasets = [
        ("nus", "voc"),
        ("voc", "nus"),
        ("nus17", "coco"),
        ("coco", "nus17"),
        ("voc", "coco60"),
        ("coco60", "voc"),
    ]

    hash_bits = [12, 24, 36, 48]

    init_my_eval(get_config, build_model, build_trans)(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        proj_name,
        None,
        evals,
        datasets,
        hash_bits,
        True,
        False,
    )
