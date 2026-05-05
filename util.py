import operator
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torchvision import transforms

from G_MLZSL.networks.VGG_model import FeatNet
from _data_zs import build_loader
from _utils import predict


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DATA_LOADER(object):
    def __init__(self, args, logger):
        tic = time.time()

        # seen attributes & unseen attributes
        seen_atts = np.load(f"{args.data_dir}/{args.seen_ds}/word2vec_300d.npy").astype("float32")
        unseen_atts = np.load(f"{args.data_dir}/{args.unseen_ds}/word2vec_300d.npy").astype("float32")

        # seen features & labels + unseen labels
        feat_net = build_feat_model()
        trans = build_trans()
        train_loader = build_loader(
            args.data_dir,
            args.seen_ds,
            "train",
            trans,
            batch_size=64,
            shuffle=False,
            num_workers=args.n_workers,
            drop_last=False,
        )
        # ret = predict_feat(feat_net, train_loader)
        ret = predict(feat_net, train_loader, -1, False)
        seen_features = ret[0].cpu()
        seen_labels = ret[1].cpu()

        unseen_labels = np.load(f"{args.data_dir}/{args.unseen_ds}/train.npy").astype("float32")

        logger.info("Data loading finished, Time taken: {}".format(time.time() - tic))

        tic = time.time()

        logger.info("attributes are combined in this order -> seen+unseen")
        self.attributes = F.normalize(torch.from_numpy(np.concatenate((seen_atts, unseen_atts), axis=0)))

        logger.info(f"USING SEEN FEATURES WITH N <= {args.N}")
        logger.info(f"before: {seen_labels.shape[0]}")
        temp_idx = sum_fit_idxes(seen_labels, operator.le, args.N)
        if temp_idx is not None:
            seen_labels = seen_labels[temp_idx]
            seen_features = seen_features[temp_idx]
        logger.info(f"after: {seen_labels.shape[0]}")

        # fix: unseen labels is not compatible for FLF_preprocess_att(...)
        # TODO: remove this part from map calc?
        logger.info(f"USING UNSEEN LABELS WITH N <= {args.N}")
        logger.info(f"before: {unseen_labels.shape[0]}")
        # temp_idx = self.unseen_labels.sum(1) <= opt.N
        temp_idx = sum_fit_idxes(unseen_labels, operator.le, args.N)
        if temp_idx is not None:
            unseen_labels = unseen_labels[temp_idx]
        logger.info(f"after: {unseen_labels.shape[0]}")

        if not args.validation:
            if args.preprocessing:
                if args.standardization:
                    logger.info("standardization...")
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                seen_features = scaler.fit_transform(seen_features)
                self.seen_features = torch.from_numpy(seen_features).float()

                mx = self.seen_features.max()
                self.seen_features.mul_(1 / mx)
                self.seen_labels = seen_labels
                self.unseen_labels = torch.from_numpy(unseen_labels)  # .long()
            else:
                self.seen_features = seen_features
                self.seen_labels = seen_labels
                self.unseen_labels = torch.from_numpy(unseen_labels)  # .long()

        self.N = args.N
        self.ntrain = self.seen_features.shape[0]

        logger.info("Data preprocessing finished, Time taken: {}".format(time.time() - tic))

    def ALF_preprocess_att(self, labels, offset=0):
        new_seen_attributes = torch.zeros(labels.shape[0], self.attributes.shape[-1])
        for i in range(len(labels)):
            idx = labels[i].nonzero().flatten()
            if len(idx) == 0:
                continue
            new_seen_attributes[i, :] = torch.mean(self.attributes[idx + offset], 0)
        return new_seen_attributes

    def FLF_preprocess_att(self, labels, offset=0):
        new_attributes = torch.zeros(labels.shape[0], self.N, self.attributes.shape[-1])  # [BS X 10 X 300]
        for i in range(len(labels)):
            idx = labels[i].nonzero().flatten()
            if len(idx) == self.N:
                new_attributes[i, :, :] = self.attributes[idx + offset]
            elif len(idx) < self.N:
                new_attributes[i, :, :] = torch.cat(
                    (self.attributes[idx + offset], torch.zeros((self.N - len(idx)), self.attributes.shape[-1]))
                )
            else:
                raise Exception(f"No. of 1s in label is {len(idx)}, more than {self.N}")
        return new_attributes

    ## Training Dataloader
    def next_train_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_features = self.seen_features[idx]
        batch_labels = self.seen_labels[idx]
        early_fusion_train_batch_att = self.ALF_preprocess_att(batch_labels)
        late_fusion_train_batch_att = self.FLF_preprocess_att(batch_labels)
        return batch_labels, batch_features, late_fusion_train_batch_att, early_fusion_train_batch_att

    ## Testing Dataloader
    def next_test_batch(self, batch_size):
        idx = torch.randperm(len(self.unseen_labels))[0:batch_size]
        batch_labels = self.unseen_labels[idx]
        offset = self.seen_labels.shape[-1]
        early_fusion_test_batch_att = self.ALF_preprocess_att(batch_labels, offset)
        late_fusion_test_batch_att = self.FLF_preprocess_att(batch_labels, offset)
        return batch_labels, late_fusion_test_batch_att, early_fusion_test_batch_att


def build_trans(_=None):
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return trans


def build_feat_model():
    model = FeatNet()
    return model.to(args.device)


# def predict_feat(feat_net, dataloader):
#     device = next(feat_net.parameters()).device
#     feats, clses = [], []
#     feat_net.eval()
#     print(f'predicting({len(dataloader.dataset)})...')
#     for x in dataloader:
#         with torch.no_grad():
#             out = feat_net(x[0].to(device))
#         feats.append(out)
#         clses.append(x[1])
#     return torch.cat(feats), torch.cat(clses).to(device)


def sum_fit_idxes(arr, opt, n):
    # '>': operator.gt
    # '<': operator.lt
    # '>=': operator.ge
    # '<=': operator.le
    # '==': operator.eq
    # '!=': operator.ne
    if type(arr) is np.ndarray:
        idxes = opt(arr.sum(-1), n).nonzero()[0]
    elif type(arr) is torch.Tensor:
        idxes = opt(arr.sum(-1), n).nonzero().flatten()
    else:
        raise Exception(f"unsupported type: {type(arr)}")
    if len(idxes) == arr.shape[0]:
        return None
    return idxes
