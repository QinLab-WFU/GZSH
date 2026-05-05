import datetime
import time
from copy import deepcopy

import torch
from loguru import logger
from timm.utils import AverageMeter
from torch import optim
from torch.utils.data import DataLoader

from G_MLZSL.util import weights_init, build_feat_model, build_trans
from _data_zs import build_loader, SimpleDataset
from _utils import mean_average_precision


class HashNet(torch.nn.Module):
    def __init__(self, input_dim, n_bits):
        super(HashNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, n_bits)

    def forward(self, x):
        o = self.fc1(x)
        # add for hashing
        # z = torch.tanh(o)
        return o


def build_hash_model(args):
    model = HashNet(args.res_size, args.n_bits)
    model.apply(weights_init)
    return model.to(args.device)


def hashing_loss(b, cls, m, alpha=0.01):
    """
    compute hashing loss of DSH
    automatically consider all n^2 pairs
    """
    y = (cls @ cls.T == 0).float()
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=-1)
    loss1 = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)
    loss1 = loss1.mean()
    loss2 = (b.abs() - 1).abs().mean()
    loss = loss1 + alpha * loss2
    return loss


def predict_hash(feat_net, hash_net, dataloader):
    device = next(feat_net.parameters()).device
    codes, clses = [], []
    feat_net.eval()
    hash_net.eval()
    print(f"predicting({len(dataloader.dataset)})...")
    for x in dataloader:
        with torch.no_grad():
            feats = feat_net(x[0].to(device))
            out = hash_net(feats)
        codes.append(out)
        clses.append(x[1])
    return torch.cat(codes).sign(), torch.cat(clses).to(device)


def prepare_loaders(args):
    trans = build_trans()
    query_loader = build_loader(
        args.data_dir,
        args.unseen_ds,
        "query",
        trans,
        batch_size=args.hash_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False,
    )
    dbase_loader = build_loader(
        args.data_dir,
        args.unseen_ds,
        "dbase",
        trans,
        batch_size=args.hash_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False,
    )
    return query_loader, dbase_loader


def train_hash(args, features, labels):
    syn_ds = SimpleDataset(features, labels)

    train_loader = DataLoader(
        syn_ds,
        batch_size=args.hash_batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
    )

    query_loader, dbase_loader = prepare_loaders(args)

    feat_net = build_feat_model()
    hash_net = build_hash_model(args)

    # setup optimizer
    optimizer = optim.Adam(hash_net.parameters(), lr=args.classifier_lr, betas=(args.beta1, 0.999))

    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    epoch_times = []
    for epoch in range(args.hash_n_epochs):
        tic = time.time()
        hash_net.train()
        loss_meter = AverageMeter()
        for feats, labels, _ in train_loader:
            # model.zero_grad()
            optimizer.zero_grad()
            train_outputs = hash_net(feats.cuda())
            loss = hashing_loss(train_outputs, labels.cuda(), 2 * args.n_bits)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        toc = time.time()
        epoch_times.append(toc - tic)
        logger.info(
            f"[Training-HASH][dataset:{args.seen_ds}->{args.unseen_ds}][bits:{args.n_bits}][epoch:{epoch}/{args.hash_n_epochs - 1}][time:{(toc - tic):.3f}][loss:{loss_meter.avg:.4f}]"
        )

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.hash_n_epochs:
            qB, qL = predict_hash(feat_net, hash_net, query_loader)
            rB, rL = predict_hash(feat_net, hash_net, dbase_loader)
            map_k = mean_average_precision(qB, rB, qL, rL, args.topk)
            logger.info(
                f"[Evaluating-HASH][dataset:{args.seen_ds}->{args.unseen_ds}][bits:{args.n_bits}][epoch:{epoch}/{args.hash_n_epochs - 1}][best-mAP@{args.topk}:{best_map:.7f}][mAP@{args.topk}:{map_k:.7f}][count:{0 if map_k > best_map else (count + 1)}]"
            )

            if map_k > best_map:
                best_map = map_k
                best_epoch = epoch
                best_checkpoint = deepcopy(hash_net.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"Without improvement, will save & exit, best mAP: {best_map}, best HASH epoch: {best_epoch}, best model training takes: {datetime.timedelta(seconds=sum(epoch_times[:best_epoch + 1]))}"
                    )
                    # torch.save(best_checkpoint, f"{param['save_dir']}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(
            f"Reach epoch limit, will save & exit, best mAP: {best_map}, best HASH epoch: {best_epoch}, best model training takes: {datetime.timedelta(seconds=sum(epoch_times[:best_epoch + 1]))}"
        )
        # torch.save(best_checkpoint, f"{param['save_dir']}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map, best_checkpoint, sum(epoch_times[: best_epoch + 1])
