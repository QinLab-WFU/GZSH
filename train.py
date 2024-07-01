import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from loguru import logger
from timm.utils import AverageMeter

import networks.CLF_model as model
import util
from G_MLZSL.config import get_config
from G_MLZSL.networks.HASH_model import train_hash
from _data_zs import get_topk, get_class_num
from _utils import init


def setup_seed(seed=random.randint(1, 10000)):
    """
    setting up seeds
    """
    print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)#为CPU中设置种子，生成随机数
    torch.cuda.manual_seed(seed)#为特定GPU设置种子，生成随机数
    torch.cuda.manual_seed_all(seed)#为所有GPU设置种子，生成随机数


def build_models(opt):
    """
    MODEL INITIALIZATION
    """
    netE = model.Encoder(opt)
    netG = model.CLF(opt)
    netD = model.Discriminator(opt)

    return netE, netG, netD


def loss_fn(recon_x, x, mean, log_var):
    ## BCE+KL divergence loss
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction="sum")  # size_average=False)
    BCE = BCE.sum() / x.size(0)
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    # Multivariate normal distribution
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return BCE + KLD


def generate_syn_feature(netG, classes, batch_size):
    ## SYNTHESIS MULTI LABEL FEATURES
    nsample = classes.shape[0]
    if nsample % batch_size != 0:
        nsample = nsample + (batch_size - (nsample % batch_size))
    # nclass = classes.shape[1]
    syn_noise = torch.FloatTensor(batch_size, args.att_size)
    syn_feature = torch.FloatTensor(nsample, args.res_size)
    syn_label = torch.LongTensor(nsample, classes.shape[1])
    syn_noise = syn_noise.cuda()
    for k, i in enumerate(range(0, nsample, batch_size)):
        batch_test_labels, late_fusion_test_batch_att, early_fusion_test_batch_att = data.next_test_batch(batch_size)
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, att=late_fusion_test_batch_att.cuda(), avg_att=early_fusion_test_batch_att.cuda())
        syn_feature.narrow(0, k * batch_size, batch_size).copy_(output)
        syn_label.narrow(0, k * batch_size, batch_size).copy_(batch_test_labels)
    return syn_feature, syn_label


def calc_gradient_penalty(netD, real_data, fake_data, input_att=None):
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates.requires_grad = True
    if input_att is None:
        disc_interpolates = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates, att=input_att)
    ones = torch.ones(disc_interpolates.size())
    ones = ones.cuda()
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambda1
    return gradient_penalty


def train_val(args, data, logger):
    netE, netG, netD = build_models(args)

    # init tensors
    noise = torch.FloatTensor(args.batch_size, args.att_size)#（64，300）
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1

    netE.cuda()
    netG.cuda()
    netD.cuda()
    noise = noise.cuda()
    one = one.cuda()
    mone = mone.cuda()

    # setup optimizer
    optimizerE = optim.Adam(netE.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # training loop
    best_epoch_vgan = 0
    best_epoch_hash = 0
    best_map = 0.0
    # best_checkpoint_D = None
    # best_checkpoint_G = None
    # best_checkpoint_E = None
    best_checkpoint_H = None
    count = 0
    epoch_times = []
    best_hash_time = 0
    for epoch in range(args.n_epochs):
        lossD_meter = AverageMeter()#管理需要用到的变量
        lossG_meter = AverageMeter()
        lossE_meter = AverageMeter()
        tic1 = time.time()
        for i in range(0, data.ntrain, args.batch_size):
            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in generator update

            for iter_d in range(args.critic_iter):  # 5
                xxx = data.next_train_batch(args.batch_size)

                # batch_labels = xxx[0].cuda()
                batch_features = xxx[1].cuda()
                late_fusion_train_batch_att = xxx[2].cuda()
                early_fusion_train_batch_att = xxx[3].cuda()

                for param in netD.parameters():
                    param.grad = None
                criticD_real = netD(batch_features, att=early_fusion_train_batch_att)
                criticD_real = args.gammaD * criticD_real.mean()
                # The -1 is multiplied to the gradient, so the loss term contains it negatively.
                # The 1 is probably not needed, but we all copied it from the DCGAN in the pytorch examples or the WGAN code.
                criticD_real.backward(mone)

                noise.normal_(0, 1)#用于生成假数据的噪声
                fake = netG(noise, att=late_fusion_train_batch_att, avg_att=early_fusion_train_batch_att)
                criticD_fake = netD(fake.detach(), att=early_fusion_train_batch_att)
                criticD_fake = args.gammaD * criticD_fake.mean()
                criticD_fake.backward(one)

                gradient_penalty = args.gammaD * calc_gradient_penalty(
                    netD, batch_features, fake.data, early_fusion_train_batch_att
                )
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
                lossD_meter.update(D_cost.item())

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in netD.parameters():
                p.requires_grad = False

            for param in netE.parameters():
                param.grad = None
            for param in netG.parameters():
                param.grad = None

            # mean=均值, log_var=log(方差), std=标准差
            means, log_var = netE(batch_features, att=early_fusion_train_batch_att)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([args.batch_size, args.att_size])
            eps = eps.cuda()
            z = eps * std + means

            recon_x = netG(z, att=late_fusion_train_batch_att, avg_att=early_fusion_train_batch_att)
            vae_loss_seen = loss_fn(recon_x, batch_features, means, log_var)
            lossE_meter.update(vae_loss_seen.item())
            errG = vae_loss_seen

            noise.normal_(0, 1)
            fake = netG(noise, att=late_fusion_train_batch_att, avg_att=early_fusion_train_batch_att)
            criticG_fake = netD(fake, att=early_fusion_train_batch_att).mean()
            G_cost = -criticG_fake
            errG += args.gammaG * G_cost
            lossG_meter.update(G_cost.item())

            errG.backward()
            optimizerE.step()
            optimizerG.step()

        # logger.info('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Loss_E: %.4f, Wasserstein_dist: %.4f' %
        #             (epoch, opt.n_epochs - 1, mean_lossD, mean_lossG, mean_lossE, Wasserstein_D.item()))
        # logger.info("Generator {}th finished time taken {}".format(epoch + 1, time.time() - tic))
        toc1 = time.time()
        epoch_times.append(toc1 - tic1)
        logger.info(
            f"[Training-GAN][dataset:{args.seen_ds}->{args.unseen_ds}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{toc1 - tic1:.3f}][D-loss:{lossD_meter.avg:.4f}][G-loss:{lossG_meter.avg:.4f}][E-loss:{lossE_meter.avg:.4f}][Wasserstein_D:{Wasserstein_D.item():.4f}]"
        )

        tic2 = toc1
        netG.eval()
        syn_feats, syn_labels = generate_syn_feature(netG, data.unseen_labels, args.fake_batch_size)

        # cat syn_features & labels with seen_feats & seen_labels
        all_feats = torch.cat((syn_feats, data.seen_features), 0)

        # all_labels = torch.cat((syn_labels, data.seen_labels), 0)
        all_labels = []
        # ADDING ONLY SEEN LABELS
        seen_labels = torch.zeros(data.seen_labels.shape[0], data.attributes.shape[0])
        seen_labels[:, : args.n_seen_classes] = data.seen_labels
        all_labels.append(seen_labels)
        # ADDING ONLY UNSEEN LABELS
        unseen_labels = torch.zeros(syn_labels.shape[0], data.attributes.shape[0])
        unseen_labels[:, args.n_seen_classes :] = syn_labels
        all_labels.append(unseen_labels)

        all_labels = torch.cat(all_labels)
        toc2 = time.time()
        epoch_times[-1] += toc2 - tic2

        epoch_hash, map_hash, checkpoint_hash, elapsed_secs = train_hash(args, all_feats, all_labels)
        if map_hash > best_map:
            best_map = map_hash
            best_epoch_vgan = epoch
            best_epoch_hash = epoch_hash
            best_checkpoint_H = checkpoint_hash
            best_hash_time = elapsed_secs
            # best_checkpoint_D = deepcopy(netD.state_dict())
            # best_checkpoint_G = deepcopy(netG.state_dict())
            # best_checkpoint_E = deepcopy(netE.state_dict())
        else:
            count += 1
            if count == 10:
                logger.info(
                    f"without improvement, will save & exit, best mAP: {best_map}, best GAN epoch: {best_epoch_vgan}, best model training takes: {datetime.timedelta(seconds=sum(epoch_times[:best_epoch_vgan + 1]) + best_hash_time)}"
                )
                # save_dict = {
                #     "map": best_map,
                #     "epoch_vgan": best_epoch_vgan,
                #     "epoch_hash": best_epoch_hash,
                #     "netD": best_checkpoint_D,
                #     "netG": best_checkpoint_G,
                #     "netE": best_checkpoint_E,
                #     "netH": best_checkpoint_H
                # }
                torch.save(best_checkpoint_H, f"{args.save_dir}/e{best_epoch_vgan}_{best_map:.3f}.pkl")
                break
        # logger.info("Hashing finished time taken: {}".format(time.time() - tic))

        # reset G to training mode
        netG.train()

    if count != 10:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {best_map}, best GAN epoch: {best_epoch_vgan}, best model training takes: {datetime.timedelta(seconds=sum(epoch_times[:best_epoch_vgan + 1] + best_hash_time))}"
        )
        # save_dict = {
        #     "map": best_map,
        #     "epoch_vgan": best_epoch_vgan,
        #     "epoch_hash": best_epoch_hash,
        #     "netD": best_checkpoint_D,
        #     "netG": best_checkpoint_G,
        #     "netE": best_checkpoint_E,
        #     "netH": best_checkpoint_H
        # }
        torch.save(best_checkpoint_H, f"{args.save_dir}/e{best_epoch_vgan}_{best_map:.3f}.pkl")

    # logger.info("Total time taken: {}".format(time.time() - tic1))

    return best_epoch_vgan, best_epoch_hash, best_map


if __name__ == "__main__":
    init("0")

    torch.set_default_tensor_type("torch.FloatTensor")#用于设置默认的张量

    args = get_config()

    setup_seed(args.manual_seed)

    dummy_logger_id = None
    rst = []
    for seen_ds, unseen_ds in [
        ("nus", "voc"),
        ("voc", "nus"),
        ("nus17", "coco"),
        ("coco", "nus17"),
        ("voc", "coco60"),
        ("coco60", "voc"),
    ]:
        print(f"processing dataset: {seen_ds}->{unseen_ds}")
        args.seen_ds = seen_ds
        args.unseen_ds = unseen_ds
        args.topk = get_topk(unseen_ds)
        # opt.feat_label_path = f"./datasets/extract_features/{opt.seen_ds}_{opt.unseen_ds}_vgg19.h5"

        args.n_seen_classes = get_class_num(args.seen_ds)  # may shrink caused by miss attributes (done in DATA_LOADER)
        args.n_unseen_classes = get_class_num(
            args.unseen_ds
        )  # may shrink caused by miss attributes (done in DATA_LOADER)

        # calling the dataloader
        data = util.DATA_LOADER(args, logger)
        logger.info(f"training samples: {data.ntrain}")

        for hash_bit in [12, 24, 36, 48]:
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            args.save_dir = f"./output/{seen_ds}_{unseen_ds}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=False)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", rotation="500 MB", level="INFO")

            with open(f"{args.save_dir}/config.json", "w+") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch_vgan, best_epoch_hash, best_map = train_val(args, data, logger)
            rst.append(
                {
                    "dataset": f"{seen_ds}->{unseen_ds}",
                    "hash_bit": hash_bit,
                    "best_epoch_vgan": best_epoch_vgan,
                    "best_epoch_hash": best_epoch_hash,
                    "best_map": best_map,
                }
            )
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch-vgan:{x['best_epoch_vgan']}][best-epoch-hash:{x['best_epoch_hash']}][best-mAP:{x['best_map']:.3f}]"
        )
