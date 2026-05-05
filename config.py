import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument(
        "--preprocessing", action="store_true", default=True, help="enable MinMaxScaler on visual features"
    )
    parser.add_argument("--standardization", action="store_true", default=False)
    parser.add_argument("--n_workers", type=int, help="number of data loading workers", default=4)
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--res_size", type=int, default=4096, help="size of visual features")
    parser.add_argument("--att_size", type=int, default=300, help="size of semantic features")
    parser.add_argument("--ndh", type=int, default=4096, help="size of the hidden units in discriminator")
    parser.add_argument("--n_epochs", type=int, default=70, help="number of epochs to train for")
    parser.add_argument("--critic_iter", type=int, default=5, help="critic iteration, following WGAN-GP")
    parser.add_argument("--lambda1", type=float, default=10, help="gradient penalty regularizer, following WGAN-GP")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate to train GANs ")
    parser.add_argument("--classifier_lr", type=float, default=0.0001, help="learning rate to train softmax classifier")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    # parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument("--manual_seed", type=int, default=3483, help="manual seed")
    parser.add_argument("--n_seen_classes", type=int, default=925, help="number of seen classes")
    parser.add_argument("--n_unseen_classes", type=int, default=81, help="number of unseen classes")
    parser.add_argument("--encoder_layer_sizes1", type=int, default=4096)
    parser.add_argument("--encoder_layer_sizes2", type=int, default=4096)
    parser.add_argument("--decoder_layer_sizes1", type=int, default=4096)
    parser.add_argument("--decoder_layer_sizes2", type=int, default=4096)
    parser.add_argument("--gammaD", type=int, default=10, help="weight on the W-GAN loss")
    parser.add_argument("--gammaG", type=int, default=10, help="weight on the W-GAN loss")
    parser.add_argument("--validation", action="store_true", default=False, help="enable cross validation mode")
    # parser.add_argument('--summary', type=str, default='test dataset type 2 split', help='details regarding the code')
    parser.add_argument("--N", type=int, default=10, help="number of seen and unseen labels per image")
    # parser.add_argument('--syn_num', type=int, default=1, help='number features to generate per class')
    parser.add_argument("--hash_n_epochs", type=int, default=100, help="classifier training epochs")
    parser.add_argument("--hash_batch_size", type=int, default=100, help="classifier training batch size")
    parser.add_argument("--fake_batch_size", type=int, default=300, help="synthesis batch size")

    # parser.add_argument("--late_fusion", action='store_true', default=False)
    # parser.add_argument("--early_fusion", action='store_true', default=False)
    # parser.add_argument("--hybrid_fusion", action='store_true', default=True)
    # parser.add_argument("--trimmed_train", action='store_true', default=True)

    # parser.add_argument('--per_seen', type=float, default=0.10, help='percent of seen classes')
    # parser.add_argument('--per_unseen', type=float, default=0.40, help='percent of unseen classes')
    # parser.add_argument('--per_seen_unseen', type=float, default=0.50, help='percent of seen unseen classes')

    parser.add_argument("--hidden_size", type=int, default=8192)

    # parser.add_argument("--seen_classifier_epoch", type=int, default=25, help="classifier training epochs")
    # parser.add_argument("--seen_classifier_batch_size", type=int, default=100, help="classifier training batch size")
    # parser.add_argument("--train", action="store_true", default=False)

    # parser.add_argument('--val_per_seen', type=float, default=0.10, help='percent of val seen classes')
    # parser.add_argument('--val_syn_num', type=int, default=1, help='number val features to generate per class')
    # parser.add_argument('--test_epoch', type=int, default=1, help='epoch at which to test the model')

    # add for this proj
    parser.add_argument("--seen_ds", type=str, default="nus", help="seen dataset name")
    parser.add_argument("--unseen_ds", type=str, default="voc", help="unseen dataset name")
    parser.add_argument("--n_bits", type=int, default=48, help="hash bit length of the Hash model")
    parser.add_argument("--topk", type=int, default=1000, help="map@topk")
    parser.add_argument("--save_dir", type=str, default="./output", help="output directory")
    parser.add_argument("--data_dir", type=str, default="../_datasets_zs", help="dataset data directory")

    args = parser.parse_args()
    args.encoder_layer_sizes = [args.encoder_layer_sizes1, args.encoder_layer_sizes2]
    args.decoder_layer_sizes = [args.decoder_layer_sizes1, args.decoder_layer_sizes2]
    args.encoder_layer_sizes[0] = args.res_size
    args.decoder_layer_sizes[-1] = args.res_size

    return args
