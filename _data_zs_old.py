import configparser
import os.path as osp
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def get_class_num(name):
    r = {"coco": 80, "coco60": 60, "nus": 18, "nus17": 17, "voc": 17}[name]
    return r


def get_topk(name):
    r = {"coco": 1000, "coco60": 1000, "nus": 1000, "nus17": 1000, "voc": 1000}[name]
    return r


def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)


def get_w2vs(name, root, normalize=True):
    # word2vec_300d
    w2vs = np.load(osp.join(root, name, "w2v_list.npy"))
    w2vs = torch.from_numpy(w2vs)
    if normalize:
        w2vs = F.normalize(w2vs)
    return w2vs


def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            # T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loaders(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("other")

    data = init_dataset(name, root)

    train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
    # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    query_loader = DataLoader(ImageDataset(data.query, other_trans), generator=torch.Generator(), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), generator=torch.Generator(), **kwargs)

    return train_loader, query_loader, dbase_loader


class BaseDataset(object):
    """
    Base class of dataset
    """

    def __init__(self, name, idx_root, img_root, verbose=True):

        self.name = name
        self.img_root = img_root

        for x1 in ["train", "query", "dbase"]:
            for x2 in ["txt", "npy"]:  # txt->img_path, npy->label(int8)
                setattr(self, f"{x1}_{x2}", osp.join(idx_root, f"{x1}.{x2}"))

        self.check_before_run()

        for x in ["train", "query", "dbase"]:
            setattr(self, x, self.process(x))

        self.set_img_abspath()  # 1.jpg -> /home/x/COCO/images/1.jpg

        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics()

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        for x1 in ["train", "query", "dbase"]:
            for x2 in ["txt", "npy"]:
                p = getattr(self, f"{x1}_{x2}")
                if not osp.exists(p):
                    raise RuntimeError("'{}' is not available".format(p))

    def get_imagedata_info(self, data):
        labs = data[1]
        n_cids = (labs.sum(axis=0) > 0).sum()
        n_imgs = len(data[0])
        return n_cids, n_imgs

    def print_dataset_statistics(self):
        n_train_cids, n_train_imgs = self.get_imagedata_info(self.train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(self.query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(self.dbase)

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

    def get_n_picks(self, usage):
        # Random select follow:
        # Transductive Zero-Shot Hashing For Multilabel Image Retrieval
        if self.name == "nus" and usage == "train":
            return 10000
        if self.name == "nus17" and usage == "train":
            return 10000
        if self.name == "nus17" and usage == "query":
            return 2000
        if self.name == "coco" and usage == "train":
            return 10000
        if self.name == "coco" and usage == "query":
            return 2000
        if self.name == "coco60" and usage == "query":
            return 1000

        return None

    def process(self, usage):
        txt_path = getattr(self, f"{usage}_txt")
        npy_path = getattr(self, f"{usage}_npy")

        with open(txt_path, "r") as f:
            imgs = [line.strip() for line in f]
        imgs = np.array(imgs)
        labs = np.load(npy_path).astype("float32")

        n_picks = self.get_n_picks(usage="train")
        if n_picks is not None:
            imgs = imgs[:n_picks]
            labs = labs[:n_picks]

        return (imgs.tolist(), labs)

    def set_img_abspath(self):
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)
            imgs = [osp.join(self.img_root, img) for img in imgs]
            setattr(self, x, (imgs, labs))


class NUSWIDE(BaseDataset):

    def __init__(self, name, idx_root, img_root, verbose=True):
        super().__init__(name, idx_root, img_root, verbose)

    def set_img_abspath(self):
        # form img_name:abspath dict
        img_dict = {p.stem: str(p) for p in Path(self.img_root).rglob("*.jpg")}

        # set abspath for image path item
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)
            imgs = [img_dict[img.replace(".jpg", "")] for img in imgs]
            setattr(self, x, (imgs, labs))


class COCO(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def set_img_abspath(self):
        # form img_name:abspath dict
        img_dict = {p.stem.split("_")[-1]: str(p) for p in Path(self.img_root).rglob("*.jpg")}

        # set abspath for image path item
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)
            imgs = [img_dict[img.replace(".jpg", "")] for img in imgs]
            setattr(self, x, (imgs, labs))


_ds_factory = {"nus": NUSWIDE, "nus17": NUSWIDE, "coco": COCO, "coco60": COCO, "voc": BaseDataset}


def init_dataset(name, root, **kwargs):

    if name not in list(_ds_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(_ds_factory.keys())))

    idx_root = osp.join(root, name)

    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)

    return _ds_factory[name](name, idx_root, img_root, **kwargs)


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        img, lab = self.data[0][idx], self.data[1][idx]

        # img path -> img tensor
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(self.data[1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db_name = "voc"
    root = "./_datasets_zs"

    dataset = init_dataset(db_name, root)

    trans = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(dataset.train, trans)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    concepts = get_concepts(db_name, root)

    for imgs, labs, _ in dataloader:
        print(imgs.shape, labs)
        plt.imshow(imgs[0].numpy().transpose(1, 2, 0))
        titles = concepts[labs[0].nonzero().squeeze(1)]
        plt.title(titles)
        plt.show()
        break
