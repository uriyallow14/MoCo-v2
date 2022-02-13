import torch
from torchvision import datasets, transforms
from os.path import join


def get_loaders(data_path, batch_size, shuffle=True):
    """

    :param data_path: path to the directory in which we have the train and validation files
    :param batch_size: the size of the train and validation batch
    :return: train and validation images as Torch DataLoaders
    """
    transform_ = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2023, 0.1994, 0.2010])]),
                  "val": transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                  [0.2023, 0.1994, 0.2010])])}

    ds_train = datasets.ImageFolder(join(data_path, 'train'), transform=transform_["train"])
    ds_val = datasets.ImageFolder(join(data_path, 'val'), transform=transform_["val"])

    train_size = len(ds_train)
    moco_train_size = int(0.8 * train_size)
    ds_train_moco, ds_train_clf = torch.utils.data.random_split(ds_train, [moco_train_size, train_size-moco_train_size])

    val_size = len(ds_val)
    moco_val_size = int(0.8 * val_size)
    ds_val_moco, ds_val_clf = torch.utils.data.random_split(ds_val, [moco_val_size, val_size - moco_val_size])

    print(f"Training moco dataset has {len(ds_train_moco)} images with a total of {len(ds_train_moco) // batch_size + 1} batches")
    print(f"Testing moco dataset has {len(ds_val_moco)} images with a total of {len(ds_val_moco) // batch_size + 1} batches")
    print(f"Training clf dataset has {len(ds_train_clf)} images with a total of {len(ds_train_clf) // batch_size + 1} batches")
    print(f"Testing clf dataset has {len(ds_val_clf)} images with a total of {len(ds_val_clf) // batch_size + 1} batches")

    # Set num_workers = 0 if this causes crashes on your machine (it's dependent on how many subcores you have available)
    dl_train_moco = torch.utils.data.DataLoader(ds_train_moco, batch_size, shuffle=shuffle, num_workers=0)
    dl_val_moco = torch.utils.data.DataLoader(ds_val_moco, batch_size, shuffle=shuffle, num_workers=0)

    dl_train_clf = torch.utils.data.DataLoader(ds_train_clf, batch_size, shuffle=shuffle, num_workers=0)
    dl_val_clf = torch.utils.data.DataLoader(ds_val_clf, batch_size, shuffle=shuffle, num_workers=0)

    return dl_train_moco, dl_val_moco, dl_train_clf, dl_val_clf
