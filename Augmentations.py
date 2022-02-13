from torchvision import datasets, transforms


def augment(image_):
    """

    :param image_: image of size 3xHxW
    :return: a random augmented image of the input image
    """
    aug_transform = transforms.Compose([transforms.ColorJitter(0.8),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomGrayscale(0.2),
                                        transforms.GaussianBlur(kernel_size=3)])

    aug_image_ = aug_transform(image_)
    return aug_image_
