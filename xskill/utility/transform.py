from torchvision import datasets, transforms
import torchvision.transforms as Tr
import torch.nn as nn

TRANSFORMS = {
    "center_crop_110_146":
    Tr.CenterCrop((110, 146)),
    "center_crop_112_112":
    Tr.CenterCrop((112, 112)),
    "random_resized_crop":
    Tr.RandomResizedCrop(size=112, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    "random_resized_crop_216_288":
    Tr.RandomResizedCrop(size=(216, 288),
                         scale=(0.8, 1.0),
                         ratio=(0.75, 1.333)),
    "random_crop_216_288":
    Tr.RandomCrop((216, 288)),
    "random_crop_110_146":
    Tr.RandomCrop((110, 146)),
    "random_crop_112_112":
    Tr.RandomCrop((112, 112)),
    "resize":
    Tr.Resize(112, ),
    "resize_224":
    Tr.Resize(224, ),
    "resize_120_160":
    Tr.Resize((120,160), ),
    "grayscale":
    Tr.RandomGrayscale(p=0.2),
    "horizontal_flip":
    Tr.RandomHorizontalFlip(),
    "vertical_flip":
    Tr.RandomVerticalFlip(),
    "rotate":
    Tr.RandomRotation(degrees=(0, 270)),
    "gaussian_blur":
    Tr.RandomApply([Tr.GaussianBlur(kernel_size=11, sigma=[1, 2])], p=0.2),
    "color_jitter":
    Tr.RandomApply(
        [
            Tr.ColorJitter(
                brightness=0.4, contrast=0.4, hue=0.1, saturation=0.1)
        ],
        p=0.8,
    ),
    "normalize":
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
}


def get_transform_pipeline(pipeline):
    pipeline_transform = [TRANSFORMS[k] for k in pipeline]
    pipeline_transform = nn.Sequential(*pipeline_transform)

    return pipeline_transform